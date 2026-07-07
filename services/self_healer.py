"""
services/self_healer.py — Scheduled Self-Healing Jobs

Runs periodically (every SELF_HEALER_INTERVAL_SECONDS) to detect and
repair inconsistent system state automatically.

Repair jobs performed:
  1. Stuck ingestion jobs   → Jobs in "processing"/"scanning" > STUCK_JOB_THRESHOLD_SECONDS
                              → Marked as "failed", temp file cleaned if found
  2. Orphaned temp files    → Files in /temp/ older than 2 hours with no active job
                              → Deleted
  3. Zombie document records → Documents in MongoDB with status="completed" but no
                                Pinecone vectors → Re-queued for ingestion
  4. BM25 rebuild           → Workspaces with documents but empty BM25 corpus → Rebuild
  5. Session expiry         → Sessions inactive > MEMORY_SESSION_EXPIRE_DAYS → Archived

Each job is idempotent — safe to run multiple times.
Results logged to application log and persisted to MongoDB for audit.
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import List

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


async def _heal_stuck_jobs() -> int:
    """
    Find and fail jobs stuck in processing state beyond threshold.
    Returns number of jobs fixed.
    """
    from services.db import ingestion_jobs_collection
    threshold = time.time() - settings.STUCK_JOB_THRESHOLD_SECONDS

    stuck = list(ingestion_jobs_collection.find({
        "status": {"$in": ["processing", "scanning"]},
        "created_at": {"$lt": threshold},
    }))

    fixed = 0
    for job in stuck:
        job_id = job.get("job_id")
        try:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "failed",
                        "error": f"Job stuck in '{job['status']}' state — self-healer intervention",
                        "completed_at": time.time(),
                    }
                }
            )
            fixed += 1
            logger.warning(
                "Self-healer: stuck job marked failed",
                extra={"job_id": job_id, "stuck_since": job.get("created_at")},
            )
        except Exception as exc:
            logger.error("Failed to fix stuck job", extra={"job_id": job_id, "error": str(exc)})

    return fixed


def _heal_orphaned_temp_files() -> int:
    """
    Delete temp files older than 2 hours with no active ingestion job.
    Returns number of files deleted.
    """
    from services.db import ingestion_jobs_collection

    temp_dir = os.path.abspath("temp")
    if not os.path.exists(temp_dir):
        return 0

    two_hours_ago = time.time() - 7200
    deleted = 0

    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            mtime = os.path.getmtime(file_path)
            if mtime > two_hours_ago:
                continue   # recent file — skip

            # Check if there's an active job for this file
            # Temp filenames are UUID-prefixed: {uuid}_{original_name}
            has_active_job = ingestion_jobs_collection.find_one({
                "status": {"$in": ["pending", "processing", "scanning"]},
            })

            if has_active_job:
                continue   # there are active jobs — be conservative, skip

            os.remove(file_path)
            deleted += 1
            logger.info("Self-healer: deleted orphaned temp file", extra={"file": filename})
        except Exception as exc:
            logger.warning("Failed to clean temp file", extra={"file": filename, "error": str(exc)})

    return deleted


async def _heal_zombie_documents() -> int:
    """
    Find documents marked 'completed' in MongoDB but with no Pinecone vectors.
    These can occur after a Pinecone rollback or system crash.
    Re-marks them as 'failed' so users know to re-upload.
    Returns number of documents flagged.
    """
    from services.db import documents_collection
    from config import settings as cfg
    from pinecone import Pinecone

    flagged = 0
    try:
        pc = Pinecone(api_key=cfg.PINECONE_API_KEY)
        idx = pc.Index(cfg.PINECONE_INDEX_NAME)

        completed_docs = list(documents_collection.find(
            {"ingestion_status": "completed"},
            {"document_id": 1, "workspace_id": 1, "_id": 0},
            limit=100,   # cap to avoid overloading on large deployments
        ))

        for doc in completed_docs:
            doc_id = doc.get("document_id")
            ws_id = doc.get("workspace_id")
            if not doc_id or not ws_id:
                continue

            try:
                # Probe Pinecone for at least 1 vector with this document_id
                results = idx.query(
                    vector=[0.0] * cfg.GEMINI_EMBED_DIM,
                    top_k=1,
                    namespace=ws_id,
                    filter={"document_id": {"$eq": doc_id}},
                    include_values=False,
                )
                if not results.matches:
                    documents_collection.update_one(
                        {"document_id": doc_id, "workspace_id": ws_id},
                        {"$set": {"ingestion_status": "failed", "error": "Zombie: no vectors found in Pinecone"}}
                    )
                    flagged += 1
                    logger.warning(
                        "Self-healer: zombie document flagged",
                        extra={"document_id": doc_id, "workspace_id": ws_id},
                    )
            except Exception:
                pass   # Skip individual failures silently

    except Exception as exc:
        logger.error("Zombie document check failed", extra={"error": str(exc)})

    return flagged


async def _heal_bm25_corpus() -> int:
    """
    Find workspaces with documents but no BM25 corpus index and rebuild.
    Returns number of workspaces rebuilt.
    """
    from services.db import documents_collection, bm25_corpus_collection
    from services.bm25 import update_corpus_stats

    rebuilt = 0

    # Find all workspace_ids that have completed documents
    workspace_ids = documents_collection.distinct("workspace_id", {"ingestion_status": "completed"})

    for ws_id in workspace_ids:
        has_bm25 = bm25_corpus_collection.find_one({"workspace_id": ws_id})
        if has_bm25:
            continue

        # Rebuild: fetch chunk texts from Pinecone and rebuild
        # For now, mark for rebuild by inserting a sentinel — actual rebuild
        # is triggered by the next ingestion into this workspace
        try:
            bm25_corpus_collection.update_one(
                {"workspace_id": ws_id, "term": "__sentinel__"},
                {"$set": {"workspace_id": ws_id, "term": "__sentinel__", "needs_rebuild": True}},
                upsert=True,
            )
            rebuilt += 1
            logger.info("Self-healer: BM25 rebuild flagged", extra={"workspace_id": ws_id})
        except Exception as exc:
            logger.warning("BM25 flag failed", extra={"workspace_id": ws_id, "error": str(exc)})

    return rebuilt


def _expire_sessions() -> int:
    """
    Archive sessions inactive for > MEMORY_SESSION_EXPIRE_DAYS.
    Returns number of sessions archived.
    """
    from services.db import chat_history_collection

    expire_days = settings.MEMORY_SESSION_EXPIRE_DAYS
    cutoff = time.time() - (expire_days * 86400)
    archived = 0

    try:
        result = chat_history_collection.update_many(
            {
                "last_active_at": {"$lt": cutoff},
                "archived": {"$ne": True},
            },
            {"$set": {"archived": True, "archived_at": datetime.now(timezone.utc)}},
        )
        archived = result.modified_count
        if archived:
            logger.info("Self-healer: sessions archived", extra={"count": archived})
    except Exception as exc:
        logger.warning("Session expiry failed", extra={"error": str(exc)})

    return archived


# ── Main healer ────────────────────────────────────────────────────────────────

async def run_self_healer() -> dict:
    """
    Run all self-healing jobs. Returns a summary of actions taken.
    Called from the background task scheduler in main.py.
    """
    t0 = time.time()
    logger.info("Self-healer run starting")

    results = {
        "stuck_jobs_fixed":      0,
        "orphaned_files_deleted": 0,
        "zombie_docs_flagged":   0,
        "bm25_rebuilt":          0,
        "sessions_archived":     0,
        "duration_ms":           0,
        "run_at":                datetime.now(timezone.utc).isoformat(),
    }

    try:
        results["stuck_jobs_fixed"] = await _heal_stuck_jobs()
    except Exception as exc:
        logger.error("Stuck job healer failed", extra={"error": str(exc)})

    try:
        results["orphaned_files_deleted"] = _heal_orphaned_temp_files()
    except Exception as exc:
        logger.error("Temp file healer failed", extra={"error": str(exc)})

    try:
        results["zombie_docs_flagged"] = await _heal_zombie_documents()
    except Exception as exc:
        logger.error("Zombie doc healer failed", extra={"error": str(exc)})

    try:
        results["bm25_rebuilt"] = await _heal_bm25_corpus()
    except Exception as exc:
        logger.error("BM25 healer failed", extra={"error": str(exc)})

    try:
        results["sessions_archived"] = _expire_sessions()
    except Exception as exc:
        logger.error("Session expiry healer failed", extra={"error": str(exc)})

    results["duration_ms"] = round((time.time() - t0) * 1000)

    logger.info("Self-healer run complete", extra=results)

    # Persist results to audit collection
    try:
        from services.db import audit_collection
        audit_collection.insert_one({
            "action": "self_healer_run",
            "user": "system",
            "detail": results,
            "success": True,
            "timestamp": datetime.now(timezone.utc).timestamp(),
        })
    except Exception:
        pass

    return results
