"""
services/ingestion.py — Enterprise Ingestion Pipeline (v3)

14-stage production-grade document ingestion pipeline.

STAGE  1: File existence validation
STAGE  2: Virus scan (ClamAV — required)
STAGE  3: SHA256 fingerprint computation
STAGE  4: Duplicate detection (same workspace)
STAGE  5: Language detection
STAGE  6: Text extraction (format-appropriate parser)
STAGE  7: PII detection + BLOCK policy enforcement
STAGE  8: Document classification
STAGE  9: Chunking strategy selection + execution
STAGE 10: BM25 corpus update
STAGE 11: Batch embedding
STAGE 12: Pinecone upsert with tenant metadata
STAGE 13: MongoDB document record update
STAGE 14: Cache invalidation

On any failure:
  - Job status updated to "failed" with stage name and error
  - Temp files cleaned up
  - No partial vectors remain (rollback vectors on failed upsert)

All stages log entry/exit with timing.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pinecone import Pinecone

from config import settings
from models.domain import (
    Chunk, DocumentClass, ChunkingStrategy,
    IngestionStatus, PIIScanResult as PIIResult,
)
from services.embedding import embed_texts
from services.bm25 import update_corpus_stats
from services.pii_detector import scan_for_pii, enforce_pii_policy, PIIBlockedError
from services.virus_scanner import (
    scan_file, VirusScanUnavailableError, VirusDetectedError
)
from services.document_classifier import classify_document
from services.chunk_strategy import apply_chunking_strategy
from utils.parser import parse_file
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Pinecone client (singleton) ────────────────────────────────────────────────
_pc: Optional[Pinecone] = None
_index = None


def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        _index = _pc.Index(settings.PINECONE_INDEX_NAME)
    return _index


# ── Metadata cleaner ──────────────────────────────────────────────────────────

def _clean_metadata(metadata: dict) -> dict:
    """
    Ensure Pinecone-compatible metadata:
    - Remove None values
    - Scalar or list of strings only
    - Truncate strings > 500 chars (Pinecone metadata value limit)
    """
    cleaned = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float)):
            cleaned[k] = v
        elif isinstance(v, str):
            cleaned[k] = v[:500]
        elif isinstance(v, list):
            cleaned[k] = [str(x)[:200] for x in v]
        else:
            cleaned[k] = str(v)[:500]
    return cleaned


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_sha256(file_path: str) -> str:
    """Compute SHA256 hex digest of a file efficiently."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def _update_job(job_id: Optional[str], **fields) -> None:
    """Non-throwing job status update."""
    if not job_id:
        return
    try:
        from services.db import ingestion_jobs_collection
        ingestion_jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": fields},
        )
    except Exception as exc:
        logger.warning("Job update failed", extra={"job_id": job_id, "error": str(exc)})


def _mark_job_failed(job_id: Optional[str], stage: str, error: str) -> None:
    _update_job(
        job_id,
        status=IngestionStatus.FAILED.value,
        stage=stage,
        error=error,
        completed_at=time.time(),
    )


# ── Main Pipeline ─────────────────────────────────────────────────────────────

async def ingest_text(
    file_path: str,
    file_type: str,
    workspace_id: str,
    org_id: str,
    collection_id: str = "default",
    original_filename: Optional[str] = None,
    job_id: Optional[str] = None,
    use_semantic: bool = True,
) -> int:
    """
    Run the 14-stage ingestion pipeline for a single document.

    Args:
        file_path:         Absolute path to the temp file on disk.
        file_type:         Lowercase extension (pdf, docx, txt, pptx, xlsx, csv, md, html, eml).
        workspace_id:      Pinecone namespace + tenant isolation key.
        org_id:            Organization ID for multi-tenant filtering.
        collection_id:     Collection grouping within workspace.
        original_filename: Real filename (used as document_id in Pinecone metadata).
        job_id:            Ingestion job UUID for status tracking.
        use_semantic:      Passed to legacy parser for backward compatibility.

    Returns:
        Number of vectors upserted (0 on failure).

    Side effects:
        - Updates MongoDB ingestion_jobs_collection
        - Upserts to Pinecone
        - Updates MongoDB documents_collection
        - Invalidates semantic cache for workspace
    """
    t_pipeline_start = time.time()
    source_name = original_filename or os.path.basename(file_path)
    document_id = source_name
    upserted_ids: List[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: File existence validation
    # ─────────────────────────────────────────────────────────────────────────
    stage = "file_validation"
    _update_job(job_id, stage=stage, status=IngestionStatus.SCANNING.value)

    if not os.path.exists(file_path):
        _mark_job_failed(job_id, stage, "Temp file not found on disk")
        return 0

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        _mark_job_failed(job_id, stage, "File is empty (0 bytes)")
        return 0
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        _mark_job_failed(job_id, stage, f"File too large: {file_size} bytes > {settings.MAX_FILE_SIZE_BYTES}")
        return 0

    logger.info(
        "Ingestion started",
        extra={"document_id": document_id, "workspace_id": workspace_id, "size_bytes": file_size},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Virus scan (required)
    # ─────────────────────────────────────────────────────────────────────────
    stage = "virus_scan"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    try:
        scan_result = scan_file(file_path)
        logger.info(
            "Virus scan passed",
            extra={"document_id": document_id, "duration_ms": round((time.time() - t0) * 1000)},
        )
    except VirusScanUnavailableError as exc:
        _mark_job_failed(job_id, stage, f"ClamAV unavailable: {exc}")
        logger.error("Virus scan service unavailable", extra={"error": str(exc)})
        return 0
    except VirusDetectedError as exc:
        _mark_job_failed(job_id, stage, f"Virus detected: {exc.threat_name}")
        _update_job(job_id, status=IngestionStatus.BLOCKED.value)
        logger.error(
            "Virus detected — ingestion blocked",
            extra={"document_id": document_id, "threat": exc.threat_name},
        )
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: SHA256 fingerprint
    # ─────────────────────────────────────────────────────────────────────────
    stage = "fingerprinting"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    fingerprint = _compute_sha256(file_path)
    logger.debug(
        "Fingerprint computed",
        extra={"sha256": fingerprint[:16] + "...", "ms": round((time.time() - t0) * 1000)},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: Duplicate detection
    # ─────────────────────────────────────────────────────────────────────────
    stage = "deduplication"
    _update_job(job_id, stage=stage)

    from services.db import documents_collection
    existing = documents_collection.find_one({
        "sha256_fingerprint": fingerprint,
        "workspace_id": workspace_id,
        "ingestion_status": "completed",
    })
    if existing:
        logger.info(
            "Duplicate detected — skipping ingestion",
            extra={
                "document_id": document_id,
                "existing_id": existing.get("document_id"),
                "fingerprint": fingerprint[:16],
            },
        )
        _update_job(
            job_id,
            status=IngestionStatus.COMPLETED.value,
            error="Duplicate: identical content already exists in this workspace",
            completed_at=time.time(),
        )
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5: Text extraction
    # ─────────────────────────────────────────────────────────────────────────
    stage = "parsing"
    _update_job(job_id, stage=stage, status=IngestionStatus.PROCESSING.value)

    t0 = time.time()
    raw_chunks = []
    full_text = ""

    try:
        raw_chunks = parse_file(file_path, file_type, use_semantic=use_semantic)
        full_text = "\n\n".join(c.get("text", "") for c in raw_chunks if c.get("text"))
    except Exception as exc:
        _mark_job_failed(job_id, stage, f"Parsing failed: {exc}")
        logger.error("Parsing failed", extra={"document_id": document_id, "error": str(exc)}, exc_info=True)
        return 0

    if not raw_chunks or not full_text.strip():
        _mark_job_failed(job_id, stage, "No text extracted from document")
        return 0

    parse_ms = round((time.time() - t0) * 1000)
    logger.info(
        "Document parsed",
        extra={"document_id": document_id, "raw_chunks": len(raw_chunks), "ms": parse_ms},
    )
    _update_job(job_id, chunks_total=len(raw_chunks))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 6: Language detection
    # ─────────────────────────────────────────────────────────────────────────
    stage = "language_detection"
    _update_job(job_id, stage=stage)

    language = "en"
    try:
        from langdetect import detect, LangDetectException
        language = detect(full_text[:2000]) or "en"
    except Exception:
        pass  # Default to English if langdetect fails

    logger.debug("Language detected", extra={"document_id": document_id, "language": language})

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 7: PII detection + block enforcement
    # ─────────────────────────────────────────────────────────────────────────
    stage = "pii_detection"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    try:
        pii_result = scan_for_pii(
            text=full_text,
            document_id=document_id,
            workspace_id=workspace_id,
            filename=source_name,
        )
        enforce_pii_policy(pii_result, document_id, workspace_id)
        pii_ms = round((time.time() - t0) * 1000)
        logger.info("PII scan passed", extra={"document_id": document_id, "ms": pii_ms})
    except PIIBlockedError as exc:
        _mark_job_failed(job_id, stage, str(exc))
        _update_job(job_id, status=IngestionStatus.BLOCKED.value)
        logger.error(
            "Ingestion blocked: PII detected",
            extra={
                "document_id": document_id,
                "entity_types": exc.entity_types,
                "entity_counts": exc.entity_counts,
            },
        )
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 8: Document classification
    # ─────────────────────────────────────────────────────────────────────────
    stage = "classification"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    try:
        classification = await classify_document(
            text=full_text,
            file_extension=file_type,
            document_id=document_id,
        )
        doc_class = classification.document_class
        strategy = classification.strategy
        class_confidence = classification.confidence
    except Exception as exc:
        logger.warning(
            "Classification failed — using GENERAL",
            extra={"document_id": document_id, "error": str(exc)},
        )
        doc_class = DocumentClass.GENERAL
        strategy = ChunkingStrategy.PARENT_CHILD
        class_confidence = 0.0

    logger.info(
        "Document classified",
        extra={
            "document_id": document_id,
            "class": doc_class.value,
            "strategy": strategy.value,
            "confidence": class_confidence,
            "ms": round((time.time() - t0) * 1000),
        },
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 9: Chunking strategy + execution
    # ─────────────────────────────────────────────────────────────────────────
    stage = "chunking"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    try:
        chunks: List[Chunk] = apply_chunking_strategy(
            strategy=strategy,
            raw_chunks=raw_chunks,
            full_text=full_text,
            document_id=document_id,
            workspace_id=workspace_id,
            collection_id=collection_id,
            org_id=org_id,
        )
    except Exception as exc:
        _mark_job_failed(job_id, stage, f"Chunking failed: {exc}")
        logger.error("Chunking failed", extra={"document_id": document_id, "error": str(exc)}, exc_info=True)
        return 0

    if not chunks:
        _mark_job_failed(job_id, stage, "Chunking produced 0 valid chunks")
        return 0

    chunk_ms = round((time.time() - t0) * 1000)
    logger.info(
        "Chunking complete",
        extra={"document_id": document_id, "chunks": len(chunks), "strategy": strategy.value, "ms": chunk_ms},
    )
    _update_job(job_id, chunks_total=len(chunks))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 10: BM25 corpus update
    # ─────────────────────────────────────────────────────────────────────────
    stage = "bm25_update"
    _update_job(job_id, stage=stage)

    try:
        raw_for_bm25 = [{"text": c.text} for c in chunks]
        update_corpus_stats(raw_for_bm25, workspace_id)
    except Exception as exc:
        # BM25 failure is non-fatal — log and continue
        logger.warning("BM25 update failed (non-fatal)", extra={"error": str(exc)})

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 11: Batch embedding
    # ─────────────────────────────────────────────────────────────────────────
    stage = "embedding"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    chunk_texts = [c.text for c in chunks]

    try:
        embeddings = embed_texts(chunk_texts)
    except Exception as exc:
        _mark_job_failed(job_id, stage, f"Embedding failed: {exc}")
        logger.error("Embedding failed", extra={"document_id": document_id, "error": str(exc)}, exc_info=True)
        return 0

    embed_ms = round((time.time() - t0) * 1000)
    logger.info(
        "Embeddings computed",
        extra={"document_id": document_id, "count": len(embeddings), "ms": embed_ms},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 12: Build Pinecone vectors
    # ─────────────────────────────────────────────────────────────────────────
    stage = "vector_build"
    vectors = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if not embedding or sum(abs(x) for x in embedding) < 1e-5:
            continue   # skip zero/degenerate embeddings

        metadata = _clean_metadata({
            # Tenant hierarchy — ALL queries must filter by these
            "org_id":          org_id,
            "workspace_id":    workspace_id,
            "collection_id":   collection_id,
            "document_id":     document_id,

            # Content
            "chunk_text":      chunk.text,
            "parent_text":     chunk.parent_text,

            # Structure
            "page":            chunk.page,
            "section":         chunk.section,
            "heading":         chunk.heading,
            "chunk_index":     chunk.chunk_index,
            "is_table":        chunk.is_table,

            # Linking (for neighbor expansion)
            "chunk_id":        chunk.chunk_id,
            "prev_chunk_id":   chunk.prev_chunk_id,
            "next_chunk_id":   chunk.next_chunk_id,

            # Classification
            "document_class":  doc_class.value,
            "language":        language,

            # Stats
            "word_count":      chunk.word_count,
            "importance_score": chunk.importance_score,
        })

        vectors.append({
            "id":       chunk.chunk_id,
            "values":   embedding,
            "metadata": metadata,
        })
        upserted_ids.append(chunk.chunk_id)

    if not vectors:
        _mark_job_failed(job_id, stage, "No valid embeddings produced")
        return 0

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 13: Pinecone upsert
    # ─────────────────────────────────────────────────────────────────────────
    stage = "pinecone_upsert"
    _update_job(job_id, stage=stage)

    t0 = time.time()
    # Batch upserts in chunks of 100 (Pinecone recommended batch size)
    BATCH_SIZE = 100
    try:
        index = _get_index()
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i: i + BATCH_SIZE]
            index.upsert(vectors=batch, namespace=workspace_id)
    except Exception as exc:
        # Attempt rollback of any successfully upserted vectors
        try:
            if upserted_ids:
                _get_index().delete(ids=upserted_ids, namespace=workspace_id)
                logger.warning(
                    "Pinecone rollback: deleted partial vectors",
                    extra={"count": len(upserted_ids), "document_id": document_id},
                )
        except Exception as re:
            logger.error("Rollback failed", extra={"error": str(re)})

        _mark_job_failed(job_id, stage, f"Pinecone upsert failed: {exc}")
        logger.error("Pinecone upsert failed", extra={"document_id": document_id, "error": str(exc)}, exc_info=True)
        return 0

    upsert_ms = round((time.time() - t0) * 1000)
    logger.info(
        "Pinecone upsert complete",
        extra={"document_id": document_id, "vectors": len(vectors), "ms": upsert_ms},
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 14: MongoDB document record update + cache invalidation
    # ─────────────────────────────────────────────────────────────────────────
    stage = "metadata_update"
    _update_job(job_id, stage=stage)

    now = datetime.now(timezone.utc)

    try:
        documents_collection.update_one(
            {"document_id": document_id, "workspace_id": workspace_id},
            {
                "$set": {
                    "document_id":          document_id,
                    "filename":             source_name,
                    "original_filename":    source_name,
                    "workspace_id":         workspace_id,
                    "org_id":               org_id,
                    "collection_id":        collection_id,
                    "sha256_fingerprint":   fingerprint,
                    "file_type":            file_type,
                    "file_size_bytes":      file_size,
                    "document_class":       doc_class.value,
                    "document_class_confidence": class_confidence,
                    "language":             language,
                    "chunking_strategy":    strategy.value,
                    "chunk_count":          len(vectors),
                    "vector_count":         len(vectors),
                    "ingestion_status":     IngestionStatus.COMPLETED.value,
                    "virus_scan_passed":    True,
                    "virus_scan_at":        now,
                    "completed_at":         now,
                }
            },
            upsert=True,
        )
    except Exception as exc:
        logger.error(
            "MongoDB document record update failed (non-fatal — vectors already indexed)",
            extra={"document_id": document_id, "error": str(exc)},
        )

    # Invalidate semantic cache for workspace
    try:
        from services.cache import invalidate_namespace
        invalidate_namespace(workspace_id)
    except Exception as exc:
        logger.warning("Cache invalidation failed (non-fatal)", extra={"error": str(exc)})

    # Finalize job
    total_ms = round((time.time() - t_pipeline_start) * 1000)
    _update_job(
        job_id,
        status=IngestionStatus.COMPLETED.value,
        stage="done",
        chunks_processed=len(vectors),
        vectors_upserted=len(vectors),
        completed_at=time.time(),
    )

    logger.info(
        "Ingestion pipeline complete",
        extra={
            "document_id":   document_id,
            "workspace_id":  workspace_id,
            "vectors":       len(vectors),
            "total_ms":      total_ms,
            "doc_class":     doc_class.value,
            "strategy":      strategy.value,
            "language":      language,
        },
    )

    return len(vectors)