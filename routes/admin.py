"""
Admin routes: dashboard summary, query explorer, failure debugger, RAG trace viewer,
audit logs, cache management, system health.
All protected by 'analyst' or 'admin' role minimum.
"""
from __future__ import annotations

import time
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query

from services.auth_service import require_role, log_audit_event
from services.db import analytics_collection, audit_collection, users_collection
from services.cache import get_cache_stats, invalidate_namespace
from services.metrics import metrics
from models.schemas import TraceResponse, TraceChunk
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _format_doc(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# ── Summary KPIs ──────────────────────────────────────────────────────────────

@router.get("/summary")
def summary(
    days: int = Query(7, ge=1, le=365),
    analyst=Depends(require_role("analyst")),
):
    """Aggregate KPI summary for the admin dashboard."""
    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).timestamp()

    pipeline = [
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {
            "$group": {
                "_id": None,
                "total_queries": {"$sum": 1},
                "avg_latency": {"$avg": "$latency_ms"},
                "cache_hits": {"$sum": {"$cond": ["$cache_hit", 1, 0]}},
                "verified": {"$sum": {"$cond": ["$verified", 1, 0]}},
                "failures": {
                    "$sum": {
                        "$cond": [
                            {
                                "$in": [
                                    "$answer",
                                    ["NOT_FOUND_IN_DOCS", "GENERATION_FAILED", "RATE_LIMITED"],
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
            }
        },
    ]

    result = list(analytics_collection.aggregate(pipeline))
    if not result:
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "cache_hit_rate": 0,
            "verification_rate": 0,
            "failure_rate": 0,
        }

    res = result[0]
    total = res["total_queries"] or 1
    return {
        "total_queries": total,
        "avg_latency_ms": round(res.get("avg_latency") or 0, 2),
        "cache_hit_rate": round(res["cache_hits"] / total, 3),
        "verification_rate": round(res["verified"] / total, 3),
        "failure_rate": round(res["failures"] / total, 3),
    }


# ── Query Explorer ────────────────────────────────────────────────────────────

@router.get("/queries")
def queries(
    limit: int = Query(50, ge=1, le=200),
    namespace: str = Query(None),
    analyst=Depends(require_role("analyst")),
):
    """Return recent query logs (excludes raw chunk data for size)."""
    query_filter = {}
    if namespace:
        query_filter["namespace"] = namespace

    cursor = analytics_collection.find(
        query_filter,
        {"retrieved_chunks": 0, "reranked_chunks": 0},
    ).sort("timestamp", -1).limit(limit)

    return [_format_doc(doc) for doc in cursor]


# ── Failure Debugger ──────────────────────────────────────────────────────────

@router.get("/failures")
def failures(
    limit: int = Query(50, ge=1, le=200),
    analyst=Depends(require_role("analyst")),
):
    """Return queries that failed verification or generation."""
    cursor = analytics_collection.find(
        {
            "$or": [
                {"answer": "NOT_FOUND_IN_DOCS"},
                {"answer": "GENERATION_FAILED"},
                {"answer": "RATE_LIMITED"},
                {"verified": False},
            ]
        }
    ).sort("timestamp", -1).limit(limit)

    return [_format_doc(doc) for doc in cursor]


# ── RAG Trace Viewer ──────────────────────────────────────────────────────────

@router.get("/trace/{query_id}", response_model=TraceResponse)
def trace(
    query_id: str,
    analyst=Depends(require_role("analyst")),
):
    """Deep-dive trace for a specific query — includes all chunk details."""
    try:
        obj_id = ObjectId(query_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid query ID format.")

    doc = analytics_collection.find_one({"_id": obj_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Trace not found.")

    def _to_chunk(c: dict) -> TraceChunk:
        return TraceChunk(
            source=c.get("source", "unknown"),
            chunk_text=c.get("chunk_text", ""),
            score=c.get("score"),
            rerank_score=c.get("rerank_score"),
            page=c.get("page"),
            section=c.get("section"),
        )

    return TraceResponse(
        id=str(doc["_id"]),
        question=doc.get("question", ""),
        retrieved_chunks=[_to_chunk(c) for c in doc.get("retrieved_chunks", [])],
        reranked_chunks=[_to_chunk(c) for c in doc.get("reranked_chunks", [])],
        final_context=doc.get("final_context", ""),
        answer=doc.get("answer", ""),
        latency_ms=doc.get("latency_ms", 0),
        timestamp=doc.get("timestamp"),
    )


# ── Cache Management ──────────────────────────────────────────────────────────

@router.get("/cache/stats")
def cache_stats(admin=Depends(require_role("admin"))):
    """Return semantic cache statistics."""
    stats = get_cache_stats()
    return {
        "hits": stats.hits,
        "misses": stats.misses,
        "stores": stats.stores,
        "errors": stats.errors,
        "hit_rate": round(stats.hit_rate, 3),
    }


@router.delete("/cache/{namespace}")
def invalidate_cache(
    namespace: str,
    admin=Depends(require_role("admin")),
):
    """Invalidate all cache entries for a namespace."""
    deleted = invalidate_namespace(namespace)
    log_audit_event("cache_invalidate", admin["username"], target=namespace)
    return {"deleted": deleted, "namespace": namespace}


# ── Audit Logs ────────────────────────────────────────────────────────────────

@router.get("/audit")
def audit_logs(
    limit: int = Query(50, ge=1, le=500),
    admin=Depends(require_role("admin")),
):
    """Return admin audit logs."""
    cursor = audit_collection.find(
        {},
        {"_id": 0},
    ).sort("timestamp", -1).limit(limit)
    return list(cursor)


# ── System Health ─────────────────────────────────────────────────────────────

@router.get("/health")
def admin_health(admin=Depends(require_role("analyst"))):
    """Detailed system health for admin panel."""
    health: dict = {"services": {}}

    # MongoDB
    t0 = time.time()
    try:
        from services.db import _client
        _client.admin.command("ping")
        health["services"]["mongodb"] = {
            "status": "ok",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as exc:
        health["services"]["mongodb"] = {"status": "down", "detail": str(exc)}

    # Pinecone
    t0 = time.time()
    try:
        from services.vectordb import index
        index.describe_index_stats()
        health["services"]["pinecone"] = {
            "status": "ok",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as exc:
        health["services"]["pinecone"] = {"status": "down", "detail": str(exc)}

    # Redis
    try:
        from services.cache import _get_redis
        rc = _get_redis()
        if rc:
            t0 = time.time()
            rc.ping()
            health["services"]["redis"] = {
                "status": "ok",
                "latency_ms": round((time.time() - t0) * 1000, 2),
            }
        else:
            health["services"]["redis"] = {"status": "not_configured"}
    except Exception as exc:
        health["services"]["redis"] = {"status": "down", "detail": str(exc)}

    # Cache stats
    stats = get_cache_stats()
    health["cache_stats"] = {
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate": round(stats.hit_rate, 3),
    }

    # Overall status
    statuses = [s.get("status", "ok") for s in health["services"].values()]
    health["overall"] = "down" if "down" in statuses else "ok"

    return health