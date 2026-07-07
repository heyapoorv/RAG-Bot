"""
Analytics Routes.
Requires analyst role or above for global analytics.
Standard users can only view their own namespace analytics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from services.auth_service import get_current_user, require_role, role_level
from services.db import analytics_collection
from models.schemas import AnalyticsSummary, RecentQueryRecord, TopQuestion, FailureRecord
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _enforce_namespace_scoping(user: dict, requested_namespace: Optional[str]) -> str:
    """
    Enforce tenant isolation.
    Analysts or above can query any namespace.
    Regular users are forced to query their own username or namespace.
    """
    u_role = user.get("role", "user")
    u_name = user.get("username")

    if role_level(u_role) >= role_level("analyst"):
        # Analysts/admins can query any namespace or global (empty string)
        return requested_namespace or ""

    # Regular users can only query their own namespace (tied to username)
    if requested_namespace and requested_namespace != u_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You can only access analytics for your own namespace.",
        )
    return u_name


# ── GET Summary Analytics ─────────────────────────────────────────────────────

@router.get("/summary", response_model=AnalyticsSummary)
def get_summary(
    days: int = Query(7, ge=1, le=365),
    namespace: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Aggregate KPI stats for a given namespace and timeframe."""
    target_ns = _enforce_namespace_scoping(user, namespace)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    match_filter: dict = {"timestamp": {"$gte": cutoff.timestamp()}}
    if target_ns:
        match_filter["namespace"] = target_ns

    pipeline = [
        {"$match": match_filter},
        {
            "$group": {
                "_id": None,
                "total_queries": {"$sum": 1},
                "avg_latency": {"$avg": "$latency_ms"},
                "cache_hits": {"$sum": {"$cond": ["$cache_hit", 1, 0]}},
                "verified_answers": {"$sum": {"$cond": ["$verified", 1, 0]}},
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
        return AnalyticsSummary(
            total_queries=0,
            avg_latency_ms=0.0,
            cache_hit_rate=0.0,
            verification_rate=0.0,
            failure_rate=0.0,
        )

    data = result[0]
    total = data["total_queries"] or 1

    return AnalyticsSummary(
        total_queries=total,
        avg_latency_ms=round(data["avg_latency"] or 0.0, 2),
        cache_hit_rate=round(data["cache_hits"] / total, 3),
        verification_rate=round(data["verified_answers"] / total, 3),
        failure_rate=round(data["failures"] / total, 3),
    )


# ── GET Recent Queries ────────────────────────────────────────────────────────

@router.get("/recent", response_model=List[RecentQueryRecord])
def get_recent(
    limit: int = Query(20, ge=1, le=100),
    days: int = Query(7, ge=1, le=365),
    namespace: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Retrieve recent queries with latency and verification data."""
    target_ns = _enforce_namespace_scoping(user, namespace)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query_filter: dict = {"timestamp": {"$gte": cutoff.timestamp()}}

    if target_ns:
        query_filter["namespace"] = target_ns

    cursor = analytics_collection.find(
        query_filter,
        {
            "_id": 0,
            "question": 1,
            "answer": 1,
            "latency_ms": 1,
            "verified": 1,
            "cache_hit": 1,
            "namespace": 1,
            "timestamp": 1,
        },
    ).sort("timestamp", -1).limit(limit)

    return [RecentQueryRecord(**doc) for doc in cursor]


# ── GET Top Questions ─────────────────────────────────────────────────────────

@router.get("/top-questions", response_model=List[TopQuestion])
def get_top_questions(
    limit: int = Query(10, ge=1, le=50),
    namespace: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Retrieve top repeating questions for key insights."""
    target_ns = _enforce_namespace_scoping(user, namespace)

    match_stage = {}
    if target_ns:
        match_stage["namespace"] = target_ns

    pipeline = []
    if match_stage:
        pipeline.append({"$match": match_stage})

    pipeline += [
        {"$group": {"_id": "$question", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit},
    ]

    result = list(analytics_collection.aggregate(pipeline))

    return [
        TopQuestion(question=r["_id"], count=r["count"])
        for r in result
    ]


# ── GET Failures ──────────────────────────────────────────────────────────────

@router.get("/failures", response_model=List[FailureRecord])
def get_failures(
    limit: int = Query(20, ge=1, le=100),
    namespace: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
):
    """Retrieve failed RAG runs (unverified or system fallback responses)."""
    target_ns = _enforce_namespace_scoping(user, namespace)

    query_filter = {
        "$or": [
            {"verified": False},
            {"answer": "NOT_FOUND_IN_DOCS"},
            {"answer": "GENERATION_FAILED"},
            {"answer": "RATE_LIMITED"},
        ]
    }

    if target_ns:
        query_filter["namespace"] = target_ns

    cursor = analytics_collection.find(
        query_filter,
        {
            "_id": 0,
            "question": 1,
            "answer": 1,
            "latency_ms": 1,
            "namespace": 1,
            "timestamp": 1,
        },
    ).sort("timestamp", -1).limit(limit)

    return [FailureRecord(**doc) for doc in cursor]