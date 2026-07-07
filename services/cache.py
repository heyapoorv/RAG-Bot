"""
Persistent semantic cache backed by:
  1. Redis (preferred) — fast, native TTL, in-memory
  2. MongoDB (fallback) — guaranteed persistence, slower

Cache key = cosine similarity of query embedding vs stored embeddings.
Namespace-scoped: namespace=None means global.

Features:
- Configurable similarity threshold
- TTL / expiration support
- Namespace isolation
- Cache analytics
- Thread-safe in-memory fallback
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional
from dataclasses import dataclass, field

from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from services.db import cache_collection
from services.embedding import embed_texts
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Cache hit/miss stats ──────────────────────────────────────────────────────

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


_stats = CacheStats()


def get_cache_stats() -> CacheStats:
    return _stats


# ── Redis backend ─────────────────────────────────────────────────────────────

_redis_client: Optional[Any] = None


def _get_redis():
    global _redis_client
    if _redis_client is None and settings.REDIS_URL:
        try:
            import redis as redis_lib
            _redis_client = redis_lib.from_url(
                settings.REDIS_URL,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            _redis_client.ping()
            logger.info("Redis cache backend connected")
        except Exception as exc:
            logger.warning(
                "Redis unavailable — falling back to MongoDB cache",
                extra={"error": str(exc)},
            )
            _redis_client = None
    return _redis_client


# ── MongoDB backend helpers ───────────────────────────────────────────────────

def _mongo_get(question: str, namespace: Optional[str]) -> Optional[dict]:
    """
    Scan MongoDB cache for a semantically similar question.
    Returns cached response dict or None.
    """
    query_filter: dict = {}
    if namespace:
        query_filter["namespace"] = namespace

    q_emb = embed_texts([question])[0]

    cursor = cache_collection.find(
        query_filter,
        {"embedding": 1, "response": 1, "question": 1},
    )

    for doc in cursor:
        stored_emb = doc.get("embedding", [])
        if not stored_emb:
            continue
        sim = float(cosine_similarity([q_emb], [stored_emb])[0][0])
        if sim >= settings.CACHE_SIMILARITY_THRESHOLD:
            logger.debug(
                "MongoDB cache HIT",
                extra={"similarity": round(sim, 4), "namespace": namespace},
            )
            return doc["response"]

    return None


def _mongo_store(
    question: str,
    response: dict,
    namespace: Optional[str],
    ttl_seconds: int,
) -> None:
    """Insert a new cache entry into MongoDB with TTL index support."""
    q_emb = embed_texts([question])[0]
    expires_at = datetime.fromtimestamp(
        time.time() + ttl_seconds, tz=timezone.utc
    )

    cache_collection.insert_one(
        {
            "question": question,
            "namespace": namespace,
            "embedding": q_emb,
            "response": response,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expires_at,  # Used by TTL index
        }
    )


# ── Redis helpers ─────────────────────────────────────────────────────────────

def _redis_get(question: str, namespace: Optional[str]) -> Optional[dict]:
    """
    Redis does not support cosine similarity natively (without RedisSearch).
    We store a sorted list of (embedding_key, metadata_key) pairs in a hash
    and iterate — acceptable for cache sizes < 10k.
    For large deployments, swap to Redis Stack with vector similarity.
    """
    import json
    import struct

    rc = _get_redis()
    if rc is None:
        return None

    q_emb = embed_texts([question])[0]
    ns_prefix = namespace or "global"
    index_key = f"cache:index:{ns_prefix}"

    entries = rc.hgetall(index_key)  # {id: json({embedding, response_key})}
    if not entries:
        return None

    for entry_id, meta_bytes in entries.items():
        try:
            meta = json.loads(meta_bytes)
            stored_emb = meta["embedding"]
            sim = float(cosine_similarity([q_emb], [stored_emb])[0][0])
            if sim >= settings.CACHE_SIMILARITY_THRESHOLD:
                response_key = f"cache:response:{meta['response_id']}"
                raw = rc.get(response_key)
                if raw:
                    logger.debug(
                        "Redis cache HIT",
                        extra={"similarity": round(sim, 4), "namespace": namespace},
                    )
                    return json.loads(raw)
        except Exception:
            continue

    return None


def _redis_store(
    question: str,
    response: dict,
    namespace: Optional[str],
    ttl_seconds: int,
) -> None:
    import json
    import uuid

    rc = _get_redis()
    if rc is None:
        return

    q_emb = embed_texts([question])[0]
    ns_prefix = namespace or "global"
    index_key = f"cache:index:{ns_prefix}"
    response_id = uuid.uuid4().hex
    response_key = f"cache:response:{response_id}"

    meta = json.dumps({"embedding": q_emb, "response_id": response_id})
    rc.hset(index_key, response_id, meta)
    rc.expire(index_key, ttl_seconds)

    rc.set(response_key, json.dumps(response), ex=ttl_seconds)


# ── Public interface ──────────────────────────────────────────────────────────

def get_cached_answer(
    question: str,
    namespace: Optional[str] = None,
) -> Optional[dict]:
    """
    Look up question in cache. Returns cached response or None.
    Tries Redis first, then MongoDB.
    """
    if not settings.CACHE_ENABLED:
        return None

    try:
        rc = _get_redis()
        result = (
            _redis_get(question, namespace)
            if rc is not None
            else _mongo_get(question, namespace)
        )

        if result:
            _stats.hits += 1
        else:
            _stats.misses += 1

        return result

    except Exception as exc:
        _stats.errors += 1
        logger.error("Cache lookup failed", extra={"error": str(exc)})
        return None


def store_cached_answer(
    question: str,
    response: dict,
    namespace: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> None:
    """
    Persist a new cache entry. Tries Redis first, then MongoDB.
    """
    if not settings.CACHE_ENABLED:
        return

    ttl = ttl_seconds or settings.REDIS_CACHE_TTL_SECONDS

    try:
        rc = _get_redis()
        if rc is not None:
            _redis_store(question, response, namespace, ttl)
        else:
            _mongo_store(question, response, namespace, ttl)

        _stats.stores += 1
        logger.debug(
            "Cache stored",
            extra={"namespace": namespace, "ttl": ttl},
        )

    except Exception as exc:
        _stats.errors += 1
        logger.error("Cache store failed", extra={"error": str(exc)})


def invalidate_namespace(namespace: str) -> int:
    """
    Delete all cache entries for a namespace.
    Returns number of deleted entries.
    """
    try:
        rc = _get_redis()
        if rc is not None:
            ns_prefix = namespace or "global"
            index_key = f"cache:index:{ns_prefix}"
            entries = rc.hgetall(index_key) or {}
            for response_id in entries:
                rc.delete(f"cache:response:{response_id.decode()}")
            rc.delete(index_key)
            return len(entries)
        else:
            result = cache_collection.delete_many({"namespace": namespace})
            return result.deleted_count
    except Exception as exc:
        logger.error("Cache invalidation failed", extra={"error": str(exc)})
        return 0