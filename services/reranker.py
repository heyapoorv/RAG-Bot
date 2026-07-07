"""
Production-grade configurable reranker.

Modes:
    local           — fast cosine similarity using Gemini embeddings
    cross_encoder   — transformer CrossEncoder (sentence-transformers)
    hybrid          — local pre-filter → cross_encoder re-score top candidates

Models (cross_encoder mode):
    Fast:     cross-encoder/ms-marco-MiniLM-L-6-v2  (~80MB)
    Balanced: BAAI/bge-reranker-base                 (~450MB)
    Accurate: BAAI/bge-reranker-large                (~1.3GB)

GPU support: auto-detected if RERANKER_USE_GPU=true.
Lazy-loaded on first call to avoid startup delays.
"""
from __future__ import annotations

import time
from typing import List, Dict, Optional, Literal

from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from services.embedding import embed_texts
from utils.logger import get_logger

logger = get_logger(__name__)


# ── CrossEncoder lazy loader ──────────────────────────────────────────────────

_cross_encoder_model: Optional[object] = None
_cross_encoder_model_name: str = ""


def _load_cross_encoder(model_name: str) -> object:
    """
    Load (or re-use) the CrossEncoder model. Thread-safe for read; first-call
    initialisation is idempotent so double-init is harmless.
    """
    global _cross_encoder_model, _cross_encoder_model_name

    if _cross_encoder_model is not None and _cross_encoder_model_name == model_name:
        return _cross_encoder_model

    try:
        from sentence_transformers import CrossEncoder

        device = "cuda" if settings.RERANKER_USE_GPU else "cpu"
        logger.info(
            "Loading CrossEncoder model",
            extra={"model": model_name, "device": device},
        )
        t0 = time.time()
        _cross_encoder_model = CrossEncoder(model_name, device=device)
        _cross_encoder_model_name = model_name
        logger.info(
            "CrossEncoder loaded",
            extra={"model": model_name, "elapsed_ms": round((time.time() - t0) * 1000)},
        )
        return _cross_encoder_model
    except Exception as exc:
        logger.error(
            "CrossEncoder load failed — falling back to local reranker",
            extra={"error": str(exc)},
        )
        return None


# ── Individual rankers ────────────────────────────────────────────────────────

def _local_rerank(
    query: str,
    chunks: List[Dict],
    top_n: int,
) -> List[Dict]:
    """
    Rerank using cosine similarity between query and chunk embeddings.
    Fast, no extra model required.
    """
    if not chunks:
        return []

    query_emb = embed_texts([query])[0]
    chunk_texts = [c.get("chunk_text", "") for c in chunks]
    chunk_embs = embed_texts(chunk_texts)

    for chunk, emb in zip(chunks, chunk_embs):
        sim = float(cosine_similarity([query_emb], [emb])[0][0])
        chunk["rerank_score"] = sim

    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_n]


def _cross_encoder_rerank(
    query: str,
    chunks: List[Dict],
    top_n: int,
    model_name: str,
) -> List[Dict]:
    """
    Rerank using a CrossEncoder model. Significantly improves relevance
    over bi-encoder cosine similarity.
    """
    if not chunks:
        return []

    model = _load_cross_encoder(model_name)
    if model is None:
        logger.warning("CrossEncoder unavailable — using local reranker as fallback")
        return _local_rerank(query, chunks, top_n)

    pairs = [(query, chunk.get("chunk_text", "")) for chunk in chunks]

    try:
        scores = model.predict(pairs)
    except Exception as exc:
        logger.error("CrossEncoder prediction failed", extra={"error": str(exc)})
        return _local_rerank(query, chunks, top_n)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    return sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_n]


def _hybrid_rerank(
    query: str,
    chunks: List[Dict],
    top_n: int,
    model_name: str,
    pre_filter_n: int = 20,
) -> List[Dict]:
    """
    Hybrid mode:
      1. Local cosine similarity to narrow from N → pre_filter_n candidates.
      2. CrossEncoder precision pass on those candidates.
    Balances speed and accuracy.
    """
    pre_filtered = _local_rerank(query, chunks, top_n=pre_filter_n)
    return _cross_encoder_rerank(query, pre_filtered, top_n=top_n, model_name=model_name)


# ── Public interface ──────────────────────────────────────────────────────────

def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_n: Optional[int] = None,
    mode: Optional[Literal["local", "cross_encoder", "hybrid"]] = None,
    model_name: Optional[str] = None,
) -> List[Dict]:
    """
    Rerank retrieved chunks by relevance to query.

    Args:
        query:      The user question.
        chunks:     Retrieved chunk dicts (must have "chunk_text" key).
        top_n:      Number of top chunks to return (defaults to settings.DEFAULT_TOP_K).
        mode:       Override for reranker mode.
        model_name: Override for model name.

    Returns:
        Sorted list of chunk dicts with added "rerank_score" key.
    """
    if not chunks:
        return []

    top_n = top_n or settings.DEFAULT_TOP_K
    mode = mode or settings.RERANKER_MODE
    model_name = model_name or settings.RERANKER_MODEL

    t0 = time.time()
    result: List[Dict] = []

    try:
        if mode == "local":
            result = _local_rerank(query, chunks, top_n)

        elif mode == "cross_encoder":
            result = _cross_encoder_rerank(query, chunks, top_n, model_name)

        elif mode == "hybrid":
            result = _hybrid_rerank(query, chunks, top_n, model_name)

        else:
            logger.warning(f"Unknown reranker mode '{mode}' — using local")
            result = _local_rerank(query, chunks, top_n)

    except Exception as exc:
        logger.error(
            "Reranker failed — returning unranked chunks",
            extra={"error": str(exc), "mode": mode},
        )
        result = chunks[:top_n]

    elapsed_ms = round((time.time() - t0) * 1000, 2)
    logger.debug(
        "Reranking complete",
        extra={
            "mode": mode,
            "input_chunks": len(chunks),
            "output_chunks": len(result),
            "elapsed_ms": elapsed_ms,
        },
    )
    return result


def preload_reranker() -> None:
    """
    Called at application startup to eagerly load the CrossEncoder model
    instead of paying the load penalty on the first request.
    """
    if settings.RERANKER_MODE in ("cross_encoder", "hybrid"):
        _load_cross_encoder(settings.RERANKER_MODEL)