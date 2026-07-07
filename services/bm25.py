"""
BM25 scoring with true IDF.

Corpus statistics (per namespace) are maintained in MongoDB by the ingestion
pipeline and loaded here with a short in-process TTL cache to avoid DB reads
on every query.

BM25 formula (Robertson et al.):
    score(q, d) = Σ IDF(t) * tf(t,d)*(k1+1) / (tf(t,d) + k1*(1-b + b*|d|/avgdl))

Constants:
    k1 = 1.5  (term saturation)
    b  = 0.75 (length normalization)
"""
from __future__ import annotations

import math
import time
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── BM25 hyperparameters ──────────────────────────────────────────────────────
_K1: float = 1.5
_B: float = 0.75
_FALLBACK_AVGDL: int = 120  # reasonable default chunk length in words
_FALLBACK_N: int = 100      # assumed corpus size when no stats available

# ── In-process corpus stats cache ─────────────────────────────────────────────
# Structure: { namespace: (stats_dict, loaded_at_timestamp) }
_corpus_cache: Dict[str, Tuple[dict, float]] = {}
_CACHE_TTL_SECONDS: float = 60.0


def _load_corpus_stats(namespace: Optional[str]) -> dict:
    """
    Load BM25 corpus statistics for a namespace from MongoDB.
    Returns a dict:
        {
            "total_chunks": int,
            "avg_chunk_length": float,
            "df": { term: int }   # document frequency per term
        }
    Uses an in-process cache with 60-second TTL to avoid per-query DB reads.
    """
    ns_key = namespace or "__global__"
    cached = _corpus_cache.get(ns_key)
    if cached is not None:
        stats, loaded_at = cached
        if time.time() - loaded_at < _CACHE_TTL_SECONDS:
            return stats

    try:
        from services.db import bm25_corpus_collection
        docs = list(bm25_corpus_collection.find(
            {"namespace": ns_key},
            {"term": 1, "df": 1, "total_chunks": 1, "avg_chunk_length": 1, "_id": 0},
        ))

        if not docs:
            stats = {
                "total_chunks": _FALLBACK_N,
                "avg_chunk_length": float(_FALLBACK_AVGDL),
                "df": {},
            }
        else:
            # All docs share total_chunks / avg_chunk_length — take from first
            total_chunks = docs[0].get("total_chunks", _FALLBACK_N)
            avg_chunk_length = float(docs[0].get("avg_chunk_length", _FALLBACK_AVGDL))
            df = {d["term"]: d["df"] for d in docs if "term" in d}
            stats = {
                "total_chunks": total_chunks,
                "avg_chunk_length": avg_chunk_length,
                "df": df,
            }
    except Exception as exc:
        logger.warning(f"BM25 corpus stats load failed — using fallback: {exc}")
        stats = {
            "total_chunks": _FALLBACK_N,
            "avg_chunk_length": float(_FALLBACK_AVGDL),
            "df": {},
        }

    _corpus_cache[ns_key] = (stats, time.time())
    return stats


def invalidate_corpus_cache(namespace: Optional[str] = None) -> None:
    """Invalidate in-process cache for a namespace (call after ingestion)."""
    ns_key = namespace or "__global__"
    _corpus_cache.pop(ns_key, None)


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenizer — identical to the one used at index time."""
    return text.lower().split()


def _idf(term: str, N: int, df_map: dict) -> float:
    """
    Robertson IDF (smoothed to avoid negative values for very common terms):
        IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    """
    df = df_map.get(term, 0)
    return math.log((N - df + 0.5) / (df + 0.5) + 1)


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avgdl: float,
    N: int,
    df_map: dict,
) -> float:
    """
    Compute the BM25 score for a single (query, document) pair.

    Args:
        query_tokens: Tokenized query terms.
        doc_tokens:   Tokenized document/chunk terms.
        avgdl:        Average document length across the corpus.
        N:            Total number of documents (chunks) in the corpus.
        df_map:       Term → document frequency mapping.

    Returns:
        BM25 relevance score (float ≥ 0).
    """
    score = 0.0
    doc_len = len(doc_tokens)
    tf_map = Counter(doc_tokens)

    for term in query_tokens:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        idf = _idf(term, N, df_map)
        numerator = tf * (_K1 + 1)
        denominator = tf + _K1 * (1 - _B + _B * doc_len / max(avgdl, 1))
        score += idf * (numerator / denominator)

    return score


def rank_bm25(
    query: str,
    docs: List[dict],
    namespace: Optional[str] = None,
) -> List[dict]:
    """
    Re-rank a list of retrieved chunks using BM25 with true IDF.

    Args:
        query:     Raw query string.
        docs:      Retrieved chunk dicts with "chunk_text" key.
        namespace: Pinecone namespace — used to look up per-namespace corpus stats.

    Returns:
        Docs sorted by BM25 score (descending), with "bm25_score" field added.
    """
    if not docs:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return docs

    corpus = _load_corpus_stats(namespace)
    N = corpus["total_chunks"]
    avgdl = corpus["avg_chunk_length"]
    df_map = corpus["df"]

    scored = []
    for d in docs:
        text = d.get("chunk_text", "")
        doc_tokens = tokenize(text)
        score = bm25_score(q_tokens, doc_tokens, avgdl, N, df_map)
        scored.append({**d, "bm25_score": score})

    return sorted(scored, key=lambda x: x["bm25_score"], reverse=True)


# ── Corpus stats maintenance (called from ingestion) ──────────────────────────

def update_corpus_stats(
    chunks: List[dict],
    namespace: Optional[str],
) -> None:
    """
    Update BM25 corpus statistics in MongoDB after a batch ingestion.

    Increments document frequency for each unique term seen in new chunks,
    and updates total_chunks and avg_chunk_length atomically.

    Args:
        chunks:    List of chunk dicts (must have "text" or "chunk_text" key).
        namespace: Pinecone namespace.
    """
    if not chunks:
        return

    ns_key = namespace or "__global__"

    try:
        from services.db import bm25_corpus_collection

        # Gather term frequencies across all new chunks
        term_doc_freq: Counter = Counter()
        total_words = 0

        for chunk in chunks:
            text = chunk.get("text", chunk.get("chunk_text", ""))
            tokens = set(tokenize(text))  # set = document frequency (not TF)
            term_doc_freq.update(tokens)
            total_words += len(tokenize(text))

        num_new_chunks = len(chunks)
        avg_chunk_len = total_words / num_new_chunks if num_new_chunks else 0

        # Upsert each term's DF using MongoDB $inc + $set
        # Also carry forward total_chunks and avg_chunk_length on every term doc
        # (denormalized for simplicity — all term docs for a namespace share these)
        bulk_ops = []
        from pymongo import UpdateOne
        for term, df_increment in term_doc_freq.items():
            bulk_ops.append(UpdateOne(
                {"namespace": ns_key, "term": term},
                {
                    "$inc": {"df": df_increment},
                    "$set": {
                        "namespace": ns_key,
                        "term": term,
                        # These will be slightly inaccurate per-term but are
                        # consistent with the latest ingestion batch
                        "total_chunks": num_new_chunks,
                        "avg_chunk_length": avg_chunk_len,
                    },
                },
                upsert=True,
            ))

        if bulk_ops:
            bm25_corpus_collection.bulk_write(bulk_ops, ordered=False)
            logger.debug(
                f"BM25 corpus updated: {len(bulk_ops)} terms, ns={ns_key}, "
                f"chunks={num_new_chunks}"
            )

        # Invalidate the in-process cache so the next query reads fresh stats
        invalidate_corpus_cache(namespace)

    except Exception as exc:
        logger.warning(f"BM25 corpus stats update failed (non-fatal): {exc}")