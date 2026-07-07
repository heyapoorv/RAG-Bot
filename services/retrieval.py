import logging
from typing import Union, List, Dict, Optional, Set

from services.embedding import embed_texts
from services.vectordb import index
from services.rrf import rrf_fusion
from services.bm25 import rank_bm25


# ── Sentinel values used during ingestion ────────────────────────────────────
_SENTINELS: Set[str] = {"START", "END"}


# ---------------------------------------------------
# Neighbor Expansion
# ---------------------------------------------------
def expand_neighbor_chunks(
    chunks: List[Dict],
    namespace: Optional[str] = None,
    window: int = 1,
) -> List[Dict]:
    """
    Expand each retrieved chunk by fetching its immediately adjacent chunks
    (prev / next) from Pinecone using the stored chunk linkage metadata.

    This fills context gaps when relevant information straddles chunk boundaries.

    Args:
        chunks:    Retrieved chunk dicts (must have prev_chunk_id / next_chunk_id in metadata).
        namespace: Pinecone namespace to fetch from.
        window:    Number of adjacent chunks to fetch on each side (default 1).

    Returns:
        List of chunks with neighbor text merged into chunk_text.
        Order and deduplication are preserved.
    """
    if not chunks or window < 1:
        return chunks

    # ── Collect neighbor IDs to fetch ─────────────────────────────────────────
    # Already-known IDs — skip fetching these to avoid duplication
    known_ids: Set[str] = {
        c.get("chunk_id", "")
        for c in chunks
        if c.get("chunk_id")
    }

    neighbor_ids: List[str] = []
    for chunk in chunks:
        for side in ("prev_chunk_id", "next_chunk_id"):
            cid = chunk.get(side, "")
            if cid and cid not in _SENTINELS and cid not in known_ids:
                neighbor_ids.append(cid)
                known_ids.add(cid)  # prevent duplicates across chunks

    if not neighbor_ids:
        return chunks

    # ── Batch fetch from Pinecone ─────────────────────────────────────────────
    neighbor_texts: Dict[str, str] = {}
    try:
        fetch_result = index.fetch(ids=neighbor_ids, namespace=namespace or "")
        fetched_vectors = fetch_result.get("vectors") or {}
        for cid, vec_data in fetched_vectors.items():
            meta = (vec_data.get("metadata") or {})
            text = meta.get("chunk_text") or meta.get("parent_text") or ""
            if text:
                neighbor_texts[cid] = text
    except Exception as exc:
        logging.warning(
            f"Neighbor expansion fetch failed — returning original chunks: {exc}"
        )
        return chunks

    # ── Merge neighbor text into each chunk ───────────────────────────────────
    expanded: List[Dict] = []
    dedup_keys: Set[str] = set()

    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        if cid in dedup_keys:
            continue
        dedup_keys.add(cid)

        prev_id = chunk.get("prev_chunk_id", "")
        next_id = chunk.get("next_chunk_id", "")
        original_text = chunk.get("chunk_text", "")

        prev_text = (
            neighbor_texts.get(prev_id, "")
            if prev_id and prev_id not in _SENTINELS
            else ""
        )
        next_text = (
            neighbor_texts.get(next_id, "")
            if next_id and next_id not in _SENTINELS
            else ""
        )

        if prev_text or next_text:
            merged_parts = [p for p in [prev_text, original_text, next_text] if p]
            expanded_text = " ".join(merged_parts)
            expanded.append({
                **chunk,
                "chunk_text": expanded_text,
                "expanded": True,
            })
        else:
            expanded.append(chunk)

    return expanded


# ---------------------------------------------------
# Confidence Filter
# ---------------------------------------------------
def filter_low_confidence(
    chunks: List[Dict],
    threshold: float = 0.20,
) -> List[Dict]:
    """
    Filter out chunks that have a very low dense retrieval score.
    Uses dense_score (cosine similarity, range 0-1) rather than rrf_score
    (which is a small fraction ~ 1/(k+rank) and always below 0.20).
    """
    return [
        c for c in chunks
        if c.get("dense_score", c.get("score", 1.0)) >= threshold
    ]


# ---------------------------------------------------
# Main Retrieval
# ---------------------------------------------------
def retrieve_contexts(
    query: Union[str, List[str]],
    top_k: int = 5,
    namespace: str = None,
    collection_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
):

    try:
        queries = query if isinstance(query, list) else [query]
        queries = [q for q in queries if q]
        
        if not queries:
            return []

        # -----------------------------
        # Metadata Filters
        # -----------------------------
        # Pinecone filters must be structured carefully
        filter_dict = {}
        if collection_ids:
            filter_dict["collection_id"] = {"$in": collection_ids}
        if document_ids:
            filter_dict["document_id"] = {"$in": document_ids}
        
        # If no filters, set filter_dict to None so we don't pass an empty dict (Pinecone might reject it)
        pinecone_filter = filter_dict if filter_dict else None

        # -----------------------------
        # Batch Embedding API Call
        # -----------------------------
        # Embed all queries in a single API call for async batching!
        all_embs = embed_texts(queries)

        vector_rankings = []
        bm25_rankings = []

        for idx, q in enumerate(queries):
            q_emb = all_embs[idx]

            res = index.query(
                vector=q_emb,
                top_k=top_k * 4,
                include_metadata=True,
                namespace=namespace,
                filter=pinecone_filter
            )
            
            matches = res.get("matches", []) or []
            v_results = []
            
            for m in matches:
                meta = m.get("metadata", {}) or {}
                score = m.get("score", 0)
                v_results.append({
                    "source": meta.get("source", "unknown"),
                    "document_id": meta.get("document_id"),
                    "chunk_id": meta.get("chunk_id"),
                    "parent_id": meta.get("parent_id"),
                    "prev_chunk_id": meta.get("prev_chunk_id"),
                    "next_chunk_id": meta.get("next_chunk_id"),
                    "chunk_text": meta.get("parent_text", meta.get("chunk_text", "")),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "score": score,
                    "dense_score": score,  # Save explicitly for Trace Viewer
                })

            vector_rankings.append(v_results)

            # -----------------------------
            # Sparse BM25 Re-Ranking
            # -----------------------------
            bm_results = rank_bm25(q, v_results, namespace=namespace)
            bm25_rankings.append(bm_results)

        # -----------------------------
        # RRF Fusion
        # -----------------------------
        fused = rrf_fusion(
            vector_rankings + bm25_rankings,
            k=60
        )

        # -----------------------------
        # Confidence Filtering
        # -----------------------------
        fused = filter_low_confidence(
            fused,
            threshold=0.20
        )

        # -----------------------------
        # Neighbor Expansion (real fetch)
        # -----------------------------
        fused = expand_neighbor_chunks(
            fused,
            namespace=namespace,
            window=1,
        )

        # -----------------------------
        # Final Dedup
        # -----------------------------
        seen_texts = set()
        deduped = []

        for chunk in fused:
            text = chunk.get("chunk_text", "")

            if text in seen_texts:
                continue

            seen_texts.add(text)
            deduped.append(chunk)

        return deduped[:top_k]

    except Exception as e:
        logging.error(
            f"Hybrid retrieval failed: {e}",
            exc_info=True
        )
        return []