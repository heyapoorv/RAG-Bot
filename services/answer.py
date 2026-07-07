"""
services/answer.py — Phase 3

Main RAG pipeline. Updated to:
  - Use services.synthesizer for answer generation (multi-doc / single-doc)
  - Use enhanced services.memory (document-scoped sessions, follow-up resolution)
  - Propagate synthesis metadata (evidence_by_document, conflicts, synthesis_mode)
  - Enrich citations with chunk_id, collection_id, and claim_text
  - Inline generation quality scoring (faithfulness, completeness, cross-doc consistency)
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from google import genai
from dotenv import load_dotenv

from config import settings
from services.retrieval import retrieve_contexts
from services.query_guard import is_valid_rag_query
from services.cache import get_cached_answer, store_cached_answer
from services.analytics import log_query_event
from services.chunk_compressor import compress_chunk_async as compress_chunk
from services.memory import (
    get_history,
    append_message,
    trim_history,
    get_active_document_ids,
    get_active_collection_id,
    resolve_followup,
    is_followup_question,
)
from services.reranker import rerank_chunks
from services.dynamic_topk import compute_dynamic_topk
from services.answer_verifier import verify_answer
from services.synthesizer import synthesize_answer
from services.prompts import SINGLE_DOC_QA_PROMPT as QA_PROMPT_TEMPLATE
from services.generation_scorer import score_generation
from services.metrics import metrics, track_latency
from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

client = genai.Client(api_key=settings.GOOGLE_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_history(history: List[Dict]) -> str:
    return "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in history
    )


def _build_citations(
    chunks: List[Dict],
    claim_index_map: Optional[Dict] = None,
) -> List[Dict]:
    """
    Build enriched citation objects from final reranked chunks.
    Populates claim_text from the synthesizer's claim_index_map when available.
    """
    claim_index_map = claim_index_map or {}
    return [
        {
            "source": c.get("source", "unknown"),
            "document_id": c.get("document_id"),
            "page": c.get("page"),
            "section": c.get("section"),
            "highlight": c.get("chunk_text", "")[:500],
            "score": c.get("rerank_score") or c.get("score"),
            "dense_score": c.get("dense_score"),
            "bm25_score": c.get("bm25_score"),
            "rrf_score": c.get("rrf_score"),
            "rerank_score": c.get("rerank_score"),
            # Phase 3 — enriched grounding
            "chunk_id": c.get("chunk_id"),
            "collection_id": c.get("collection_id"),
            # Phase 3 — claim linkage
            "claim_text": claim_index_map.get(idx),
        }
        for idx, c in enumerate(chunks)
    ]


def _is_ambiguous_query(query: str) -> bool:
    """Only rewrite query if it is short or contains ambiguous pronouns."""
    words = query.lower().split()
    if len(words) < 6:
        return True
    ambiguous_keywords = {"it", "this", "that", "about", "they", "them", "those", "these", "here", "there"}
    if any(w in ambiguous_keywords for w in words):
        return True
    return False


async def _generate_query_variants(question: str) -> List[str]:
    """Generate 3 semantic variants locally and deterministically, bypassing Gemini."""
    import re
    clean_q = re.sub(r'[^\w\s]', '', question).lower().strip()
    words = clean_q.split()
    stop_words = {"what", "is", "the", "for", "a", "an", "of", "in", "on", "at", "by", "to", "and", "or", "are", "do", "does", "did", "when", "how", "why"}
    keywords = [w for w in words if w not in stop_words]

    variants = [question]
    if keywords:
        variants.append(" ".join(keywords))
        if len(keywords) >= 2:
            variants.append(f"{' '.join(keywords)} details insurance policy")
        else:
            variants.append(f"{question} terms clauses")
    else:
        variants.extend([f"{question} insurance", f"{question} policy"])

    return list(dict.fromkeys(variants))[:3]


async def _call_ollama(prompt: str) -> str:
    """Local Ollama generation fallback if Gemini is offline/limit reached."""
    model = "llama3.1:8b"
    try:
        import torch
        if torch.cuda.is_available():
            model = "qwen2.5:14b"
    except ImportError:
        if getattr(settings, "RERANKER_USE_GPU", False):
            model = "qwen2.5:14b"

    logger.info("Routing query generation to Ollama local instance", extra={"model": model})
    import httpx
    try:
        async with httpx.AsyncClient(timeout=45.0) as client_http:
            res = await client_http.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            if res.status_code == 200:
                return res.json().get("response", "").strip()
            logger.warning(
                "Ollama returned non-200 status",
                extra={"status_code": res.status_code},
            )
    except Exception as exc:
        logger.error(
            "Local Ollama fallback generation failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def answer_single_question(
    question: str,
    namespace: str,
    session_id: Optional[str] = None,
    collection_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Full RAG pipeline (Phase 3):
      cache → guard → follow-up resolve → rewrite → retrieve →
      multi-query expand → rerank → compress → synthesize → verify → log
    """
    start_time = time.time()

    # 1. Cache lookup
    cached = get_cached_answer(question, namespace=namespace)
    if cached:
        cached["cache_hit"] = True
        if metrics:
            metrics.cache_hits_total.labels(namespace=namespace or "global").inc()
        logger.debug("Cache hit", extra={"question": question[:60]})
        return cached

    if metrics:
        metrics.cache_misses_total.labels(namespace=namespace or "global").inc()

    # 2. Query guard
    if not is_valid_rag_query(question):
        return {
            "question": question,
            "answer": "I could not find this information in the document.",
            "sources": [],
            "citations": [],
            "evidence_by_document": {},
            "conflicts": [],
            "document_count": 0,
            "synthesis_mode": "single_doc",
            "cache_hit": False,
        }

    # 3. Follow-up resolution (Phase 3) — before query rewrite
    history = get_history(session_id) if session_id else []
    if session_id and is_followup_question(question):
        question_for_retrieval = resolve_followup(question, history, session_id)
        logger.debug("Follow-up resolved", extra={"original": question, "resolved": question_for_retrieval[:80]})
    else:
        question_for_retrieval = question

    # 4. Merge session-scoped document/collection context into query filters
    if session_id:
        if not document_ids:
            session_doc_ids = get_active_document_ids(session_id)
            if session_doc_ids:
                document_ids = session_doc_ids
        if not collection_ids:
            session_col = get_active_collection_id(session_id)
            if session_col:
                collection_ids = [session_col]

    # 5. Conditional Query rewrite
    if _is_ambiguous_query(question_for_retrieval):
        from services.query_rewriter import rewrite_query
        rewritten_query = await rewrite_query(question=question_for_retrieval, session_id=session_id)
    else:
        rewritten_query = question_for_retrieval

    # 6. Dynamic top_k
    dynamic_topk = compute_dynamic_topk(question)

    # 7. Retrieval
    with track_latency(metrics.retrieval_latency_ms if metrics else None):
        retrieved_chunks: List[Dict] = await asyncio.to_thread(
            retrieve_contexts,
            query=rewritten_query,
            top_k=dynamic_topk * 3,
            namespace=namespace,
            collection_ids=collection_ids,
            document_ids=document_ids,
        )

    # 8. Conditional Multi-Query Expansion
    retrieval_confidence = retrieved_chunks[0].get("dense_score", retrieved_chunks[0].get("score", 0.0)) if retrieved_chunks else 0.0

    if retrieval_confidence < 0.65:
        logger.info(
            "Low retrieval confidence — performing Multi-Query Expansion",
            extra={"confidence": round(retrieval_confidence, 4)},
        )
        variants = await _generate_query_variants(rewritten_query)
        retrieved_chunks = await asyncio.to_thread(
            retrieve_contexts,
            query=variants,
            top_k=dynamic_topk * 3,
            namespace=namespace,
            collection_ids=collection_ids,
            document_ids=document_ids,
        )

    if not retrieved_chunks:
        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": "I could not find this information in the document.",
            "sources": [],
            "citations": [],
            "evidence_by_document": {},
            "conflicts": [],
            "document_count": 0,
            "synthesis_mode": "single_doc",
            "cache_hit": False,
        }

    # 9. Reranking
    with track_latency(metrics.reranker_latency_ms if metrics else None):
        reranked_chunks = rerank_chunks(
            query=question,
            chunks=retrieved_chunks,
            top_n=dynamic_topk,
        )

    # 10. Compression (local sentence deduplication)
    compressed_texts = await asyncio.gather(
        *[compress_chunk(c["chunk_text"]) for c in reranked_chunks]
    )
    compressed_chunks: List[Dict] = [
        {**c, "chunk_text": text}
        for c, text in zip(reranked_chunks, compressed_texts)
    ]

    # 11. Synthesis (Phase 3 — replaces inline generate_answer)
    history_block = _format_history(history)
    with track_latency(metrics.generation_latency_ms if metrics else None):
        synthesis = await synthesize_answer(
            question=question,
            chunks=compressed_chunks,
            history_block=history_block,
            session_id=session_id,
            namespace=namespace,
        )

    # Inline generation quality scoring (heuristic, no extra LLM call)
    gen_scores = score_generation(
        answer=synthesis["answer"],
        chunks=compressed_chunks,
        evidence_by_document=synthesis.get("evidence_by_document", {}),
        synthesis_mode=synthesis.get("synthesis_mode", "single_doc"),
    )

    result: Dict = {
        "question": question,
        "rewritten_query": rewritten_query,
        "answer": synthesis["answer"],
        "sources": synthesis["sources"],
        "citations": _build_citations(
            compressed_chunks,
            claim_index_map=synthesis.get("claim_index_map", {}),
        ),
        "evidence_by_document": synthesis.get("evidence_by_document", {}),
        "conflicts": synthesis.get("conflicts", []),
        "structured_sources": synthesis.get("structured_sources", []),
        "document_count": synthesis.get("document_count", 1),
        "synthesis_mode": synthesis.get("synthesis_mode", "single_doc"),
        "generation_scores": gen_scores.to_dict(),
        "cache_hit": False,
    }

    # 12. Persist session memory (Phase 3 — track document_ids used)
    if session_id:
        used_doc_ids = list({c.get("document_id") for c in compressed_chunks if c.get("document_id")})
        used_sources = list({c.get("source") for c in compressed_chunks if c.get("source")})
        append_message(
            session_id, "user", question,
            namespace=namespace,
            document_ids=used_doc_ids or used_sources,
        )
        append_message(
            session_id, "assistant", result["answer"],
            namespace=namespace,
        )
        trim_history(session_id)

    # 13. Verification
    verified = True
    if settings.VERIFICATION_ENABLED:
        if "could not find this information in the document" in result["answer"].lower():
            verified = True
        else:
            verified = await verify_answer(
                question=question,
                answer=result["answer"],
                context_chunks=compressed_chunks,
            )
            if not verified:
                result["answer"] = "I could not find this information in the document."

    result["verified"] = verified

    if metrics:
        metrics.verification_total.labels(
            result="verified" if verified else "unsupported"
        ).inc()

    # 14. Cache
    store_cached_answer(question, result, namespace=namespace)

    # 15. Analytics
    latency_ms = (time.time() - start_time) * 1000
    final_context = "\n\n".join(
        f"[Source: {c['source']}] {c['chunk_text']}"
        for c in compressed_chunks
    )
    log_query_event(
        question=question,
        answer=result["answer"],
        namespace=namespace,
        latency_ms=latency_ms,
        verified=verified,
        cache_hit=False,
        retrieved_chunks=retrieved_chunks,
        reranked_chunks=reranked_chunks,
        final_context=final_context,
    )

    if metrics:
        metrics.rag_query_total.labels(
            namespace=namespace or "global",
            verified=str(verified),
            cache_hit="false",
        ).inc()
        metrics.rag_latency_ms.observe(latency_ms)

    result["latency_ms"] = round(latency_ms, 2)
    return result


async def answer_questions(
    questions: List[str],
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    collection_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
) -> Dict:
    """Execute multiple questions concurrently."""
    tasks = [
        answer_single_question(
            q,
            namespace=namespace or "",
            session_id=session_id,
            collection_ids=collection_ids,
            document_ids=document_ids,
        )
        for q in questions
    ]
    results = await asyncio.gather(*tasks)
    return {"answers": list(results)}


# ─────────────────────────────────────────────────────────────────────────────
# Streaming (kept separate — synthesizer not used here to preserve token streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def stream_generate_answer(
    question: str,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    collection_ids: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
):
    """
    Streaming answer generator (async generator).
    Yields raw text tokens for SSE/StreamingResponse.
    Uses single-doc prompt for streaming (structured parsing not possible mid-stream).
    """
    if not is_valid_rag_query(question):
        yield "I could not find this information in the document."
        return

    history = get_history(session_id) if session_id else []
    if session_id and is_followup_question(question):
        question_for_retrieval = resolve_followup(question, history, session_id)
    else:
        question_for_retrieval = question

    if _is_ambiguous_query(question_for_retrieval):
        from services.query_rewriter import rewrite_query
        rewritten_query = await rewrite_query(question=question_for_retrieval, session_id=session_id)
    else:
        rewritten_query = question_for_retrieval

    dynamic_topk = compute_dynamic_topk(question)

    retrieved_chunks: List[Dict] = await asyncio.to_thread(
        retrieve_contexts,
        query=rewritten_query,
        top_k=dynamic_topk * 3,
        namespace=namespace,
        collection_ids=collection_ids,
        document_ids=document_ids,
    )

    retrieval_confidence = retrieved_chunks[0].get("dense_score", retrieved_chunks[0].get("score", 0.0)) if retrieved_chunks else 0.0

    if retrieval_confidence < 0.65:
        variants = await _generate_query_variants(rewritten_query)
        retrieved_chunks = await asyncio.to_thread(
            retrieve_contexts,
            query=variants,
            top_k=dynamic_topk * 3,
            namespace=namespace,
            collection_ids=collection_ids,
            document_ids=document_ids,
        )

    if not retrieved_chunks:
        yield "I could not find this information in the document."
        return

    reranked_chunks = rerank_chunks(query=question, chunks=retrieved_chunks, top_n=dynamic_topk)

    context_text = "\n\n".join(
        f"[Source: {c['source']}] {c.get('chunk_text', '')}"
        for c in reranked_chunks
    )

    history_block = _format_history(history)
    prompt = QA_PROMPT_TEMPLATE.format(
        history_block=history_block,
        context_text=context_text,
        question=question,
    )

    try:
        if settings.GOOGLE_API_KEY:
            stream = client.models.generate_content_stream(
                model=settings.GEMINI_GENERATION_MODEL,
                contents=prompt,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        else:
            answer_text = await _call_ollama(prompt)
            yield answer_text if answer_text else "I could not find this information in the document."
    except Exception as exc:
        logger.error(
            "Streaming generation failed (Gemini) — attempting Ollama fallback",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
            exc_info=True,
        )
        try:
            answer_text = await _call_ollama(prompt)
            if answer_text:
                yield answer_text
            else:
                logger.error(
                    "Ollama fallback also failed in streaming path — both LLMs unavailable",
                    extra={"question_head": question[:80]},
                )
                yield "I could not find this information in the document."
        except Exception as fallback_exc:
            logger.error(
                "Ollama fallback raised exception in streaming path",
                extra={"error_type": type(fallback_exc).__name__, "error": str(fallback_exc)},
                exc_info=True,
            )
            yield "I could not find this information in the document."