"""
services/synthesizer.py — Phase 3 (Enterprise v3)

Responsibilities:
  - Group retrieved chunks by source document
  - Format context with labeled [DOC | Page N | Chunk seq] headers for the LLM
  - Select the appropriate prompt (single-doc vs multi-doc) based on intent
  - Route to appropriate model via model_router (circuit breaker + intent routing)
  - Parse the structured multi-doc JSON response into machine-readable fields:
      * evidence_by_document  : {doc: [claims]}
      * conflicts             : [conflict strings]
      * structured_sources    : [{doc, page, chunk_seq, description}]
  - Link claims back to citations via claim_text
  - Return an enriched result dict

v3 changes:
  - Uses model_router.generate() instead of direct Gemini calls
  - Adds intent parameter for model selection (comparative → Pro)
  - Removes duplicate Ollama fallback (handled by model_router)
"""
from __future__ import annotations

import re
import json
import logging
from typing import Dict, List, Optional, Tuple

from config import settings
from services.prompts import SINGLE_DOC_QA_PROMPT, MULTI_DOC_SYNTHESIS_PROMPT
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Context Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _format_chunk_label(chunk: Dict, seq: int) -> str:
    """Build the [DOC: ... | Page N | Chunk seq] label for a chunk."""
    source   = chunk.get("source", "unknown")
    page     = chunk.get("page")
    page_str = f"Page {page}" if page is not None else "Page ?"
    return f"[DOC: {source} | {page_str} | Chunk {seq}]"


def format_multi_doc_context(chunks: List[Dict]) -> Tuple[str, Dict[str, int]]:
    """
    Format chunks into labeled blocks for the multi-doc synthesis prompt.

    Returns:
        context_text  : formatted string for prompt injection
        chunk_seq_map : {str(chunk_index): per-doc sequence number}
    """
    parts: List[str]        = []
    chunk_seq_map: Dict[str, int] = {}
    doc_counters: Dict[str, int]  = {}

    for idx, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        doc_counters[source] = doc_counters.get(source, 0) + 1
        seq = doc_counters[source]
        chunk_seq_map[str(idx)] = seq

        label = _format_chunk_label(chunk, seq)
        text  = chunk.get("chunk_text", "").strip()
        parts.append(f"{label}\n{text}")

    return "\n\n".join(parts), chunk_seq_map


def format_single_doc_context(chunks: List[Dict]) -> str:
    """Format chunks as a flat context block (original behavior)."""
    return "\n\n".join(
        f"[Source: {c.get('source', 'unknown')}] {c.get('chunk_text', '')}"
        for c in chunks
    )


def group_chunks_by_source(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """Group chunk list by source document name."""
    groups: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        groups.setdefault(source, []).append(chunk)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Doc JSON Schema
# ─────────────────────────────────────────────────────────────────────────────

_MULTIDOC_JSON_PROMPT = """
You are an expert analyst synthesizing information across multiple documents.
Answer ONLY from the provided context — never from memory or external knowledge.

CRITICAL RULES:
1. Every factual claim MUST include a chunk-level inline citation: [Source: <doc>, Page <N>, Chunk <seq>]
2. Actively compare documents. Report EVERY case where two documents give DIFFERENT values for the same item.
3. In the conflicts list, include ALL numeric/date/term differences found. Do NOT write an empty conflicts list if differences exist.
4. evidence_by_document: for each document, list 2–5 key claims it supports.
5. structured_sources: one entry per chunk actually cited in your answer.
6. IMPORTANT: The context below contains relevant information. You MUST use it to answer the question.
   Do NOT respond with "I could not find" or "not in the document" — the context IS the document.
   If the exact number/term is present in any chunk, cite and use it.

CONVERSATION HISTORY:
{history_block}

DOCUMENT CONTEXT (each chunk labeled [DOC: <name> | Page <N> | Chunk <seq>]):
{context_text}

QUESTION:
{question}

Respond ONLY with a valid JSON object matching EXACTLY this structure:
{{
  "answer": "<synthesized answer with inline citations>",
  "evidence_by_document": {{
    "<document name>": ["<claim 1>", "<claim 2>"]
  }},
  "conflicts": [
    "<Document A> states <X>, but <Document B> states <Y>."
  ],
  "structured_sources": [
    {{"doc": "<document name>", "page": <int or null>, "chunk_seq": <int>, "description": "<one-line description>"}}
  ]
}}

If no conflicts exist after comparing all values, set "conflicts" to [].
"""


# ─────────────────────────────────────────────────────────────────────────────
# Structured Response Parser (JSON — Fix 3: no regex fragility)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(
    raw_text: str,
) -> Tuple[str, Dict[str, List[str]], List[str], List[Dict]]:
    """
    Parse Gemini's JSON multi-doc response.

    Attempts (in order):
      1. Strict json.loads on stripped text.
      2. Strip markdown fences (```json ... ```) and retry.
      3. Locate the first '{' ... last '}' substring (handles prose preambles).

    Returns (answer_text, evidence_by_document, conflicts, structured_sources).
    """
    text = raw_text.strip()

    # Pass 1: strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()

    # Pass 2: try strict parse
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pass 3: extract first JSON object from within prose (handles "Here is the JSON: {...}")
    if data is None:
        brace_start = text.find("{")
        brace_end   = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            candidate = text[brace_start : brace_end + 1]
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Multi-doc JSON parse failed after all extraction attempts",
                    extra={"error": str(exc), "raw_snippet": raw_text[:400]},
                )
                # Return raw text as best-effort answer rather than "not found"
                return raw_text.strip(), {}, [], []

    if data is None:
        logger.error(
            "Multi-doc JSON parse: no JSON object found in response",
            extra={"raw_snippet": raw_text[:400]},
        )
        return raw_text.strip(), {}, [], []

    answer_text        = data.get("answer", raw_text.strip())
    evidence_by_doc    = {}
    conflicts          = []
    structured_sources = []

    # evidence_by_document: {str: list[str]}
    raw_evidence = data.get("evidence_by_document", {})
    if isinstance(raw_evidence, dict):
        for doc, claims in raw_evidence.items():
            if isinstance(claims, list):
                evidence_by_doc[str(doc)] = [str(c) for c in claims if c]

    # conflicts: list[str]
    raw_conflicts = data.get("conflicts", [])
    if isinstance(raw_conflicts, list):
        conflicts = [str(c) for c in raw_conflicts if c]

    # structured_sources: list[{doc, page, chunk_seq, description}]
    raw_sources = data.get("structured_sources", [])
    if isinstance(raw_sources, list):
        for entry in raw_sources:
            if not isinstance(entry, dict):
                continue
            structured_sources.append({
                "doc":         str(entry.get("doc", "")),
                "page":        int(entry["page"]) if str(entry.get("page", "")).isdigit() else None,
                "chunk_seq":   int(entry.get("chunk_seq", 0)),
                "description": str(entry.get("description", "")),
            })

    return answer_text, evidence_by_doc, conflicts, structured_sources


# Legacy markdown-section parser kept for single-doc plain-text fallback
def _extract_section(text: str, header: str) -> str:
    pattern = rf"##\s*{re.escape(header)}\s*\n(.*?)(?=\n##\s|\Z)"
    match   = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


# ─────────────────────────────────────────────────────────────────────────────
# Claim → Citation Linking
# ─────────────────────────────────────────────────────────────────────────────

def link_claims_to_chunks(
    chunks: List[Dict],
    evidence_by_document: Dict[str, List[str]],
) -> Dict[int, str]:
    """
    For each chunk index, find the best matching claim from evidence_by_document.
    Returns {chunk_index: claim_text_string}.
    """
    index_to_claim: Dict[int, str] = {}
    if not evidence_by_document:
        return index_to_claim

    for idx, chunk in enumerate(chunks):
        source = chunk.get("source", "unknown")
        claims = evidence_by_document.get(source, [])

        if not claims:
            # Fuzzy match on filename stem
            base = source.rsplit(".", 1)[0].lower()
            for doc_key, doc_claims in evidence_by_document.items():
                if base in doc_key.lower():
                    claims = doc_claims
                    break

        if claims:
            chunk_tokens = set(chunk.get("chunk_text", "").lower().split())
            best_claim, best_overlap = "", -1
            for claim in claims:
                claim_tokens = set(claim.lower().split())
                if not claim_tokens:
                    continue
                overlap = len(chunk_tokens & claim_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_claim   = claim
            if best_claim:
                index_to_claim[idx] = best_claim

    return index_to_claim


# ─────────────────────────────────────────────────────────────────────────────
# Model-Router backed generation call
# ─────────────────────────────────────────────────────────────────────────────

async def _call_model(
    prompt: str,
    use_json_mode: bool = False,
    intent=None,
) -> str:
    """
    Call the appropriate LLM via model_router.

    Routes Gemini Flash vs Pro based on intent.
    Falls back to Ollama on circuit-open or retries exhausted.
    Returns response text, or "" on complete failure.
    """
    from services.model_router import generate
    from models.domain import QueryIntent

    try:
        text, model_used = await generate(
            prompt=prompt,
            intent=intent,
            use_json_mode=use_json_mode,
            allow_fallback=True,
        )
        logger.debug("Generation complete", extra={"model": model_used, "len": len(text)})
        return text
    except Exception as exc:
        logger.error(
            "Model router generation failed",
            extra={
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
                "use_json_mode": use_json_mode,
            },
        )
        return ""



# ─────────────────────────────────────────────────────────────────────────────
# Main Synthesis Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def synthesize_answer(
    question: str,
    chunks: List[Dict],
    history_block: str = "",
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    intent=None,
) -> Dict:
    """
    Generate a grounded, structured answer from retrieved chunks.

    Automatically selects:
      - SINGLE_DOC_QA_PROMPT   : when all chunks come from one source → plain text
      - JSON multi-doc prompt  : when chunks span ≥2 sources → JSON response

    Model routing:
      - Comparative/Aggregation/Temporal intent → Gemini Pro
      - All other intents → Gemini Flash
      - Gemini unavailable → Ollama fallback (via model_router)

    Error handling:
      - model_router handles retry + circuit breaker + fallback
      - JSON parse error → logs raw snippet → returns best-effort plain text answer

    Returns a dict with:
      answer, sources, evidence_by_document, conflicts,
      structured_sources, claim_index_map,
      document_count, synthesis_mode, raw_response
    """
    NOT_FOUND = "I could not find this information in the document."

    if not chunks:
        logger.warning("synthesize_answer called with empty chunks")
        return {
            "answer": NOT_FOUND,
            "sources": [], "evidence_by_document": {}, "conflicts": [],
            "structured_sources": [], "claim_index_map": {},
            "document_count": 0, "synthesis_mode": "single_doc", "raw_response": "",
        }

    # ── Mode detection ────────────────────────────────────────────────────────
    source_groups  = group_chunks_by_source(chunks)
    document_count = len(source_groups)
    is_multi_doc   = document_count >= 2

    if is_multi_doc:
        context_text, _ = format_multi_doc_context(chunks)
        # Sanity-check: never send an empty context to Gemini in multi-doc mode
        if not context_text.strip():
            logger.error(
                "Multi-doc synthesis: context_text is empty after formatting — "
                "chunks have no text content",
                extra={"chunk_count": len(chunks), "sources": list(source_groups.keys())},
            )
            return {
                "answer": NOT_FOUND,
                "sources": list(source_groups.keys()),
                "evidence_by_document": {}, "conflicts": [],
                "structured_sources": [], "claim_index_map": {},
                "document_count": document_count, "synthesis_mode": synthesis_mode,
                "raw_response": "",
            }
        prompt = _MULTIDOC_JSON_PROMPT.format(
            history_block=history_block,
            context_text=context_text,
            question=question,
        )
        synthesis_mode = "multi_doc"
        logger.info(
            "Multi-doc synthesis activated",
            extra={"document_count": document_count, "sources": list(source_groups.keys())},
        )
    else:
        context_text  = format_single_doc_context(chunks)
        prompt        = SINGLE_DOC_QA_PROMPT.format(
            history_block=history_block,
            context_text=context_text,
            question=question,
        )
        synthesis_mode = "single_doc"

    # ── Model-router call (handles retry + circuit breaker + fallback) ──────────
    raw_response = await _call_model(
        prompt,
        use_json_mode=is_multi_doc,
        intent=intent,
    )

    if not raw_response:
        logger.error(
            "Model generation failed — returning NOT_FOUND",
            extra={"question": question[:100], "synthesis_mode": synthesis_mode},
        )
        raw_response = NOT_FOUND

    # ── Parse structured output (multi-doc → JSON, single-doc → plain text) ──
    if is_multi_doc:
        answer_text, evidence_by_document, conflicts, structured_sources = \
            _parse_json_response(raw_response)
        # If JSON parse returned the raw text as answer, log it for diagnosis
        if not evidence_by_document and not structured_sources:
            logger.warning(
                "Multi-doc JSON parse yielded no structured fields",
                extra={"raw_snippet": raw_response[:200]},
            )
        claim_index_map = link_claims_to_chunks(chunks, evidence_by_document)
    else:
        answer_text        = raw_response
        evidence_by_document = {}
        conflicts          = []
        structured_sources = []
        claim_index_map    = {}

    return {
        "answer":               answer_text,
        "sources":              list(source_groups.keys()),
        "evidence_by_document": evidence_by_document,
        "conflicts":            conflicts,
        "structured_sources":   structured_sources,
        "claim_index_map":      claim_index_map,
        "document_count":       document_count,
        "synthesis_mode":       synthesis_mode,
        "raw_response":         raw_response,
    }
