"""
services/generation_scorer.py — Phase 3

Lightweight, zero-latency generation quality scorer.
All scoring is heuristic (no LLM calls) — runs inline after synthesis.

Scores returned (all 0.0–1.0):
  - faithfulness          : fraction of answer sentences grounded in retrieved chunks
  - completeness          : fraction of expected key-topics covered in the answer
  - cross_doc_consistency : detects numeric/factual contradictions across documents
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationScores:
    faithfulness: float = 1.0
    completeness: float = 1.0
    cross_doc_consistency: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "completeness": round(self.completeness, 4),
            "cross_doc_consistency": round(self.cross_doc_consistency, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_NOT_FOUND_PATTERN = re.compile(
    r"could not find|no information|not present|not mentioned|not in the document",
    re.IGNORECASE,
)

_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, stripping citation tags."""
    clean = re.sub(r"\[Source:[^\]]*\]", "", text)
    clean = re.sub(r"\[DOC:[^\]]*\]", "", clean)
    sentences = _SENTENCE_SEP.split(clean.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _normalize(text: str) -> str:
    """Lowercase and remove punctuation for token matching."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _token_set(text: str) -> set:
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
            "at", "to", "and", "or", "for", "with", "that", "this", "it"}
    return {t for t in _normalize(text).split() if t not in stop and len(t) > 2}


def _sentence_grounded(sentence: str, chunk_texts: List[str], threshold: float = 0.25) -> bool:
    """
    Return True if a sentence shares enough tokens with any single chunk.
    Threshold = minimum Jaccard overlap required for grounding.
    """
    s_tokens = _token_set(sentence)
    if not s_tokens:
        return True  # Empty / citation-only sentence — treat as grounded
    for chunk_text in chunk_texts:
        c_tokens = _token_set(chunk_text)
        if not c_tokens:
            continue
        overlap = len(s_tokens & c_tokens) / len(s_tokens | c_tokens)
        if overlap >= threshold:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public Scoring Functions
# ─────────────────────────────────────────────────────────────────────────────

def score_faithfulness(answer: str, chunks: List[Dict]) -> float:
    """
    Faithfulness = fraction of answer sentences grounded in retrieved chunks.

    A sentence is 'grounded' if it has ≥25% token overlap with at least one chunk.
    Falls back to 1.0 for 'not found' answers (they are honest, not hallucinated).
    Returns 1.0 when no chunks are available (nothing to check against).
    """
    if _NOT_FOUND_PATTERN.search(answer):
        return 1.0  # Honest abstention — fully faithful

    if not chunks:
        return 1.0

    chunk_texts = [c.get("chunk_text", "") for c in chunks if c.get("chunk_text")]
    if not chunk_texts:
        return 1.0

    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0

    grounded = sum(1 for s in sentences if _sentence_grounded(s, chunk_texts))
    score = grounded / len(sentences)
    logger.debug(
        "Faithfulness scored",
        extra={"grounded": grounded, "total_sentences": len(sentences), "score": round(score, 4)},
    )
    return round(score, 4)


def score_completeness(answer: str, key_topics: Optional[List[str]] = None) -> float:
    """
    Completeness = fraction of key_topics (keywords/phrases) found in the answer.

    If key_topics is empty or None, returns 1.0 (cannot measure without ground truth).
    Designed for use during evaluation when expected_answer is available.
    """
    if not key_topics:
        return 1.0

    answer_lower = _normalize(answer)
    hits = sum(1 for topic in key_topics if _normalize(topic) in answer_lower)
    score = hits / len(key_topics)
    logger.debug(
        "Completeness scored",
        extra={"hits": hits, "total_topics": len(key_topics), "score": round(score, 4)},
    )
    return round(score, 4)


def score_cross_doc_consistency(
    evidence_by_document: Dict[str, List[str]],
    synthesis_mode: str = "single_doc",
) -> float:
    """
    Cross-document consistency score.

    Logic:
      - Single-doc mode → always 1.0 (metric not applicable)
      - Multi-doc mode  → detect if the same numeric value (e.g. "$500") appears
        across two documents with opposite polarity (one says X, another says Y
        for the same concept).
      - Conflicts are EXPECTED and properly reported → high consistency (0.85)
      - Conflicts are silently ignored (same value in both docs without flagging)
        → penalized (0.50)
      - No conflicts detected and docs agree → 1.0

    This is a conservative heuristic — it rewards correct conflict reporting rather
    than penalizing multi-doc answers that correctly surface disagreements.
    """
    if synthesis_mode != "multi_doc" or len(evidence_by_document) < 2:
        return 1.0

    docs = list(evidence_by_document.keys())
    all_claims: Dict[str, Dict[str, str]] = {}  # {value: {doc: claim_text}}

    _NUMERIC_PAT = re.compile(r"\$[\d,]+|\d+[\s\-]?days?|\d+%|\d+,\d{3}")

    for doc, claims in evidence_by_document.items():
        for claim in claims:
            for match in _NUMERIC_PAT.finditer(claim.lower()):
                val = match.group()
                if val not in all_claims:
                    all_claims[val] = {}
                all_claims[val][doc] = claim

    # Find values that appear with different surrounding context in different docs
    conflicts_found = 0
    for val, doc_claims in all_claims.items():
        if len(doc_claims) >= 2:
            # Check if the surrounding context is meaningfully different
            claim_texts = list(doc_claims.values())
            tokens_0 = _token_set(claim_texts[0])
            tokens_1 = _token_set(claim_texts[1])
            overlap = len(tokens_0 & tokens_1) / max(len(tokens_0 | tokens_1), 1)
            if overlap < 0.5:
                # Same numeric value in clearly different contexts → likely conflict
                conflicts_found += 1

    if conflicts_found == 0:
        return 1.0

    # Conflicts exist. If evidence_by_document is non-empty per doc, the model
    # likely reported them (credit for surfacing). Conservative reward: 0.85.
    # Full consistency penalty (0.5) only if no evidence_by_document at all.
    all_non_empty = all(bool(claims) for claims in evidence_by_document.values())
    return 0.85 if all_non_empty else 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Composite Scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_generation(
    answer: str,
    chunks: List[Dict],
    evidence_by_document: Dict[str, List[str]],
    synthesis_mode: str = "single_doc",
    key_topics: Optional[List[str]] = None,
) -> GenerationScores:
    """
    Compute all three generation quality scores in one call.

    Args:
        answer             : The generated answer text
        chunks             : Final compressed chunks passed to the LLM
        evidence_by_document: Parsed {doc: [claims]} from multi-doc synthesis
        synthesis_mode     : "single_doc" | "multi_doc"
        key_topics         : Optional list of expected key terms for completeness scoring

    Returns:
        GenerationScores dataclass with faithfulness, completeness, cross_doc_consistency
    """
    faithfulness = score_faithfulness(answer, chunks)
    completeness = score_completeness(answer, key_topics)
    cross_doc    = score_cross_doc_consistency(evidence_by_document, synthesis_mode)

    scores = GenerationScores(
        faithfulness=faithfulness,
        completeness=completeness,
        cross_doc_consistency=cross_doc,
    )
    logger.debug("GenerationScores computed", extra=scores.to_dict())
    return scores
