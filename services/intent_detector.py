"""
services/intent_detector.py — Query Intent Classification

Classifies queries into intent categories that drive:
  1. Model selection (Flash vs Pro via model_router)
  2. Synthesis prompt selection (factual vs comparative vs aggregation)
  3. Retrieval strategy adjustments (top_k, fusion weights)

Intent categories:
  FACTUAL     → Direct lookup: "What is the deductible for Plan A?"
  COMPARATIVE → Compare: "How does Plan A differ from Plan B?"
  AGGREGATION → Summarize: "List all clauses about termination."
  TEMPORAL    → Timeline: "What changed in the 2023 amendment?"
  PROCEDURAL  → How-to: "How do I file a claim?"
  AMBIGUOUS   → Unclear scope — use FACTUAL behavior as safe default

Classification is rule-based (regex pattern scoring) — no API call.
Average latency: <1ms.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from models.domain import QueryIntent
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intent rules: (pattern, weight, intent)
# Scored across all patterns; highest-scoring intent wins.
# Minimum score threshold of 2.0 required to assign non-AMBIGUOUS intent.
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_RULES: List[Tuple[re.Pattern, float, QueryIntent]] = [
    # ── COMPARATIVE ──────────────────────────────────────────────────────────
    (re.compile(r"\b(compar|differ|vs\.?|versus|contrast|between|distinguish|both|neither)\b", re.I), 3.0, QueryIntent.COMPARATIVE),
    (re.compile(r"\bhow\s+(does|do|is)\s+.{3,30}\s+(differ|compare)\b", re.I), 2.5, QueryIntent.COMPARATIVE),
    (re.compile(r"\b(same|similar|different|unlike|like)\s+\w+\b", re.I), 1.5, QueryIntent.COMPARATIVE),

    # ── AGGREGATION ───────────────────────────────────────────────────────────
    (re.compile(r"\b(list|all|every|enumerate|summarize|overview|provide\s+a\s+summary|aggregate)\b", re.I), 3.0, QueryIntent.AGGREGATION),
    (re.compile(r"\bhow\s+many\b", re.I), 2.5, QueryIntent.AGGREGATION),
    (re.compile(r"\b(total|count|number\s+of|complete\s+list)\b", re.I), 2.0, QueryIntent.AGGREGATION),
    (re.compile(r"\bwhat\s+are\s+(all|the\s+main|the\s+key|the\s+different)\b", re.I), 2.0, QueryIntent.AGGREGATION),

    # ── TEMPORAL ─────────────────────────────────────────────────────────────
    (re.compile(r"\b(when|before|after|since|until|timeline|chronolog|history|date|year|amendment|update)\b", re.I), 2.5, QueryIntent.TEMPORAL),
    (re.compile(r"\b(has\s+changed|was\s+updated|previously|formerly|originally|latest|most\s+recent)\b", re.I), 2.0, QueryIntent.TEMPORAL),
    (re.compile(r"\bin\s+\d{4}\b|\b(q[1-4]\s*\d{4}|\d{4}\s*q[1-4])\b", re.I), 2.0, QueryIntent.TEMPORAL),

    # ── PROCEDURAL ───────────────────────────────────────────────────────────
    (re.compile(r"\bhow\s+(do\s+i|can\s+i|to|should\s+i|do\s+you)\b", re.I), 3.0, QueryIntent.PROCEDURAL),
    (re.compile(r"\b(steps?\s+to|process\s+for|instructions?\s+to|procedure\s+for|guide\s+to)\b", re.I), 2.5, QueryIntent.PROCEDURAL),
    (re.compile(r"\b(file\s+a|submit\s+a|apply\s+for|register|set\s+up|configure)\b", re.I), 2.0, QueryIntent.PROCEDURAL),

    # ── FACTUAL ──────────────────────────────────────────────────────────────
    (re.compile(r"\bwhat\s+is\b|\bwho\s+is\b|\bwhere\s+is\b|\bwhich\s+is\b", re.I), 2.0, QueryIntent.FACTUAL),
    (re.compile(r"\b(define|definition\s+of|meaning\s+of|explain)\b", re.I), 1.5, QueryIntent.FACTUAL),
    (re.compile(r"\b(amount|percentage|rate|value|limit|threshold|maximum|minimum)\b", re.I), 1.5, QueryIntent.FACTUAL),
]

_MIN_SCORE = 2.0   # Minimum score to assign non-AMBIGUOUS intent


def detect_intent(query: str) -> QueryIntent:
    """
    Classify query intent using weighted keyword pattern matching.

    Args:
        query: The raw query string (after cleaning/rewriting).

    Returns:
        QueryIntent enum value. Defaults to FACTUAL for low-confidence cases.
    """
    scores: dict[QueryIntent, float] = {intent: 0.0 for intent in QueryIntent}

    for pattern, weight, intent in _INTENT_RULES:
        if pattern.search(query):
            scores[intent] += weight

    # Special case: queries with 2+ documents mentioned are likely comparative
    doc_mentions = len(re.findall(r"(?:document|doc|report|policy|plan|contract|file)\s*[A-Z]?\d*", query, re.I))
    if doc_mentions >= 2:
        scores[QueryIntent.COMPARATIVE] += 3.0

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score < _MIN_SCORE:
        # Low confidence — default to FACTUAL (safest behavior)
        intent = QueryIntent.FACTUAL
    else:
        intent = best_intent

    logger.debug(
        "Intent detected",
        extra={
            "query_head": query[:80],
            "intent": intent.value,
            "score": round(best_score, 2),
            "all_scores": {k.value: round(v, 2) for k, v in scores.items() if v > 0},
        },
    )

    return intent


def get_top_k_for_intent(intent: QueryIntent, base_top_k: int = 5) -> int:
    """
    Adjust retrieval top_k based on intent.
    Aggregation queries need more results to ensure completeness.
    """
    intent_multipliers = {
        QueryIntent.FACTUAL:     1.0,
        QueryIntent.PROCEDURAL:  1.0,
        QueryIntent.AMBIGUOUS:   1.0,
        QueryIntent.TEMPORAL:    1.2,
        QueryIntent.COMPARATIVE: 1.5,
        QueryIntent.AGGREGATION: 2.0,
    }
    multiplier = intent_multipliers.get(intent, 1.0)
    return min(20, max(3, round(base_top_k * multiplier)))
