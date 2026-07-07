"""
services/query_guard.py — Enterprise Query Guard

Enforces safety rules on incoming queries before the RAG pipeline runs:

1. Prompt Injection Detection — detect attempts to override system instructions
2. Jailbreak Pattern Matching — detect adversarial patterns
3. Query Length Enforcement — reject unreasonably long queries
4. Off-Topic / Non-RAG Query Detection — reject questions unrelated to documents

All detections are logged to the audit log.
On detection, the guard raises QueryBlockedError with a reason code.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Block reasons ─────────────────────────────────────────────────────────────

class BlockReason(str, Enum):
    PROMPT_INJECTION   = "prompt_injection"
    JAILBREAK          = "jailbreak"
    QUERY_TOO_LONG     = "query_too_long"
    EMPTY_QUERY        = "empty_query"
    OFF_TOPIC          = "off_topic"


class QueryBlockedError(Exception):
    """Raised when a query is blocked by the guard."""
    def __init__(self, reason: BlockReason, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"Query blocked ({reason}): {detail}")


# ── Length limits ─────────────────────────────────────────────────────────────

_MAX_QUERY_CHARS = 2000    # ~500 words — more than enough for any legitimate question
_MIN_QUERY_CHARS = 2       # must have at least 2 non-whitespace characters


# ── Prompt injection patterns ─────────────────────────────────────────────────
# These target the most common techniques used to override system prompts.
# Pattern design principles:
#   - Match the *intent*, not just exact phrasing
#   - Use word boundaries to reduce false positives
#   - Avoid blocking legitimate questions about AI/instructions

_INJECTION_PATTERNS: list[re.Pattern] = [
    # Direct instruction override
    re.compile(r"\bignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|context|rules?)\b", re.I),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|context|rules?)\b", re.I),
    re.compile(r"\bforget\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|context)\b", re.I),

    # System prompt leakage attempts
    re.compile(r"\bprint\s+(your\s+)?(system\s+prompt|instructions|prompt)\b", re.I),
    re.compile(r"\bshow\s+me\s+(your\s+)?(system\s+prompt|instructions|full\s+prompt)\b", re.I),
    re.compile(r"\brepeat\s+(your\s+)?(system\s+prompt|instructions|prompt)\b", re.I),
    re.compile(r"\bwhat\s+(are|is)\s+your\s+(system\s+)?instructions\b", re.I),

    # Role-switching attacks
    re.compile(r"\bact\s+as\s+(if\s+you\s+(are|were)|an?\s+)?(DAN|evil|unconstrained|unfiltered|jailbreak)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(DAN|an?\s+unrestricted|an?\s+evil|an?\s+uncensored)\b", re.I),
    re.compile(r"\bpretend\s+(you\s+are|you're|to\s+be)\s+(an?\s+)?AI\s+(without|with\s+no)\s+restrictions\b", re.I),

    # Delimiter injection (trying to inject new prompt sections)
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\[INST\]|\[SYS\]|\[SYSTEM\]", re.I),
    re.compile(r"###\s*(system|instruction|prompt)\s*:", re.I),

    # End-of-context tricks
    re.compile(r"---+\s*(new\s+)?(instruction|task|prompt|context)", re.I),
]


# ── Jailbreak patterns ────────────────────────────────────────────────────────

_JAILBREAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bDAN\b"),   # "Do Anything Now"
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bunrestricted\s+mode\b", re.I),
    re.compile(r"\bdev\s+mode\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\bgod\s+mode\b", re.I),
    re.compile(r"\bbypass\s+(all\s+)?(safety|restrictions|guidelines|filters)\b", re.I),
]


# ── Off-topic detection ───────────────────────────────────────────────────────
# Flag queries that are clearly unrelated to document retrieval.
# We use an allowlist approach: if the query contains any content keyword, allow it.
# This avoids false positives on legitimate questions.

_OFF_TOPIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(write|generate|create|make)\s+(me\s+)?(a|an|some)?\s*(poem|song|story|joke|code|program|script|essay|letter|email\s+to|cover\s+letter)\b", re.I),
    re.compile(r"\bplay\s+(a\s+)?(game|chess|word\s+game)\b", re.I),
    re.compile(r"\bgenerate\s+(random|fake)\s+(data|names|passwords?)\b", re.I),
    re.compile(r"\b(hack|exploit|vulnerability|malware|ransomware)\b", re.I),
]


# ── Public interface ──────────────────────────────────────────────────────────

def validate_query(
    query: str,
    session_id: Optional[str] = None,
    user: Optional[dict] = None,
) -> str:
    """
    Validate an incoming query through all guard layers.

    Returns the (possibly stripped) query if valid.
    Raises QueryBlockedError with a reason code if blocked.

    Args:
        query:      The raw query string.
        session_id: Optional session ID for audit logging.
        user:       Optional user dict for audit logging.

    Returns:
        Cleaned query string (whitespace stripped).

    Raises:
        QueryBlockedError on any rule violation.
    """
    from config import settings

    cleaned = query.strip()

    # 1. Empty query
    if len(cleaned) < _MIN_QUERY_CHARS:
        raise QueryBlockedError(
            BlockReason.EMPTY_QUERY,
            "Query is empty or too short.",
        )

    # 2. Query too long
    if len(cleaned) > _MAX_QUERY_CHARS:
        raise QueryBlockedError(
            BlockReason.QUERY_TOO_LONG,
            f"Query exceeds maximum length ({len(cleaned)} / {_MAX_QUERY_CHARS} characters).",
        )

    if not settings.PROMPT_INJECTION_ENABLED:
        return cleaned

    # 3. Prompt injection detection
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            _log_blocked(BlockReason.PROMPT_INJECTION, cleaned, session_id, user)
            raise QueryBlockedError(
                BlockReason.PROMPT_INJECTION,
                "Query contains patterns associated with prompt injection attacks.",
            )

    # 4. Jailbreak detection
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(cleaned):
            _log_blocked(BlockReason.JAILBREAK, cleaned, session_id, user)
            raise QueryBlockedError(
                BlockReason.JAILBREAK,
                "Query contains patterns associated with jailbreak attempts.",
            )

    # 5. Off-topic detection (soft — only flag clear non-RAG requests)
    for pattern in _OFF_TOPIC_PATTERNS:
        if pattern.search(cleaned):
            _log_blocked(BlockReason.OFF_TOPIC, cleaned, session_id, user)
            raise QueryBlockedError(
                BlockReason.OFF_TOPIC,
                "This system is for querying documents. Please ask questions about your uploaded content.",
            )

    return cleaned


def is_valid_rag_query(query: str) -> bool:
    """
    Legacy compatibility shim used by existing code paths.
    Returns True if valid, False if blocked (no exception).
    """
    try:
        validate_query(query)
        return True
    except QueryBlockedError:
        return False


def _log_blocked(
    reason: BlockReason,
    query: str,
    session_id: Optional[str],
    user: Optional[dict],
) -> None:
    """Log a blocked query to audit and application logs."""
    username = user.get("username", "anonymous") if user else "anonymous"
    logger.warning(
        "Query blocked by guard",
        extra={
            "reason": reason.value,
            "session_id": session_id,
            "username": username,
            # Truncate query in logs — avoid leaking full injection attempt
            "query_head": query[:100],
        },
    )
    try:
        from services.audit import log_audit_event
        log_audit_event(
            action="query_blocked",
            user=username,
            detail={"reason": reason.value, "query_head": query[:100]},
            success=False,
        )
    except Exception:
        pass   # Never let audit failure block the guard