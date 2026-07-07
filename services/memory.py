"""
services/memory.py — Phase 3

Improvements:
  - Document-scoped sessions: sessions store active_document_ids + active_collection_id
  - Summarized memory: when history exceeds N turns, oldest turns are compressed
  - Follow-up resolution: resolve pronoun references using session document context
    + inject active_document_ids as query scope hint
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional

from services.db import chat_history_collection
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core History Operations
# ─────────────────────────────────────────────────────────────────────────────

def get_history(session_id: str) -> List[Dict]:
    doc = chat_history_collection.find_one({"session_id": session_id})
    return doc.get("messages", []) if doc else []


def get_session_doc(session_id: str) -> Dict:
    """Return the full session document, or empty dict if not found."""
    return chat_history_collection.find_one({"session_id": session_id}) or {}


def append_message(
    session_id: str,
    role: str,
    content: str,
    namespace: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    collection_id: Optional[str] = None,
) -> None:
    """
    Append a message to session history.
    Also updates document_ids and collection_id if provided (document-scoped sessions).
    Silently upgrades legacy sessions that lack these fields.
    """
    update_doc: Dict = {
        "$push": {"messages": {"role": role, "content": content}}
    }

    set_fields: Dict = {}
    if namespace:
        set_fields["namespace"] = namespace
    if collection_id:
        set_fields["active_collection_id"] = collection_id

    if set_fields:
        update_doc["$set"] = set_fields

    # Track document IDs used in this session (addToSet avoids duplicates)
    if document_ids:
        update_doc["$addToSet"] = {"active_document_ids": {"$each": document_ids}}

    chat_history_collection.update_one(
        {"session_id": session_id},
        update_doc,
        upsert=True,
    )


def get_active_document_ids(session_id: str) -> List[str]:
    """Return the document IDs that have been active in this session."""
    doc = get_session_doc(session_id)
    return doc.get("active_document_ids", [])


def get_active_collection_id(session_id: str) -> Optional[str]:
    """Return the collection_id scoped to this session, if set."""
    doc = get_session_doc(session_id)
    return doc.get("active_collection_id")


def get_conversation_summary(session_id: str, max_turns: int = 3) -> str:
    """
    Return a concise summary of the last N Q&A turns in the session,
    suitable for injecting into the synthesis prompt as context.

    Format:
      [Q]: <user question>
      [A]: <assistant answer (first 200 chars)>
    """
    history = get_history(session_id)
    if not history:
        return ""

    # Only include user/assistant pairs (skip system summaries)
    qa_pairs: List[tuple] = []
    i = 0
    while i < len(history) - 1:
        if history[i].get("role") == "user" and history[i + 1].get("role") == "assistant":
            qa_pairs.append((history[i]["content"], history[i + 1]["content"]))
            i += 2
        else:
            i += 1

    # Take the last max_turns pairs
    recent = qa_pairs[-max_turns:]
    lines = []
    for q, a in recent:
        lines.append(f"[Q]: {q[:150].strip()}")
        lines.append(f"[A]: {a[:200].strip()}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Memory Trimming with Compression
# ─────────────────────────────────────────────────────────────────────────────

def trim_history(session_id: str, max_turns: int = 10) -> None:
    """
    Trim history. When > max_turns * 2 messages exist, the oldest (max_turns-2)*2
    messages are compressed into a single summary message via the memory summarizer.
    This replaces naive slicing with context-preserving summarization.
    """
    doc = chat_history_collection.find_one({"session_id": session_id})
    if not doc or "messages" not in doc:
        return

    history = doc["messages"]
    threshold = max_turns * 2

    if len(history) <= threshold:
        return

    # Split: compress the old tail, keep the recent head
    compress_count = (max_turns - 2) * 2  # keep last 4 turns verbatim
    old_messages = history[:compress_count]
    recent_messages = history[compress_count:]

    # Attempt LLM compression; fall back to a simple header if it fails
    try:
        from services.memory_summarizer import summarize_history
        summary_text = summarize_history(old_messages)
    except Exception:
        summary_text = f"[Earlier conversation covering {len(old_messages) // 2} turns — context compressed]"

    if not summary_text:
        summary_text = f"[Earlier conversation: {len(old_messages) // 2} turns compressed]"

    summary_msg = {"role": "system", "content": f"[CONVERSATION SUMMARY] {summary_text}"}
    new_history = [summary_msg] + recent_messages

    chat_history_collection.update_one(
        {"session_id": session_id},
        {"$set": {"messages": new_history}},
    )
    logger.debug(f"Session {session_id}: compressed {len(old_messages)} messages into summary")


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up Resolution
# ─────────────────────────────────────────────────────────────────────────────

# Pronouns / references that indicate the question depends on prior context
_REFERENCE_PATTERNS = re.compile(
    r"\b(it|this|that|they|them|those|these|the document|the policy|the above|"
    r"mentioned|aforementioned|previous|last|same|said)\b",
    re.IGNORECASE,
)

# Compiled separately so resolve_followup can use re.search without recompiling
_REF_PAT = re.compile(
    r"\b(it|this|that|they|them|those|these|the document|the policy|the above|"
    r"mentioned|aforementioned|previous|last|same|said)\b",
    re.IGNORECASE,
)


def is_followup_question(question: str) -> bool:
    """Return True if the question contains references that imply prior context."""
    words = question.lower().split()
    if len(words) < 8:
        return True  # Short questions are often follow-ups
    return bool(_REF_PAT.search(question))


def resolve_followup(
    question: str,
    history: List[Dict],
    session_id: Optional[str] = None,
) -> str:
    """
    Rewrite a follow-up question to be self-contained by injecting:
      1. Context from the last assistant turn (answer snippet)
      2. Document scope from the session's active_document_ids

    Uses rule-based resolution (no extra LLM call). The rewritten question
    is passed to the existing query_rewriter for semantic cleaning.
    """
    if not history or not is_followup_question(question):
        return question

    # Find the last assistant message
    last_assistant = next(
        (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
        None,
    )

    parts: List[str] = []

    # 1. Inject document scope from session (if available)
    if session_id:
        active_doc_ids = get_active_document_ids(session_id)
        if active_doc_ids:
            scope_str = ", ".join(active_doc_ids[:4])  # cap at 4 to avoid bloat
            parts.append(f"[Document scope: {scope_str}]")

    # 2. Inject last answer as context prefix
    if last_assistant:
        context_snippet = last_assistant[:300].strip()
        parts.append(f"[Previous answer: {context_snippet}]")

    if parts:
        prefix = " ".join(parts)
        rewritten = f"{prefix} {question}"
        logger.debug(
            "Follow-up resolved with document scope",
            extra={"original": question, "rewritten": rewritten[:120]},
        )
        return rewritten

    return question