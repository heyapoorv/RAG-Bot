"""
Query routes: single query, batch, streaming, session history.
All endpoints require authentication. Namespace = document scope.
Users may only query within their own namespace (admins may access any).
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from services.auth_service import get_current_user, validate_namespace_access
from services.answer import answer_questions, stream_generate_answer
from services.db import chat_history_collection
from models.schemas import QueryRequest, QueryResponse
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Batch / Single Query ──────────────────────────────────────────────────────

@router.post("/", response_model=QueryResponse)
async def query_questions(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
):
    """
    Execute one or more RAG questions and return grounded answers.
    Namespace is scoped per user — users may only query their own documents.
    """
    if not request.namespace:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'namespace' is required. Upload a document first.",
        )

    # Enforce tenant isolation before any retrieval
    validate_namespace_access(request.namespace, user, require_write=False)

    try:
        results = await answer_questions(
            questions=request.questions,
            session_id=request.session_id,
            namespace=request.namespace,
            collection_ids=request.collection_ids,
            document_ids=request.document_ids,
        )
        return results
    except Exception as exc:
        logger.error("Query processing failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal server error.")


# ── Streaming Query ───────────────────────────────────────────────────────────

@router.post("/stream")
async def stream_query(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
):
    """
    Stream answer tokens for the first question in the request.
    Returns text/plain StreamingResponse.
    """
    if not request.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No questions provided.",
        )
    if not request.namespace:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="'namespace' is required.",
        )

    # Enforce tenant isolation before streaming
    validate_namespace_access(request.namespace, user, require_write=False)

    question = request.questions[0]

    async def token_generator():
        async for token in stream_generate_answer(
            question=question,
            session_id=request.session_id,
            namespace=request.namespace,
            collection_ids=request.collection_ids,
            document_ids=request.document_ids,
        ):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


# ── Session Management ────────────────────────────────────────────────────────

@router.get("/sessions")
def get_sessions(
    namespace: str = Query(..., description="Document namespace / user scope"),
    user: dict = Depends(get_current_user),
):
    """List all chat sessions for a namespace."""
    # Enforce namespace ownership — users can only list their own sessions
    validate_namespace_access(namespace, user, require_write=False)

    try:
        sessions = (
            chat_history_collection.find({"namespace": namespace})
            .sort("_id", -1)
            .limit(100)
        )
        result = []
        for s in sessions:
            msgs = s.get("messages", [])
            preview = msgs[0]["content"][:50] + "..." if msgs else "Empty session"
            result.append(
                {
                    "session_id": s["session_id"],
                    "preview": preview,
                    "message_count": len(msgs),
                }
            )
        return result
    except Exception as exc:
        logger.error("Session list failed", extra={"error": str(exc)})
        return []


@router.get("/history/{session_id}")
def get_session_history(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Return full message history for a session."""
    doc = chat_history_collection.find_one({"session_id": session_id})
    if doc:
        # Enforce ownership: the session's namespace must be accessible to the user
        session_namespace = doc.get("namespace", "")
        if session_namespace:
            validate_namespace_access(session_namespace, user, require_write=False)
    return {"messages": doc.get("messages", []) if doc else []}


@router.delete("/history/{session_id}")
def clear_session_history(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Delete a session's chat history."""
    doc = chat_history_collection.find_one({"session_id": session_id}, {"namespace": 1})
    if doc:
        session_namespace = doc.get("namespace", "")
        if session_namespace:
            validate_namespace_access(session_namespace, user, require_write=True)
    chat_history_collection.delete_one({"session_id": session_id})
    return {"message": f"Session '{session_id}' cleared."}