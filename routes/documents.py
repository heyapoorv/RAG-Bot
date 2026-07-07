"""
Document AI Intelligence routes.
All AI analysis features: summarization, clause extraction, risk analysis,
entity extraction, document comparison, advanced RAG.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from services.auth_service import get_current_user, require_role, validate_namespace_access
from services.ai_features import (
    summarize_document,
    extract_clauses,
    analyze_risks,
    extract_entities,
    compare_documents,
    multi_query_retrieve,
    hyde_retrieve,
)
from models.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    ClauseExtractionResponse,
    CompareDocumentsRequest,
    CompareDocumentsResponse,
    EntityExtractionResponse,
    QueryRequest,
)
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Document Listing ─────────────────────────────────────────────────────────

@router.get("/", response_model=List[dict])
def list_documents(
    namespace: str,
    user=Depends(get_current_user),
):
    """List all uploaded documents in the namespace."""
    from services.db import documents_collection
    
    # Enforce namespace isolation using the shared ownership utility
    validate_namespace_access(namespace, user, require_write=False)

    try:
        docs = list(documents_collection.find({"namespace": namespace}, {"_id": 0}))
        return docs
    except Exception as e:
        logger.error("Failed to list documents", extra={"error": str(e)})
        return []


# ── Summarization ─────────────────────────────────────────────────────────────

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    payload: SummarizeRequest,
    user=Depends(get_current_user),
):
    """Generate a concise document summary with key topics."""
    validate_namespace_access(payload.namespace, user, require_write=False)
    try:
        result = await summarize_document(
            document_id=payload.document_id,
            namespace=payload.namespace,
            max_length=payload.max_length,
        )
        return SummarizeResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Summarization failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Summarization failed.")


# ── Clause Extraction ─────────────────────────────────────────────────────────

@router.post("/clauses", response_model=ClauseExtractionResponse)
async def clauses(
    document_id: str,
    namespace: str,
    user=Depends(get_current_user),
):
    """Extract structured clauses from legal/policy documents."""
    validate_namespace_access(namespace, user, require_write=False)
    try:
        result = await extract_clauses(document_id=document_id, namespace=namespace)
        return ClauseExtractionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Clause extraction failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Clause extraction failed.")


# ── Risk Analysis ─────────────────────────────────────────────────────────────

@router.post("/risks")
async def risks(
    document_id: str,
    namespace: str,
    user=Depends(get_current_user),
):
    """Identify and score risks in a document."""
    validate_namespace_access(namespace, user, require_write=False)
    try:
        return await analyze_risks(document_id=document_id, namespace=namespace)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Risk analysis failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Risk analysis failed.")


# ── Entity Extraction ─────────────────────────────────────────────────────────

@router.post("/entities", response_model=EntityExtractionResponse)
async def entities(
    document_id: str,
    namespace: str,
    user=Depends(get_current_user),
):
    """Extract named entities from a document."""
    validate_namespace_access(namespace, user, require_write=False)
    try:
        result = await extract_entities(document_id=document_id, namespace=namespace)
        return EntityExtractionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Entity extraction failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Entity extraction failed.")


# ── Document Comparison ───────────────────────────────────────────────────────

@router.post("/compare", response_model=CompareDocumentsResponse)
async def compare(
    payload: CompareDocumentsRequest,
    user=Depends(get_current_user),
):
    """Compare two uploaded documents for similarities and differences."""
    validate_namespace_access(payload.namespace, user, require_write=False)
    if payload.document_id_a == payload.document_id_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id_a and document_id_b must be different.",
        )
    try:
        result = await compare_documents(
            namespace=payload.namespace,
            document_id_a=payload.document_id_a,
            document_id_b=payload.document_id_b,
        )
        return CompareDocumentsResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Document comparison failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Comparison failed.")


# ── Multi-Query Retrieval ─────────────────────────────────────────────────────

@router.post("/multi-query")
async def multi_query(
    question: str,
    namespace: str,
    top_k: int = 5,
    user=Depends(get_current_user),
):
    """
    Multi-query retrieval: generate multiple query phrasings and merge results.
    Returns enriched context chunks.
    """
    validate_namespace_access(namespace, user, require_write=False)
    try:
        chunks = await multi_query_retrieve(
            question=question,
            namespace=namespace,
            top_k=top_k,
        )
        return {"chunks": chunks, "total": len(chunks)}
    except Exception as e:
        logger.error("Multi-query retrieval failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="Multi-query retrieval failed.")


# ── HyDE Retrieval ────────────────────────────────────────────────────────────

@router.post("/hyde")
async def hyde(
    question: str,
    namespace: str,
    top_k: int = 5,
    user=Depends(get_current_user),
):
    """
    HyDE retrieval: generate a hypothetical answer then use it as the search query.
    """
    validate_namespace_access(namespace, user, require_write=False)
    try:
        chunks = await hyde_retrieve(
            question=question,
            namespace=namespace,
            top_k=top_k,
        )
        return {"chunks": chunks, "total": len(chunks)}
    except Exception as e:
        logger.error("HyDE retrieval failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail="HyDE retrieval failed.")
