"""
Document AI Intelligence features:
  - Summarization with key topic extraction
  - Clause extraction (legal / policy documents)
  - Risk analysis
  - Key entity extraction
  - Document comparison (diff two uploaded documents)
  - Duplicate detection
  - Multi-query retrieval
  - HyDE (Hypothetical Document Embeddings) retrieval
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

from google import genai

from config import settings
from services.embedding import embed_texts
from services.retrieval import retrieve_contexts
from services.vectordb import index as pinecone_index
from utils.logger import get_logger

logger = get_logger(__name__)

client = genai.Client(api_key=settings.GOOGLE_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _gemini(prompt: str) -> str:
    """Simple async Gemini wrapper with error handling."""
    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=settings.GEMINI_GENERATION_MODEL,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as exc:
        logger.error("Gemini call failed in ai_features", extra={"error": str(exc)})
        raise


def _get_document_chunks(document_id: str, namespace: str, max_chunks: int = 30) -> List[str]:
    """
    Fetch all stored chunks for a document from Pinecone by document_id.
    """
    try:
        # Query Pinecone with a broad filter
        dummy_emb = embed_texts(["document content overview"])[0]
        result = pinecone_index.query(
            vector=dummy_emb,
            top_k=max_chunks,
            include_metadata=True,
            namespace=namespace,
            filter={"document_id": {"$eq": document_id}},
        )
        matches = result.get("matches", []) or []
        return [m["metadata"].get("chunk_text", "") for m in matches if m.get("metadata")]
    except Exception as exc:
        logger.error("Chunk fetch failed", extra={"error": str(exc), "document_id": document_id})
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 1. Document Summarization
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_document(
    document_id: str,
    namespace: str,
    max_length: int = 500,
) -> Dict[str, Any]:
    """
    Generate a concise summary + key topics from document chunks.
    """
    chunks = _get_document_chunks(document_id, namespace)
    if not chunks:
        raise ValueError(f"No chunks found for document '{document_id}' in namespace '{namespace}'.")

    combined = "\n\n".join(chunks[:20])  # First 20 chunks for context
    word_count = len(combined.split())

    prompt = f"""
You are an expert document analyst. Analyze the following document content and produce:
1. A concise summary (max {max_length} words)
2. A JSON list of 5-8 key topics

DOCUMENT:
{combined}

Respond EXACTLY in this JSON format:
{{
    "summary": "...",
    "key_topics": ["topic1", "topic2", ...]
}}
"""
    raw = await _gemini(prompt)

    try:
        # Extract JSON from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = {"summary": raw, "key_topics": []}
    except (json.JSONDecodeError, Exception):
        data = {"summary": raw, "key_topics": []}

    return {
        "document_id": document_id,
        "summary": data.get("summary", raw),
        "key_topics": data.get("key_topics", []),
        "word_count": word_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clause Extraction
# ─────────────────────────────────────────────────────────────────────────────

async def extract_clauses(
    document_id: str,
    namespace: str,
) -> Dict[str, Any]:
    """
    Extract structured clauses from legal/policy documents.
    """
    chunks = _get_document_chunks(document_id, namespace)
    if not chunks:
        raise ValueError(f"No chunks found for document '{document_id}'.")

    combined = "\n\n".join(chunks[:25])

    prompt = f"""
You are a legal document analyst specializing in clause extraction.

Extract all distinct clauses from the document below. For each clause provide:
- title: Short clause name
- type: e.g. "Liability", "Termination", "Payment", "Confidentiality", etc.
- text: Full clause text
- risk_level: "low", "medium", or "high"

DOCUMENT:
{combined}

Return a valid JSON array of clauses:
[
  {{
    "title": "...",
    "type": "...",
    "text": "...",
    "risk_level": "low|medium|high"
  }}
]
"""
    raw = await _gemini(prompt)

    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        clauses = json.loads(match.group()) if match else []
    except (json.JSONDecodeError, Exception):
        clauses = []

    return {
        "document_id": document_id,
        "clauses": clauses,
        "total_clauses": len(clauses),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Risk Analysis
# ─────────────────────────────────────────────────────────────────────────────

async def analyze_risks(
    document_id: str,
    namespace: str,
) -> Dict[str, Any]:
    """
    Identify and score risks in a document.
    """
    chunks = _get_document_chunks(document_id, namespace)
    if not chunks:
        raise ValueError(f"No chunks found for document '{document_id}'.")

    combined = "\n\n".join(chunks[:25])

    prompt = f"""
You are a risk analyst. Identify all risks in the document below.

For each risk provide:
- category: Risk category (e.g. "Financial", "Legal", "Operational", "Compliance")
- description: What the risk is
- severity: "critical", "high", "medium", or "low"
- mitigation: Recommended mitigation strategy

DOCUMENT:
{combined}

Return valid JSON array:
[
  {{
    "category": "...",
    "description": "...",
    "severity": "...",
    "mitigation": "..."
  }}
]
"""
    raw = await _gemini(prompt)

    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        risks = json.loads(match.group()) if match else []
    except (json.JSONDecodeError, Exception):
        risks = []

    risk_score = sum(
        {"critical": 10, "high": 7, "medium": 4, "low": 1}.get(r.get("severity", "low"), 1)
        for r in risks
    )

    return {
        "document_id": document_id,
        "risks": risks,
        "total_risks": len(risks),
        "overall_risk_score": risk_score,
        "risk_level": (
            "critical" if risk_score > 50
            else "high" if risk_score > 25
            else "medium" if risk_score > 10
            else "low"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entity Extraction
# ─────────────────────────────────────────────────────────────────────────────

async def extract_entities(
    document_id: str,
    namespace: str,
) -> Dict[str, Any]:
    """
    Extract named entities (people, organizations, dates, amounts, locations).
    """
    chunks = _get_document_chunks(document_id, namespace)
    if not chunks:
        raise ValueError(f"No chunks found for document '{document_id}'.")

    combined = "\n\n".join(chunks[:20])

    prompt = f"""
Extract all named entities from the document below.

Categories to extract:
- PERSON: Individual people
- ORG: Companies, institutions
- DATE: Specific dates or time periods
- AMOUNT: Monetary values, percentages
- LOCATION: Countries, cities, addresses
- LEGAL: Case numbers, regulation references
- PRODUCT: Products, services, software

DOCUMENT:
{combined}

Return valid JSON:
{{
  "PERSON": [],
  "ORG": [],
  "DATE": [],
  "AMOUNT": [],
  "LOCATION": [],
  "LEGAL": [],
  "PRODUCT": []
}}
"""
    raw = await _gemini(prompt)

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        entities = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, Exception):
        entities = {}

    return {
        "document_id": document_id,
        "entities": entities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Document Comparison
# ─────────────────────────────────────────────────────────────────────────────

async def compare_documents(
    namespace: str,
    document_id_a: str,
    document_id_b: str,
) -> Dict[str, Any]:
    """
    Compare two uploaded documents. Identifies similarities, differences,
    and provides a recommendation.
    """
    chunks_a = _get_document_chunks(document_id_a, namespace, max_chunks=15)
    chunks_b = _get_document_chunks(document_id_b, namespace, max_chunks=15)

    if not chunks_a:
        raise ValueError(f"No chunks found for document '{document_id_a}'.")
    if not chunks_b:
        raise ValueError(f"No chunks found for document '{document_id_b}'.")

    text_a = "\n".join(chunks_a[:10])
    text_b = "\n".join(chunks_b[:10])

    prompt = f"""
You are an expert document comparison analyst.

Compare DOCUMENT A and DOCUMENT B below. Identify:
1. Key similarities (shared topics, terms, clauses)
2. Key differences (conflicting terms, missing sections, different values)
3. A professional recommendation

DOCUMENT A (ID: {document_id_a}):
{text_a}

DOCUMENT B (ID: {document_id_b}):
{text_b}

Return valid JSON:
{{
  "similarities": ["...", "..."],
  "differences": ["...", "..."],
  "recommendation": "..."
}}
"""
    raw = await _gemini(prompt)

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(match.group()) if match else {}
    except (json.JSONDecodeError, Exception):
        result = {"similarities": [], "differences": [], "recommendation": raw}

    return {
        "document_id_a": document_id_a,
        "document_id_b": document_id_b,
        **result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Advanced RAG: Multi-Query Retrieval
# ─────────────────────────────────────────────────────────────────────────────

async def multi_query_retrieve(
    question: str,
    namespace: str,
    top_k: int = 5,
    num_queries: int = 3,
) -> List[Dict]:
    """
    Generate multiple phrasings of the query, retrieve for each,
    then merge and deduplicate results.
    """
    prompt = f"""
Generate {num_queries} different ways to ask the following question.
Each phrasing should target different aspects or vocabulary.
Return ONLY a JSON array of strings.

QUESTION: {question}

["phrasing1", "phrasing2", "phrasing3"]
"""
    raw = await _gemini(prompt)

    try:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        phrasings = json.loads(match.group()) if match else [question]
    except (json.JSONDecodeError, Exception):
        phrasings = [question]

    # Ensure original question is included
    if question not in phrasings:
        phrasings.insert(0, question)

    # Retrieve for all phrasings concurrently
    tasks = [
        asyncio.to_thread(retrieve_contexts, q, top_k * 2, namespace)
        for q in phrasings[:num_queries]
    ]
    all_results = await asyncio.gather(*tasks)

    # Merge and deduplicate
    seen_texts: set = set()
    merged: List[Dict] = []
    for result_list in all_results:
        for chunk in result_list:
            text = chunk.get("chunk_text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                merged.append(chunk)

    return merged[:top_k * 2]


# ─────────────────────────────────────────────────────────────────────────────
# 7. HyDE Retrieval (Hypothetical Document Embeddings)
# ─────────────────────────────────────────────────────────────────────────────

async def hyde_retrieve(
    question: str,
    namespace: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Generate a hypothetical answer, embed it, and use it to retrieve
    real document chunks. Improves recall for abstract queries.
    """
    prompt = f"""
Write a concise, factual hypothetical answer to the question below.
This will be used as a search query — focus on domain-specific terminology.

QUESTION: {question}

HYPOTHETICAL ANSWER:
"""
    hypothetical_answer = await _gemini(prompt)

    # Retrieve using the hypothetical answer as query
    return await asyncio.to_thread(
        retrieve_contexts,
        hypothetical_answer,
        top_k * 2,
        namespace,
    )
