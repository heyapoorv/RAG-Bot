"""
Pydantic request/response schemas — single source of truth for API contracts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None


class UserLoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    role: str


class UserProfile(BaseModel):
    username: str
    email: Optional[str]
    role: str
    created_at: Optional[datetime]


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: str
    file: str
    namespace: str
    job_id: str
    message: str


class IngestionJobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    filename: str
    namespace: str
    collection_id: str
    chunks_total: int
    chunks_processed: int
    error: Optional[str]
    created_at: float
    completed_at: Optional[float]


# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = None
    namespace: Optional[str] = None
    collection_ids: Optional[List[str]] = None
    document_ids: Optional[List[str]] = None
    stream: bool = False

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, v: List[str]) -> List[str]:
        if not v or not any(q.strip() for q in v):
            raise ValueError("At least one non-empty question is required.")
        return [q.strip() for q in v if q.strip()]


class CitationSchema(BaseModel):
    source: str
    document_id: Optional[str]
    page: Optional[int]
    section: Optional[str]
    highlight: str
    score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    # Phase 3 — enriched grounding
    chunk_id: Optional[str] = None
    collection_id: Optional[str] = None
    claim_text: Optional[str] = None  # The specific claim this chunk supports


class StructuredSourceSchema(BaseModel):
    """One entry from the ## Sources section of a multi-doc synthesis response."""
    doc: str
    page: Optional[int] = None
    chunk_seq: Optional[int] = None
    description: Optional[str] = None


class GenerationScoresSchema(BaseModel):
    """Heuristic generation quality scores computed inline (no LLM call)."""
    faithfulness: float = Field(default=1.0, ge=0.0, le=1.0,
        description="Fraction of answer sentences grounded in retrieved chunks")
    completeness: float = Field(default=1.0, ge=0.0, le=1.0,
        description="Fraction of expected key topics covered in the answer")
    cross_doc_consistency: float = Field(default=1.0, ge=0.0, le=1.0,
        description="Whether multi-doc facts are internally consistent or conflicts are properly reported")


class AnswerSchema(BaseModel):
    question: str
    rewritten_query: Optional[str] = None
    answer: str
    sources: List[str]
    citations: List[CitationSchema]
    verified: Optional[bool] = None
    cache_hit: bool = False
    latency_ms: Optional[float] = None
    # Phase 3 — multi-doc synthesis fields
    evidence_by_document: Optional[Dict[str, List[str]]] = None  # {doc: [claims]}
    conflicts: Optional[List[str]] = None                        # detected contradictions
    structured_sources: Optional[List[StructuredSourceSchema]] = None  # parsed ## Sources
    document_count: int = 1
    synthesis_mode: str = "single_doc"  # "single_doc" | "multi_doc"
    generation_scores: Optional[GenerationScoresSchema] = None   # heuristic quality scores


class QueryResponse(BaseModel):
    answers: List[AnswerSchema]


# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_queries: int
    avg_latency_ms: float
    cache_hit_rate: float
    verification_rate: float
    failure_rate: float


class RecentQueryRecord(BaseModel):
    question: str
    answer: str
    latency_ms: float
    verified: Optional[bool]
    cache_hit: bool
    namespace: Optional[str]
    timestamp: float


class FailureRecord(BaseModel):
    question: str
    answer: str
    latency_ms: float
    namespace: Optional[str]
    timestamp: float


class TopQuestion(BaseModel):
    question: str
    count: int


# ─────────────────────────────────────────────
# ADMIN / TRACE
# ─────────────────────────────────────────────

class TraceChunk(BaseModel):
    source: str
    chunk_text: str
    score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    page: Optional[int] = None
    section: Optional[str] = None


class TraceResponse(BaseModel):
    id: str
    question: str
    retrieved_chunks: List[TraceChunk]
    reranked_chunks: List[TraceChunk]
    final_context: str
    answer: str
    latency_ms: float
    timestamp: Optional[float]


# ─────────────────────────────────────────────
# SYSTEM CONFIG
# ─────────────────────────────────────────────

class SystemConfigSchema(BaseModel):
    chunk_parent_size: int = Field(default=700, ge=100, le=2000)
    chunk_child_size: int = Field(default=180, ge=50, le=500)
    chunk_overlap_sentences: int = Field(default=2, ge=0, le=10)
    default_top_k: int = Field(default=5, ge=1, le=20)
    cache_threshold: float = Field(default=0.94, ge=0.0, le=1.0)
    cache_enabled: bool = True
    reranker_mode: Literal["local", "cross_encoder", "hybrid"] = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    verification_enabled: bool = True
    generation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generation_model: str = "gemini-2.5-flash"
    confidence_threshold: float = Field(default=0.20, ge=0.0, le=1.0)


class ConfigUpdateRequest(BaseModel):
    config: SystemConfigSchema
    reason: Optional[str] = None  # Audit log reason


class ConfigResponse(BaseModel):
    config: SystemConfigSchema
    updated_at: Optional[datetime]
    updated_by: Optional[str]
    version: int


# ─────────────────────────────────────────────
# DOCUMENT AI
# ─────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    document_id: str
    namespace: str
    max_length: int = Field(default=500, ge=100, le=2000)


class SummarizeResponse(BaseModel):
    document_id: str
    summary: str
    key_topics: List[str]
    word_count: int


class ClauseExtractionResponse(BaseModel):
    document_id: str
    clauses: List[Dict[str, Any]]
    total_clauses: int


class CompareDocumentsRequest(BaseModel):
    namespace: str
    document_id_a: str
    document_id_b: str


class CompareDocumentsResponse(BaseModel):
    document_id_a: str
    document_id_b: str
    similarities: List[str]
    differences: List[str]
    recommendation: str


class EntityExtractionResponse(BaseModel):
    document_id: str
    entities: Dict[str, List[str]]  # e.g. {"PERSON": [...], "ORG": [...]}


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

class ServiceHealth(BaseModel):
    status: Literal["ok", "degraded", "down"]
    latency_ms: Optional[float]
    detail: Optional[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    services: Dict[str, ServiceHealth]
    uptime_seconds: float
