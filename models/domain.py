"""
models/domain.py — Enterprise domain model hierarchy.

Org → Workspace → Collection → Document → Chunk

These are the internal domain models used for business logic.
API request/response schemas live in models/schemas.py.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    SUPER_ADMIN = "super_admin"
    ORG_ADMIN   = "org_admin"
    WORKSPACE_ADMIN = "workspace_admin"
    ANALYST     = "analyst"
    USER        = "user"

    @property
    def level(self) -> int:
        return {
            "super_admin":      100,
            "org_admin":         80,
            "workspace_admin":   60,
            "analyst":           40,
            "user":              10,
        }.get(self.value, 0)

    def can_access_org(self, other_role: "UserRole") -> bool:
        return self.level >= other_role.level


class DocumentClass(str, Enum):
    """Document classification used to select chunking strategy."""
    LEGAL       = "legal"
    CONTRACT    = "contract"
    POLICY      = "policy"
    INSURANCE   = "insurance"
    MEDICAL     = "medical"
    FINANCE     = "finance"
    INVOICE     = "invoice"
    HR          = "hr"
    RESEARCH    = "research"
    MANUAL      = "manual"
    SOP         = "sop"
    EMAIL       = "email"
    REPORT      = "report"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    GENERAL     = "general"


class IngestionStatus(str, Enum):
    PENDING    = "pending"
    SCANNING   = "scanning"        # virus scan in progress
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    BLOCKED    = "blocked"         # blocked due to PII/virus detection


class ChunkingStrategy(str, Enum):
    CLAUSE_AWARE  = "clause_aware"    # Legal, Contracts, Policy
    SEMANTIC      = "semantic"         # Research papers
    HIERARCHICAL  = "hierarchical"     # Manuals, SOPs (heading + section)
    SLIDING_WINDOW = "sliding_window"  # General-purpose
    STRUCTURE_PRESERVE = "structure_preserve"  # Emails (header + body)
    ROW_BASED     = "row_based"        # XLSX, CSV (row + column header)
    PARENT_CHILD  = "parent_child"     # Default hybrid


class QueryIntent(str, Enum):
    FACTUAL      = "factual"       # Direct lookup: "What is the deductible?"
    COMPARATIVE  = "comparative"   # Compare: "How does A differ from B?"
    AGGREGATION  = "aggregation"   # Summarize: "List all clauses about..."
    TEMPORAL     = "temporal"      # Timeline: "What happened before 2023?"
    AMBIGUOUS    = "ambiguous"     # Unclear scope — needs clarification
    PROCEDURAL   = "procedural"    # How-to: "How do I file a claim?"


class CircuitState(str, Enum):
    CLOSED    = "closed"      # Normal operation
    OPEN      = "open"        # Failing — requests routed to fallback
    HALF_OPEN = "half_open"   # Probing — one test request allowed


# ─────────────────────────────────────────────────────────────────────────────
# Organization
# ─────────────────────────────────────────────────────────────────────────────

class Organization(BaseModel):
    org_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    plan: str = "enterprise"
    settings: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str  # super_admin username


# ─────────────────────────────────────────────────────────────────────────────
# Workspace
# ─────────────────────────────────────────────────────────────────────────────

class Workspace(BaseModel):
    workspace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    org_id: str
    name: str
    description: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str  # org_admin username


# ─────────────────────────────────────────────────────────────────────────────
# Collection (group of documents within a workspace)
# ─────────────────────────────────────────────────────────────────────────────

class Collection(BaseModel):
    collection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str
    org_id: str
    name: str
    description: Optional[str] = None
    # Access: "workspace" (all users in workspace) or "restricted" (explicit user list)
    access_policy: str = "workspace"
    allowed_user_ids: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str


# ─────────────────────────────────────────────────────────────────────────────
# Document
# ─────────────────────────────────────────────────────────────────────────────

class Document(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    collection_id: str
    workspace_id: str
    org_id: str

    # File identity
    filename: str
    original_filename: str
    file_type: str                          # pdf, docx, txt, etc.
    sha256_fingerprint: str                 # content hash for dedup
    file_size_bytes: int

    # Classification & language
    document_class: DocumentClass = DocumentClass.GENERAL
    document_class_confidence: float = 0.0
    language: str = "en"
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.PARENT_CHILD

    # Ingestion state
    ingestion_status: IngestionStatus = IngestionStatus.PENDING
    job_id: Optional[str] = None
    chunk_count: int = 0
    vector_count: int = 0
    error_message: Optional[str] = None

    # PII & security
    pii_detected: bool = False
    pii_entity_types: List[str] = Field(default_factory=list)
    virus_scan_passed: bool = False
    virus_scan_at: Optional[datetime] = None

    # Metadata
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    has_tables: bool = False
    has_images: bool = False

    # Timestamps
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uploaded_by: str
    completed_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Chunk (in-memory, not persisted — stored in Pinecone)
# ─────────────────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    collection_id: str
    workspace_id: str
    org_id: str

    text: str
    parent_text: Optional[str] = None   # larger parent window for context

    # Structure
    page: Optional[int] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    chunk_index: int = 0
    is_table: bool = False
    is_heading: bool = False

    # Linking for neighbor expansion
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

    # Stats
    word_count: int = 0
    char_count: int = 0
    importance_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Session / Memory
# ─────────────────────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    SHORT_TERM = "short_term"   # Current session messages
    SUMMARY    = "summary"      # Compressed older turns
    LONG_TERM  = "long_term"    # Extracted key facts — never pruned


class LongTermFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    workspace_id: str
    entity: str           # e.g. "deductible", "contract_party"
    value: str            # e.g. "$500", "Acme Corp"
    source_document_id: Optional[str] = None
    confidence: float = 1.0
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_referenced_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion Job
# ─────────────────────────────────────────────────────────────────────────────

class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    workspace_id: str
    org_id: str
    collection_id: str

    status: IngestionStatus = IngestionStatus.PENDING
    stage: str = "queued"   # Current pipeline stage name for debugging

    chunks_total: int = 0
    chunks_processed: int = 0
    vectors_upserted: int = 0

    error: Optional[str] = None
    error_stage: Optional[str] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────────────
# PII Scan Result
# ─────────────────────────────────────────────────────────────────────────────

class PIIScanResult(BaseModel):
    document_id: str
    workspace_id: str
    filename: str
    pii_found: bool
    entity_types: List[str] = Field(default_factory=list)
    # Number of PII instances detected per entity type
    entity_counts: Dict[str, int] = Field(default_factory=dict)
    # Sample snippets (truncated, for audit only)
    samples: List[str] = Field(default_factory=list)
    scanned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
