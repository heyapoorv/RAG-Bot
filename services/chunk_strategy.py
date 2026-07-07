"""
services/chunk_strategy.py — Chunking Strategy Factory

Selects and executes the appropriate chunking strategy based on DocumentClass.

Available strategies:
  CLAUSE_AWARE       → Legal, Contract, Policy, Insurance
  SEMANTIC           → Research papers
  HIERARCHICAL       → Manuals, SOPs, Presentations
  STRUCTURE_PRESERVE → Emails (header + body preserved)
  ROW_BASED          → XLSX, CSV (each row = chunk with column headers)
  SLIDING_WINDOW     → General fallback
  PARENT_CHILD       → Default hybrid (current behavior)

Each strategy produces a list of Chunk objects with full metadata populated.
"""
from __future__ import annotations

import re
import uuid
from typing import List, Optional, Dict

from models.domain import Chunk, ChunkingStrategy
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base chunk factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(
    text: str,
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
    chunk_index: int,
    page: Optional[int] = None,
    section: Optional[str] = None,
    heading: Optional[str] = None,
    parent_text: Optional[str] = None,
    is_table: bool = False,
    is_heading: bool = False,
    prev_id: Optional[str] = None,
    next_id: Optional[str] = None,
) -> Chunk:
    chunk_id = f"{document_id}-{chunk_index}-{uuid.uuid4().hex[:6]}"
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        collection_id=collection_id,
        workspace_id=workspace_id,
        org_id=org_id,
        text=text.strip(),
        parent_text=parent_text,
        page=page,
        section=section,
        heading=heading,
        chunk_index=chunk_index,
        is_table=is_table,
        is_heading=is_heading,
        prev_chunk_id=prev_id,
        next_chunk_id=next_id,
        word_count=len(text.split()),
        char_count=len(text),
    )


def _link_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Set prev_chunk_id and next_chunk_id for navigational expansion."""
    for i, chunk in enumerate(chunks):
        chunk.prev_chunk_id = chunks[i - 1].chunk_id if i > 0 else "START"
        chunk.next_chunk_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else "END"
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: PARENT_CHILD (default hybrid)
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter using punctuation and newlines."""
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [s.strip() for s in sentences if s.strip()]


def _parent_child_chunks(
    raw_chunks: List[Dict],
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
) -> List[Chunk]:
    """
    Classic parent-child chunking.
    raw_chunks: list of {text, page, section, parent_text, ...} dicts from parser.
    """
    chunks = []
    for i, raw in enumerate(raw_chunks):
        text = raw.get("text", "").strip()
        if not text or len(text) < 20:
            continue
        c = _make_chunk(
            text=text,
            document_id=document_id,
            workspace_id=workspace_id,
            collection_id=collection_id,
            org_id=org_id,
            chunk_index=i,
            page=raw.get("page"),
            section=raw.get("section"),
            parent_text=raw.get("parent_text"),
            importance_score=raw.get("importance_score", 0.0),
        )
        chunks.append(c)
    return _link_chunks(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: CLAUSE_AWARE (Legal, Contract, Policy)
# ─────────────────────────────────────────────────────────────────────────────

# Clause boundary markers for legal/contract docs
_CLAUSE_PATTERNS = re.compile(
    r"(?:^|\n)(?="
    r"(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT|ADDENDUM|APPENDIX|WHEREAS|NOW,\s*THEREFORE|§)\s*[\dIVXA-Z\.]+"
    r"|(?:\d+\.\s+[A-Z])"
    r")",
    re.MULTILINE | re.IGNORECASE,
)


def _clause_aware_chunks(
    text: str,
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
    page_map: Optional[Dict[int, int]] = None,  # char_offset → page_number
) -> List[Chunk]:
    """
    Split legal/contract text at clause boundaries.
    Very long clauses are further split into child chunks with the clause header as parent.
    """
    max_child = settings.CHUNK_CHILD_SIZE * 5  # ~900 chars for legal content
    splits = _CLAUSE_PATTERNS.split(text)
    chunks = []
    idx = 0

    for split in splits:
        split = split.strip()
        if not split or len(split) < 30:
            continue

        # Extract clause heading (first line)
        lines = split.splitlines()
        heading = lines[0].strip() if lines else None
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else split

        if len(split) <= max_child:
            # Clause fits in one chunk
            c = _make_chunk(
                text=split,
                document_id=document_id,
                workspace_id=workspace_id,
                collection_id=collection_id,
                org_id=org_id,
                chunk_index=idx,
                heading=heading,
                section=heading,
            )
            chunks.append(c)
            idx += 1
        else:
            # Split long clause into child chunks
            words = body.split()
            window = max_child // 5  # approximate word count
            for start in range(0, len(words), window):
                segment = " ".join(words[start: start + window])
                c = _make_chunk(
                    text=f"{heading}\n{segment}" if heading else segment,
                    document_id=document_id,
                    workspace_id=workspace_id,
                    collection_id=collection_id,
                    org_id=org_id,
                    chunk_index=idx,
                    heading=heading,
                    section=heading,
                    parent_text=split[:500],  # full clause as parent
                )
                chunks.append(c)
                idx += 1

    return _link_chunks(chunks) if chunks else []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: HIERARCHICAL (Manuals, SOPs, Presentations)
# ─────────────────────────────────────────────────────────────────────────────

_HEADING_PATTERN = re.compile(
    r"^(#{1,4}\s+.+|[A-Z][A-Z\s\d\-:]{3,60}$|\d+\.\d*\s+[A-Z].{5,60}$)",
    re.MULTILINE,
)


def _hierarchical_chunks(
    raw_chunks: List[Dict],
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
) -> List[Chunk]:
    """
    Group content under its nearest heading.
    Each heading + content block becomes one chunk.
    """
    chunks = []
    current_heading: Optional[str] = None
    current_buffer: List[str] = []
    idx = 0
    current_page: Optional[int] = None

    def flush():
        nonlocal idx
        if not current_buffer:
            return
        text = "\n".join(current_buffer).strip()
        if not text:
            return
        combined = f"{current_heading}\n{text}" if current_heading else text
        c = _make_chunk(
            text=combined,
            document_id=document_id,
            workspace_id=workspace_id,
            collection_id=collection_id,
            org_id=org_id,
            chunk_index=idx,
            heading=current_heading,
            section=current_heading,
            page=current_page,
            is_heading=(current_buffer == [current_heading]),
        )
        chunks.append(c)
        idx += 1
        current_buffer.clear()

    for raw in raw_chunks:
        text = raw.get("text", "").strip()
        page = raw.get("page")
        if not text:
            continue

        if _HEADING_PATTERN.match(text) and len(text) < 200:
            flush()
            current_heading = text
            current_page = page
        else:
            current_buffer.append(text)
            if page is not None:
                current_page = page

    flush()
    return _link_chunks(chunks) if chunks else []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 4: STRUCTURE_PRESERVE (Emails)
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_HEADER_PATTERN = re.compile(
    r"^(From|To|Cc|Bcc|Subject|Date|Sent|Received|Reply-To):\s*.+$",
    re.MULTILINE | re.IGNORECASE,
)


def _structure_preserve_chunks(
    text: str,
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
) -> List[Chunk]:
    """
    Email: extract headers as metadata, split body into paragraphs.
    Each paragraph becomes a chunk with the email headers as parent context.
    """
    headers = "\n".join(_EMAIL_HEADER_PATTERN.findall(text))
    body = _EMAIL_HEADER_PATTERN.sub("", text).strip()

    paragraphs = re.split(r"\n{2,}", body)
    chunks = []
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:
            continue
        c = _make_chunk(
            text=para,
            document_id=document_id,
            workspace_id=workspace_id,
            collection_id=collection_id,
            org_id=org_id,
            chunk_index=idx,
            section="email_body",
            parent_text=f"{headers}\n\n{para[:200]}" if headers else para[:200],
        )
        chunks.append(c)
        idx += 1

    return _link_chunks(chunks) if chunks else []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 5: ROW_BASED (XLSX, CSV)
# ─────────────────────────────────────────────────────────────────────────────

def _row_based_chunks(
    rows: List[Dict],  # list of {header_context, row_text, row_index}
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
) -> List[Chunk]:
    """
    Each row + column headers becomes one chunk.
    Empty rows are skipped. Rows are grouped in windows of 5 for context.
    """
    chunks = []
    window_size = 5

    for start in range(0, len(rows), window_size):
        window = rows[start: start + window_size]
        valid_rows = [r for r in window if r.get("row_text", "").strip()]
        if not valid_rows:
            continue

        header_context = valid_rows[0].get("header_context", "")
        row_texts = "\n".join(r["row_text"] for r in valid_rows)
        text = f"{header_context}\n{row_texts}" if header_context else row_texts

        c = _make_chunk(
            text=text,
            document_id=document_id,
            workspace_id=workspace_id,
            collection_id=collection_id,
            org_id=org_id,
            chunk_index=start // window_size,
            section=f"rows_{start}_to_{start + len(valid_rows) - 1}",
            is_table=True,
        )
        chunks.append(c)

    return _link_chunks(chunks) if chunks else []


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 6: SEMANTIC (Research papers)
# ─────────────────────────────────────────────────────────────────────────────

def _semantic_chunks(
    raw_chunks: List[Dict],
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
) -> List[Chunk]:
    """
    For research papers: use section-aware sliding window.
    Preserve section names as metadata.
    Overlap: re-include last 2 sentences of previous chunk.
    """
    chunks = []
    idx = 0
    prev_tail: str = ""

    for raw in raw_chunks:
        text = raw.get("text", "").strip()
        if not text or len(text) < 30:
            continue

        combined = f"{prev_tail} {text}".strip() if prev_tail else text

        c = _make_chunk(
            text=combined,
            document_id=document_id,
            workspace_id=workspace_id,
            collection_id=collection_id,
            org_id=org_id,
            chunk_index=idx,
            page=raw.get("page"),
            section=raw.get("section"),
            parent_text=raw.get("parent_text"),
        )
        chunks.append(c)
        idx += 1

        # Preserve last 2 sentences as overlap
        sentences = re.split(r'(?<=[.!?])\s+', combined)
        prev_tail = " ".join(sentences[-2:]) if len(sentences) >= 2 else ""

    return _link_chunks(chunks) if chunks else []


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def apply_chunking_strategy(
    strategy: ChunkingStrategy,
    raw_chunks: List[Dict],
    full_text: str,
    document_id: str,
    workspace_id: str,
    collection_id: str,
    org_id: str,
) -> List[Chunk]:
    """
    Apply the appropriate chunking strategy and return Chunk objects.

    Args:
        strategy:       Selected strategy based on DocumentClass.
        raw_chunks:     List of raw dict chunks from the parser.
        full_text:      Full document text (used by some strategies).
        document_id:    For metadata.
        workspace_id:   For metadata.
        collection_id:  For metadata.
        org_id:         For metadata.

    Returns:
        List of Chunk objects ready for embedding.
    """
    kwargs = dict(
        document_id=document_id,
        workspace_id=workspace_id,
        collection_id=collection_id,
        org_id=org_id,
    )

    try:
        if strategy == ChunkingStrategy.CLAUSE_AWARE:
            return _clause_aware_chunks(full_text, **kwargs)

        elif strategy == ChunkingStrategy.HIERARCHICAL:
            return _hierarchical_chunks(raw_chunks, **kwargs)

        elif strategy == ChunkingStrategy.STRUCTURE_PRESERVE:
            return _structure_preserve_chunks(full_text, **kwargs)

        elif strategy == ChunkingStrategy.ROW_BASED:
            # raw_chunks must already contain {header_context, row_text} from parser
            return _row_based_chunks(raw_chunks, **kwargs)

        elif strategy == ChunkingStrategy.SEMANTIC:
            return _semantic_chunks(raw_chunks, **kwargs)

        else:  # PARENT_CHILD, SLIDING_WINDOW, or unknown
            return _parent_child_chunks(raw_chunks, **kwargs)

    except Exception as exc:
        logger.error(
            "Chunking strategy failed — falling back to parent_child",
            extra={
                "strategy": strategy.value,
                "document_id": document_id,
                "error": str(exc),
            },
            exc_info=True,
        )
        return _parent_child_chunks(raw_chunks, **kwargs)
