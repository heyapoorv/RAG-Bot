"""
Text chunking utilities for RAG document preprocessing.
Includes:
- basic sentence-based chunking
- clause-aware overlapping sentence chunking (for legal & insurance documents)
- parent-child hierarchical chunking (optimized context indexing)
"""
from __future__ import annotations

import re
from typing import List, Dict, Any

SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')


def chunk_text(
    text: str,
    max_tokens: int = 180,
) -> List[str]:
    """
    Basic sentence-based chunking without overlap.
    """
    sentences = SENTENCE_ENDINGS.split(text)

    chunks = []
    current = []
    tokens = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        t = len(s.split())

        if current and tokens + t > max_tokens:
            chunks.append(" ".join(current))
            current = [s]
            tokens = t
        else:
            current.append(s)
            tokens += t

    if current:
        chunks.append(" ".join(current))

    return chunks


def clause_aware_chunk(
    text: str,
    max_tokens: int = 180,
    overlap_sentences: int = 2,
) -> List[str]:
    """
    Clause-aware overlapping chunking.
    Keeps context sliding window over sentence transitions.
    """
    sentences = SENTENCE_ENDINGS.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []
    tokens = 0

    for sentence in sentences:
        t = len(sentence.split())

        if current and tokens + t > max_tokens:
            chunks.append(" ".join(current))

            overlap = (
                current[-overlap_sentences:]
                if len(current) >= overlap_sentences
                else current
            )

            current = overlap + [sentence]
            tokens = sum(len(x.split()) for x in current)
        else:
            current.append(sentence)
            tokens += t

    if current:
        chunks.append(" ".join(current))

    return chunks


def parent_child_chunk(
    text: str,
    parent_size: int = 700,
    child_size: int = 180,
) -> List[Dict[str, Any]]:
    """
    Creates hierarchical chunks:
    - Parent chunks for answer generation context.
    - Child chunks for dense semantic vector retrieval.
    """
    parent_chunks = clause_aware_chunk(
        text,
        max_tokens=parent_size,
        overlap_sentences=3,
    )

    structured_chunks = []

    for parent_idx, parent in enumerate(parent_chunks):
        child_chunks = clause_aware_chunk(
            parent,
            max_tokens=child_size,
            overlap_sentences=2,
        )

        for child_idx, child in enumerate(child_chunks):
            structured_chunks.append({
                "parent_id": f"parent_{parent_idx}",
                "parent_text": parent,
                "child_text": child,
            })

    return structured_chunks