"""
Context chunk compressor.
Local deterministic cleanup that deduplicates sentences and lines to optimize context window,
without summarizing or omitting dates, numbers, percentages, conditions, clauses, or citations.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def compress_chunk(chunk_text: str) -> str:
    """
    Compression does NOT summarize. It only cleans up:
    - removes duplicate lines
    - removes duplicate sentences
    - preserves all clauses, conditions, dates, numbers, percentages, citations.
    """
    if not getattr(settings, "COMPRESSION_ENABLED", False):
        return chunk_text

    if not chunk_text or len(chunk_text.strip()) < 80:
        return chunk_text

    # Deduplicate paragraphs/lines
    lines = chunk_text.split("\n")
    seen_lines = set()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        normalized = " ".join(stripped.split())
        if normalized.lower() in seen_lines:
            continue
        seen_lines.add(normalized.lower())
        cleaned_lines.append(stripped)

    merged_text = "\n".join(cleaned_lines)

    # Deduplicate sentences within the text
    sentences = merged_text.split(". ")
    seen_sentences = set()
    cleaned_sentences = []
    for s in sentences:
        s_stripped = s.strip()
        if not s_stripped:
            continue
        normalized_s = " ".join(s_stripped.split())
        if normalized_s.lower() in seen_sentences:
            continue
        seen_sentences.add(normalized_s.lower())
        cleaned_sentences.append(s_stripped)

    return ". ".join(cleaned_sentences)


async def compress_chunk_async(chunk_text: str) -> str:
    """Non-blocking async wrapper for use inside pipelines."""
    return await asyncio.to_thread(compress_chunk, chunk_text)