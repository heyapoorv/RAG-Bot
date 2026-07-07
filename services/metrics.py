"""
Prometheus metrics for DocIntel AI.

Tracked metrics:
  - request latency (histogram by endpoint)
  - RAG query latency breakdown (retrieval / reranking / generation)
  - cache hit/miss counter
  - reranker accuracy approximation
  - ingestion throughput
  - verification pass/fail rate
  - active sessions gauge
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)


def _make_metrics():
    if not PROMETHEUS_AVAILABLE:
        return None

    class _Metrics:
        # ── HTTP ──────────────────────────────────────────────────────────
        http_requests_total = Counter(
            "dociintel_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )
        http_request_duration_seconds = Histogram(
            "docinteel_http_request_duration_seconds",
            "HTTP request duration",
            ["endpoint"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # ── RAG pipeline ──────────────────────────────────────────────────
        rag_query_total = Counter(
            "docinteel_rag_query_total",
            "Total RAG queries",
            ["namespace", "verified", "cache_hit"],
        )
        rag_latency_ms = Histogram(
            "docinteel_rag_latency_ms",
            "End-to-end RAG latency (ms)",
            buckets=[100, 250, 500, 1000, 2500, 5000, 10000],
        )
        retrieval_latency_ms = Histogram(
            "docinteel_retrieval_latency_ms",
            "Retrieval stage latency (ms)",
            buckets=[50, 100, 250, 500, 1000, 2500],
        )
        reranker_latency_ms = Histogram(
            "docinteel_reranker_latency_ms",
            "Reranker stage latency (ms)",
            buckets=[10, 25, 50, 100, 250, 500],
        )
        generation_latency_ms = Histogram(
            "docinteel_generation_latency_ms",
            "LLM generation latency (ms)",
            buckets=[100, 250, 500, 1000, 2500, 5000],
        )

        # ── Cache ─────────────────────────────────────────────────────────
        cache_hits_total = Counter(
            "docinteel_cache_hits_total",
            "Total cache hits",
            ["namespace"],
        )
        cache_misses_total = Counter(
            "docinteel_cache_misses_total",
            "Total cache misses",
            ["namespace"],
        )

        # ── Verification ──────────────────────────────────────────────────
        verification_total = Counter(
            "docinteel_verification_total",
            "Answer verification results",
            ["result"],  # "verified" | "unsupported"
        )

        # ── Ingestion ─────────────────────────────────────────────────────
        ingestion_total = Counter(
            "docinteel_ingestion_total",
            "Total documents ingested",
            ["file_type"],
        )
        ingestion_chunks_total = Counter(
            "docinteel_ingestion_chunks_total",
            "Total chunks upserted to Pinecone",
        )

        # ── System ────────────────────────────────────────────────────────
        active_sessions = Gauge(
            "docinteel_active_sessions",
            "Current active chat sessions",
        )
        errors_total = Counter(
            "docinteel_errors_total",
            "Total application errors",
            ["component"],
        )

    return _Metrics()


metrics = _make_metrics()


@contextmanager
def track_latency(histogram) -> Generator[None, None, None]:
    """Context manager to record elapsed time into a Histogram (ms)."""
    t0 = time.time()
    try:
        yield
    finally:
        if histogram is not None:
            elapsed = (time.time() - t0) * 1000
            histogram.observe(elapsed)


def get_metrics_response():
    """Return raw Prometheus metrics text for /metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return "# Prometheus not installed\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
