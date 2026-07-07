"""
services/health_checker.py — Background Service Health Monitor

Probes all critical infrastructure services on a configurable interval
and stores results in MongoDB service_health collection.

Services monitored:
  - MongoDB (self)        → ping command
  - Pinecone              → dummy describe_index_stats call
  - Redis                 → ping command
  - Gemini API            → lightweight models.list call
  - ClamAV               → ping command to clamd daemon
  - Ollama               → GET /api/tags endpoint

Results exposed at GET /health/detailed endpoint.

Health states:
  "ok"       → service is reachable and responding normally
  "degraded" → service responds but with errors or high latency (> 3s)
  "down"     → service is unreachable
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

HealthStatus = Literal["ok", "degraded", "down"]


@dataclass
class ServiceStatus:
    service: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    detail: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Individual probes ─────────────────────────────────────────────────────────

async def _check_mongodb() -> ServiceStatus:
    t0 = time.time()
    try:
        from services.db import _client
        _client.admin.command("ping")
        return ServiceStatus("mongodb", "ok", round((time.time() - t0) * 1000))
    except Exception as exc:
        return ServiceStatus("mongodb", "down", detail=str(exc)[:200])


async def _check_pinecone() -> ServiceStatus:
    t0 = time.time()
    try:
        from config import settings
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        idx = pc.Index(settings.PINECONE_INDEX_NAME)
        idx.describe_index_stats()
        ms = round((time.time() - t0) * 1000)
        status = "degraded" if ms > 3000 else "ok"
        return ServiceStatus("pinecone", status, ms)
    except Exception as exc:
        return ServiceStatus("pinecone", "down", detail=str(exc)[:200])


async def _check_redis() -> ServiceStatus:
    t0 = time.time()
    try:
        from config import settings
        if not settings.REDIS_URL:
            return ServiceStatus("redis", "ok", detail="Not configured (optional)")
        import redis as redis_lib
        rc = redis_lib.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        rc.ping()
        return ServiceStatus("redis", "ok", round((time.time() - t0) * 1000))
    except Exception as exc:
        return ServiceStatus("redis", "down", detail=str(exc)[:200])


async def _check_gemini() -> ServiceStatus:
    t0 = time.time()
    try:
        from config import settings
        if not settings.GOOGLE_API_KEY:
            return ServiceStatus("gemini", "down", detail="GOOGLE_API_KEY not set")
        from google import genai
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        # List models — lightweight, no generation cost
        list(client.models.list())
        ms = round((time.time() - t0) * 1000)
        status = "degraded" if ms > 5000 else "ok"
        return ServiceStatus("gemini", status, ms)
    except Exception as exc:
        err = str(exc)
        if "429" in err or "quota" in err.lower():
            return ServiceStatus("gemini", "degraded", detail="Rate limited (429)")
        return ServiceStatus("gemini", "down", detail=err[:200])


async def _check_clamav() -> ServiceStatus:
    t0 = time.time()
    try:
        from config import settings
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect((settings.CLAMAV_HOST, settings.CLAMAV_PORT))
        sock.sendall(b"zPING\0")
        response = sock.recv(32).decode("utf-8", errors="replace")
        sock.close()
        if "PONG" in response:
            return ServiceStatus("clamav", "ok", round((time.time() - t0) * 1000))
        return ServiceStatus("clamav", "degraded", detail=f"Unexpected response: {response}")
    except Exception as exc:
        return ServiceStatus("clamav", "down", detail=str(exc)[:200])


async def _check_ollama() -> ServiceStatus:
    t0 = time.time()
    try:
        from config import settings
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as http:
            r = await http.get(f"{settings.OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                return ServiceStatus("ollama", "ok", round((time.time() - t0) * 1000))
            return ServiceStatus("ollama", "degraded", detail=f"HTTP {r.status_code}")
    except Exception as exc:
        return ServiceStatus("ollama", "down", detail=str(exc)[:200])


# ── Aggregate health check ────────────────────────────────────────────────────

async def check_all_services() -> Dict[str, ServiceStatus]:
    """
    Run all service health checks concurrently.
    Returns a dict of service_name → ServiceStatus.
    Total wall time is bounded by the slowest individual probe.
    """
    results = await asyncio.gather(
        _check_mongodb(),
        _check_pinecone(),
        _check_redis(),
        _check_gemini(),
        _check_clamav(),
        _check_ollama(),
        return_exceptions=True,
    )

    services = ["mongodb", "pinecone", "redis", "gemini", "clamav", "ollama"]
    status_map: Dict[str, ServiceStatus] = {}

    for service, result in zip(services, results):
        if isinstance(result, Exception):
            status_map[service] = ServiceStatus(service, "down", detail=str(result)[:200])
        else:
            status_map[service] = result

    # Persist to MongoDB for dashboard access
    _persist_health(status_map)

    return status_map


def _persist_health(status_map: Dict[str, ServiceStatus]) -> None:
    """Write health check results to MongoDB."""
    try:
        from services.db import service_health_collection
        docs = []
        for service, s in status_map.items():
            docs.append({
                "service": service,
                "status": s.status,
                "latency_ms": s.latency_ms,
                "detail": s.detail,
                "checked_at": s.checked_at,
            })
        if docs:
            service_health_collection.insert_many(docs)
    except Exception as exc:
        logger.warning("Failed to persist health results", extra={"error": str(exc)})


def get_overall_status(status_map: Dict[str, ServiceStatus]) -> HealthStatus:
    """
    Derive overall system health from individual service statuses.

    Critical services: mongodb, pinecone, gemini
    Non-critical (degraded tolerated): redis, ollama, clamav
    """
    critical = ["mongodb", "pinecone", "gemini"]
    non_critical = ["redis", "ollama", "clamav"]

    for svc in critical:
        s = status_map.get(svc)
        if s and s.status == "down":
            return "down"

    has_degraded = any(
        s and s.status in ("down", "degraded")
        for svc, s in status_map.items()
        if svc in critical + non_critical
    )

    return "degraded" if has_degraded else "ok"
