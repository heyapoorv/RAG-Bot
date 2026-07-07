"""
Main FastAPI application entrypoint — Enterprise v3.
Production-ready: lifespan, CORS, rate limiting, middleware, metrics,
self-healing background scheduler, detailed health endpoint.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import settings
from utils.logger import configure_logging, get_logger
from services.db import ensure_indexes
from services.reranker import preload_reranker
from services.metrics import get_metrics_response, metrics
from middleware.security import SecurityMiddleware

configure_logging(
    level="DEBUG" if settings.DEBUG else "INFO",
    json_output=(settings.ENVIRONMENT == "production"),
)
logger = get_logger(__name__)

# ── Import all routers ─────────────────────────────────────────────────────────
from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.admin import router as admin_router
from routes.analytics import router as analytics_router
from routes.dashboard import router as dashboard_router
from routes.auth import router as auth_router
from routes.config import router as config_router
from routes.documents import router as documents_router

# ── App startup time ─────────────────────────────────────────────────────────
_start_time: float = 0.0
_healer_task: asyncio.Task | None = None


# ── Self-healer background loop ───────────────────────────────────────────────

async def _self_healer_loop():
    """Run the self-healer on a fixed interval for the lifetime of the app."""
    while True:
        await asyncio.sleep(settings.SELF_HEALER_INTERVAL_SECONDS)
        try:
            from services.self_healer import run_self_healer
            await run_self_healer()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Self-healer loop error", extra={"error": str(exc)}, exc_info=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time, _healer_task
    _start_time = time.time()

    logger.info(
        "DocIntel AI starting",
        extra={"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT},
    )

    # ── Startup tasks ─────────────────────────────────────────────────────────
    try:
        ensure_indexes()
    except Exception as exc:
        logger.error("MongoDB index creation failed", extra={"error": str(exc)})

    try:
        preload_reranker()
    except Exception as exc:
        logger.warning("Reranker preload skipped", extra={"error": str(exc)})

    # Start self-healer background task
    _healer_task = asyncio.create_task(_self_healer_loop())
    logger.info("Self-healer background task started")

    yield

    # ── Shutdown tasks ────────────────────────────────────────────────────────
    if _healer_task and not _healer_task.done():
        _healer_task.cancel()
        try:
            await asyncio.wait_for(_healer_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    logger.info("DocIntel AI shut down gracefully.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise AI Document Intelligence Platform — production-grade RAG.",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
allowed_origins = (
    [settings.FRONTEND_URL]
    if settings.FRONTEND_URL != "*"
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Security + Request ID middleware ──────────────────────────────────────────
app.add_middleware(SecurityMiddleware, environment=settings.ENVIRONMENT)


# ── Request timing + metrics middleware ───────────────────────────────────────

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - t0) * 1000, 2)

    if metrics:
        metrics.http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        metrics.http_request_duration_seconds.labels(
            endpoint=request.url.path,
        ).observe(elapsed / 1000)

    return response


# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from middleware.security import get_current_request_id
    request_id = get_current_request_id() or "unknown"

    logger.error(
        "Unhandled exception",
        extra={
            "path": str(request.url),
            "error": str(exc),
            "request_id": request_id,
        },
        exc_info=True,
    )
    if metrics:
        metrics.errors_total.labels(component="global").inc()

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred.",
            "request_id": request_id,
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth_router,      prefix="/auth",      tags=["Auth"])
app.include_router(upload_router,    prefix="/upload",    tags=["Upload"])
app.include_router(query_router,     prefix="/query",     tags=["Query"])
app.include_router(admin_router,     prefix="/admin",     tags=["Admin"])
app.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(config_router,    prefix="/config",    tags=["Config"])
app.include_router(documents_router, prefix="/documents", tags=["Documents"])


# ── Liveness probe ────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    """Basic liveness probe."""
    uptime = round(time.time() - _start_time, 2)
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "uptime_seconds": uptime,
        "environment": settings.ENVIRONMENT,
    }


# ── Detailed health (all service probes) ──────────────────────────────────────

@app.get("/health/detailed", tags=["Health"])
async def health_detailed():
    """
    Detailed readiness probe — checks all infrastructure services.
    Response includes per-service status + latency.
    """
    from services.health_checker import check_all_services, get_overall_status

    status_map = await check_all_services()
    overall = get_overall_status(status_map)

    uptime = round(time.time() - _start_time, 2)
    return {
        "status": overall,
        "version": settings.APP_VERSION,
        "uptime_seconds": uptime,
        "environment": settings.ENVIRONMENT,
        "services": {
            name: {
                "status": s.status,
                "latency_ms": s.latency_ms,
                "detail": s.detail,
            }
            for name, s in status_map.items()
        },
    }


# ── Admin: self-healer trigger ────────────────────────────────────────────────

@app.post("/admin/self-heal", tags=["Admin"], include_in_schema=False)
async def trigger_self_heal(request: Request):
    """Manually trigger a self-healer run (super_admin only)."""
    from services.auth_service import require_role
    from fastapi import Depends
    from services.self_healer import run_self_healer
    return await run_self_healer()


# ── Prometheus metrics endpoint ───────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus-compatible metrics scrape endpoint."""
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)