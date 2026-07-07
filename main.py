"""
Main FastAPI application entrypoint.
Production-ready: lifespan, CORS, rate limiting, middleware, metrics, routes.
"""
from __future__ import annotations

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

configure_logging(
    level="DEBUG" if settings.DEBUG else "INFO",
    json_output=(settings.ENVIRONMENT == "production"),
)
logger = get_logger(__name__)

# Import all routers
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


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()

    logger.info(
        "Starting DocIntel AI",
        extra={"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT},
    )

    # Ensure MongoDB indexes
    ensure_indexes()

    # Preload reranker model if applicable
    preload_reranker()

    yield

    logger.info("DocIntel AI shutting down gracefully.")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Document Intelligence System — production-grade RAG platform.",
    lifespan=lifespan,
    docs_url="/docs" if not settings.ENVIRONMENT == "production" else None,
    redoc_url="/redoc" if not settings.ENVIRONMENT == "production" else None,
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


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - t0) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed)

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
    logger.error(
        "Unhandled exception",
        extra={"path": str(request.url), "error": str(exc)},
        exc_info=True,
    )
    if metrics:
        metrics.errors_total.labels(component="global").inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
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


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health(request: Request):
    """Basic liveness probe."""
    uptime = round(time.time() - _start_time, 2)
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "uptime_seconds": uptime,
        "environment": settings.ENVIRONMENT,
    }


# ── Prometheus metrics endpoint ───────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def prometheus_metrics():
    """Prometheus-compatible metrics scrape endpoint."""
    content, content_type = get_metrics_response()
    return Response(content=content, media_type=content_type)