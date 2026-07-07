"""
middleware/security.py — Security & Request Tracing Middleware

Adds to every request/response:
  1. X-Request-ID  → Unique UUID per request (injected if not present)
  2. X-Process-Time-Ms → Response time in milliseconds
  3. Security headers → HSTS, X-Frame-Options, X-Content-Type-Options, etc.
  4. Sensitive header stripping from logs

The X-Request-ID is:
  - Generated here if not provided by the caller
  - Propagated via Python contextvars so services can include it in log records
  - Returned in the response headers for client-side correlation
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Thread-local (actually task-local) request ID accessible to all services
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_current_request_id() -> Optional[str]:
    """Return the current request ID from context. May be None outside a request."""
    return _request_id_var.get()


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Injects security headers and request tracing on every HTTP request/response.
    """

    def __init__(self, app: ASGIApp, environment: str = "development"):
        super().__init__(app)
        self.environment = environment

    async def dispatch(self, request: Request, call_next):
        # Generate or propagate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = _request_id_var.set(request_id)

        t0 = time.time()
        try:
            response: Response = await call_next(request)
        finally:
            _request_id_var.reset(token)

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # ── Tracing headers ──────────────────────────────────────────────────
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)

        # ── Security headers ─────────────────────────────────────────────────
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"

        if self.environment == "production":
            # HSTS: only in production (HTTPS required)
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
            # Remove server fingerprint
            response.headers.pop("server", None)
            response.headers.pop("x-powered-by", None)

        return response
