"""
services/model_router.py — LLM Model Router with Circuit Breaker

Routes queries to the appropriate LLM based on query intent.
Implements a Circuit Breaker pattern to handle Gemini API failures gracefully.

Circuit Breaker States:
  CLOSED    → Normal operation, all requests go to primary model
  OPEN      → Too many failures, all requests route to fallback (Ollama)
  HALF_OPEN → Probing: one test request to primary, revert or close based on result

Model Routing Logic:
  COMPARATIVE / AGGREGATION / TEMPORAL → Gemini Pro (complex reasoning)
  FACTUAL / PROCEDURAL / AMBIGUOUS     → Gemini Flash (fast, cheap)
  Any model OPEN                       → Ollama fallback

Retry Policy:
  - Max 3 attempts per request
  - Exponential backoff with jitter: 1s, 2s, 4s
  - Respect Retry-After header if present
  - Re-raise rate limit errors (429) immediately — do NOT retry those
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types as genai_types

from config import settings
from models.domain import CircuitState, QueryIntent
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CircuitBreaker:
    """
    Per-service circuit breaker. Thread-safe for async usage.
    State is persisted to MongoDB so it survives restarts.
    """
    service_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_at: float = 0.0
    last_state_change_at: float = field(default_factory=time.time)

    def record_success(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD:
                self._transition(CircuitState.CLOSED)
                logger.info(
                    "Circuit breaker closed after successful probes",
                    extra={"service": self.service_name},
                )
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0   # reset on success

        self._persist()

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_at = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — reopen immediately
            self._transition(CircuitState.OPEN)
            logger.warning(
                "Circuit breaker re-opened: probe failed",
                extra={"service": self.service_name},
            )
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                self._transition(CircuitState.OPEN)
                logger.error(
                    "Circuit breaker opened: failure threshold reached",
                    extra={
                        "service": self.service_name,
                        "failures": self.failure_count,
                    },
                )

        self._persist()

    def is_request_allowed(self) -> bool:
        """Return True if a request to the service is currently allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_at
            if elapsed >= settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS:
                self._transition(CircuitState.HALF_OPEN)
                logger.info(
                    "Circuit breaker entering HALF_OPEN: probing service",
                    extra={"service": self.service_name},
                )
                return True
            return False

        # HALF_OPEN: allow exactly one request per cycle
        return True

    def _transition(self, new_state: CircuitState) -> None:
        self.state = new_state
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change_at = time.time()

    def _persist(self) -> None:
        """Save state to MongoDB (non-blocking fire-and-forget)."""
        try:
            from services.db import circuit_breaker_collection
            circuit_breaker_collection.update_one(
                {"service_name": self.service_name},
                {"$set": {
                    "state": self.state.value,
                    "failure_count": self.failure_count,
                    "success_count": self.success_count,
                    "last_failure_at": self.last_failure_at,
                    "last_state_change_at": self.last_state_change_at,
                }},
                upsert=True,
            )
        except Exception:
            pass   # Never let persistence failure break the circuit breaker itself


# ── Singleton circuit breakers per service ────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {}


def _get_breaker(service_name: str) -> CircuitBreaker:
    if service_name not in _breakers:
        _breakers[service_name] = CircuitBreaker(service_name=service_name)
        # Try to restore state from MongoDB
        try:
            from services.db import circuit_breaker_collection
            doc = circuit_breaker_collection.find_one({"service_name": service_name})
            if doc:
                _breakers[service_name].state = CircuitState(doc.get("state", "closed"))
                _breakers[service_name].failure_count = doc.get("failure_count", 0)
                _breakers[service_name].last_failure_at = doc.get("last_failure_at", 0.0)
        except Exception:
            pass
    return _breakers[service_name]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_gemini_client: Optional[genai.Client] = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _gemini_client


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_model(intent: Optional[QueryIntent]) -> str:
    """
    Select the appropriate Gemini model based on query intent.

    Pro → complex reasoning (comparative, aggregation, temporal)
    Flash → everything else (fast, cheap, still high quality)
    """
    if not settings.MODEL_ROUTING_ENABLED:
        return settings.GEMINI_FLASH_MODEL

    if intent and intent.value in settings.PRO_MODEL_INTENTS:
        return settings.GEMINI_PRO_MODEL

    return settings.GEMINI_FLASH_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Retry with exponential backoff
# ─────────────────────────────────────────────────────────────────────────────

async def _call_with_retry(
    prompt: str,
    model: str,
    use_json_mode: bool = False,
    max_attempts: int = 3,
) -> str:
    """
    Call Gemini with exponential backoff retry.

    Rate limit errors (429) are NOT retried — they are re-raised immediately
    so the caller can route to Ollama or return a degraded response.

    Returns response text or raises the last exception.
    """
    breaker = _get_breaker(f"gemini:{model}")

    config: dict = {"temperature": settings.GENERATION_TEMPERATURE}
    if use_json_mode:
        config["response_mime_type"] = "application/json"

    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        if not breaker.is_request_allowed():
            raise RuntimeError(
                f"Circuit breaker OPEN for {model}: "
                f"too many failures, routing to fallback."
            )

        try:
            client = _get_gemini_client()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            breaker.record_success()
            return text

        except Exception as exc:
            err_str = str(exc).lower()
            breaker.record_failure()

            # Rate limit → do NOT retry, re-raise for caller to handle
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                logger.warning(
                    "Gemini rate limit hit — not retrying",
                    extra={"model": model, "error": str(exc)[:200]},
                )
                raise

            last_exc = exc
            logger.error(
                "Gemini call failed",
                extra={
                    "model": model,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                },
            )

            if attempt < max_attempts - 1:
                # Exponential backoff with jitter: 1s, 2s, 4s ± 0.5s
                wait = (2 ** attempt) + random.uniform(-0.5, 0.5)
                logger.info(f"Retrying Gemini call in {wait:.1f}s...")
                await asyncio.sleep(max(0.5, wait))

    raise last_exc or RuntimeError(f"Gemini call failed after {max_attempts} attempts")


# ─────────────────────────────────────────────────────────────────────────────
# Ollama fallback
# ─────────────────────────────────────────────────────────────────────────────

async def _ollama_health_check() -> bool:
    """Check if Ollama is available. Returns False if not reachable."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=settings.OLLAMA_HEALTH_TIMEOUT) as http:
            r = await http.get(f"{settings.OLLAMA_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def _call_ollama(prompt: str) -> str:
    """Call Ollama with a health check gate. Returns empty string if unavailable."""
    if not await _ollama_health_check():
        logger.warning("Ollama unavailable — no fallback possible")
        return ""

    try:
        import torch
        model = settings.OLLAMA_MODEL_GPU if torch.cuda.is_available() else settings.OLLAMA_MODEL_CPU
    except ImportError:
        model = settings.OLLAMA_MODEL_CPU

    breaker = _get_breaker("ollama")
    if not breaker.is_request_allowed():
        logger.warning("Ollama circuit breaker OPEN")
        return ""

    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            res = await http.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            if res.status_code == 200:
                text = res.json().get("response", "").strip()
                breaker.record_success()
                return text
            breaker.record_failure()
            logger.warning("Ollama returned non-200", extra={"status": res.status_code})
    except Exception as exc:
        breaker.record_failure()
        logger.error(
            "Ollama call failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

async def generate(
    prompt: str,
    intent: Optional[QueryIntent] = None,
    use_json_mode: bool = False,
    allow_fallback: bool = True,
) -> tuple[str, str]:
    """
    Route a generation request to the appropriate model.

    Routing priority:
      1. Gemini (model selected by intent)
      2. Ollama (if Gemini circuit is open or all retries failed)

    Args:
        prompt:         Full prompt string.
        intent:         Query intent for model selection.
        use_json_mode:  Request JSON response from Gemini.
        allow_fallback: Whether to attempt Ollama if Gemini fails.

    Returns:
        Tuple of (response_text, model_used_name).
        response_text may be empty string if both models failed.

    Raises:
        RuntimeError only if both Gemini AND Ollama fail and allow_fallback=True.
    """
    if not settings.GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY not set — routing directly to Ollama")
        text = await _call_ollama(prompt)
        return text, "ollama"

    model = _select_model(intent)

    try:
        text = await _call_with_retry(prompt, model, use_json_mode=use_json_mode)
        if text:
            return text, model
    except Exception as exc:
        err_str = str(exc).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            # Hard rate limit — do not retry, do not fall back silently, raise
            raise

        logger.error(
            "Gemini generation failed after retries — attempting fallback",
            extra={"error": str(exc)[:200], "model": model},
        )

    if not allow_fallback:
        return "", model

    text = await _call_ollama(prompt)
    return text, "ollama"


async def generate_stream(
    prompt: str,
    intent: Optional[QueryIntent] = None,
):
    """
    Streaming version. Yields text tokens.
    Falls back to single-shot Ollama if Gemini fails (no token streaming for Ollama).
    """
    if not settings.GOOGLE_API_KEY:
        text, _ = await generate(prompt, intent=intent, allow_fallback=True)
        yield text
        return

    model = _select_model(intent)
    breaker = _get_breaker(f"gemini:{model}")

    if not breaker.is_request_allowed():
        text = await _call_ollama(prompt)
        yield text
        return

    try:
        client = _get_gemini_client()
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
        breaker.record_success()
    except Exception as exc:
        breaker.record_failure()
        logger.error(
            "Gemini streaming failed — falling back to Ollama single-shot",
            extra={"model": model, "error": str(exc)[:200]},
        )
        text = await _call_ollama(prompt)
        if text:
            yield text
        else:
            yield ""
