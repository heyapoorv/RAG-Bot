"""
Centralized application configuration using pydantic-settings.
All settings are loaded from environment variables with strong typing.
"""
from functools import lru_cache
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ── App ──────────────────────────────────────────────────────────
    APP_NAME: str = "DocIntel AI"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ── Security ─────────────────────────────────────────────────────
    JWT_SECRET: str = Field(default="change-me-in-production-32-chars-long", min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 7
    ADMIN_KEY: str = "supersecret"

    # ── MongoDB ──────────────────────────────────────────────────────
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "rag_system"

    # ── Redis ────────────────────────────────────────────────────────
    REDIS_URL: Optional[str] = None           # None → fallback to MongoDB cache
    REDIS_CACHE_TTL_SECONDS: int = 3600       # 1 hour default

    # ── Pinecone ─────────────────────────────────────────────────────
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "newrag"
    PINECONE_ENV: str = "us-east-1"

    # ── Google / Gemini ──────────────────────────────────────────────
    GOOGLE_API_KEY: str = ""
    GEMINI_GENERATION_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBED_DIM: int = 768
    GENERATION_TEMPERATURE: float = 0.1

    # ── Retrieval ────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 5
    BM25_WEIGHT: float = 0.3
    VECTOR_WEIGHT: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.20
    RRF_K: int = 60

    # ── Cache ────────────────────────────────────────────────────────
    CACHE_ENABLED: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.94
    CACHE_MAX_SIZE: int = 1000

    # ── Reranker ─────────────────────────────────────────────────────
    RERANKER_MODE: Literal["local", "cross_encoder", "hybrid"] = "local"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_USE_GPU: bool = False

    # ── Chunking ─────────────────────────────────────────────────────
    CHUNK_PARENT_SIZE: int = 700
    CHUNK_CHILD_SIZE: int = 180
    CHUNK_OVERLAP_SENTENCES: int = 2

    # ── Verification ─────────────────────────────────────────────────
    VERIFICATION_ENABLED: bool = True
    VERIFICATION_MODEL: str = "gemini-2.5-flash"

    # ── CORS ─────────────────────────────────────────────────────────
    FRONTEND_URL: str = "*"

    # ── Rate Limiting ─────────────────────────────────────────────────
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_QUERY: str = "30/minute"
    RATE_LIMIT_HEALTH: str = "60/minute"

    # ── Monitoring ───────────────────────────────────────────────────
    PROMETHEUS_ENABLED: bool = True
    OTEL_ENABLED: bool = False
    OTEL_ENDPOINT: Optional[str] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()


# Module-level shortcut used by non-DI code paths
settings = get_settings()
