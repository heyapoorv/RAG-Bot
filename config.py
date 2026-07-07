"""
Centralized application configuration using pydantic-settings.
All settings are loaded from environment variables with strong typing.
All new enterprise settings are documented with inline comments.
"""
from functools import lru_cache
from typing import Literal, List, Optional
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
    APP_VERSION: str = "3.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ── Security ─────────────────────────────────────────────────────
    JWT_SECRET: str = Field(default="change-me-in-production-32-chars-long", min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_DAYS: int = 7
    ADMIN_KEY: str = "supersecret"
    # Max login attempts before account lockout (per 15-minute window)
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

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
    # Fast, cheap — for factual queries and evaluation judging
    GEMINI_FLASH_MODEL: str = "gemini-2.5-flash"
    # Powerful — for complex reasoning, multi-doc synthesis, comparative analysis
    GEMINI_PRO_MODEL: str = "gemini-2.5-pro"
    # Keep backward compat alias
    GEMINI_GENERATION_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    GEMINI_EMBED_DIM: int = 768
    GENERATION_TEMPERATURE: float = 0.1

    # ── Model Routing ─────────────────────────────────────────────────
    MODEL_ROUTING_ENABLED: bool = True
    # Intents that require Pro model — Flash handles all others
    PRO_MODEL_INTENTS: List[str] = ["COMPARATIVE", "AGGREGATION", "TEMPORAL"]

    # ── Circuit Breaker ───────────────────────────────────────────────
    # Number of consecutive failures before opening the circuit
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    # Seconds to wait in OPEN state before attempting HALF_OPEN probe
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = 60
    # Successes in HALF_OPEN required to CLOSE the circuit
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = 2

    # ── Ollama Fallback ───────────────────────────────────────────────
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_CPU: str = "llama3.1:8b"
    OLLAMA_MODEL_GPU: str = "qwen2.5:14b"
    # Seconds to wait for Ollama health check before marking unavailable
    OLLAMA_HEALTH_TIMEOUT: float = 3.0

    # ── Retrieval ────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 5
    BM25_WEIGHT: float = 0.3
    VECTOR_WEIGHT: float = 0.7
    CONFIDENCE_THRESHOLD: float = 0.20
    RRF_K: int = 60

    # ── Cache ────────────────────────────────────────────────────────
    CACHE_ENABLED: bool = True
    CACHE_SIMILARITY_THRESHOLD: float = 0.94
    # L1 in-memory LRU size per process
    CACHE_L1_MAX_SIZE: int = 100
    # L1 TTL in seconds
    CACHE_L1_TTL_SECONDS: int = 300
    # Max MongoDB cache entries per namespace before oldest is evicted
    CACHE_L3_MAX_ENTRIES_PER_NS: int = 500

    # ── Reranker ─────────────────────────────────────────────────────
    RERANKER_MODE: Literal["local", "cross_encoder", "hybrid"] = "local"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_USE_GPU: bool = False

    # ── Chunking ─────────────────────────────────────────────────────
    CHUNK_PARENT_SIZE: int = 700
    CHUNK_CHILD_SIZE: int = 180
    CHUNK_OVERLAP_SENTENCES: int = 2

    # ── Document Ingestion ────────────────────────────────────────────
    # Maximum file size allowed for upload (bytes)
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt", "eml", "pptx", "xlsx", "csv", "md", "html"]

    # ── Virus Scanning (ClamAV) ───────────────────────────────────────
    CLAMAV_HOST: str = "127.0.0.1"
    CLAMAV_PORT: int = 3310
    # Timeout for ClamAV daemon connection (seconds)
    CLAMAV_TIMEOUT: float = 30.0

    # ── PII Detection ─────────────────────────────────────────────────
    PII_DETECTION_ENABLED: bool = True
    # Presidio entity types to detect. Empty list = detect all.
    PII_ENTITY_TYPES: List[str] = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
        "US_SSN", "US_PASSPORT", "IBAN_CODE", "IP_ADDRESS", "LOCATION",
        "DATE_TIME", "NRP",
    ]
    # Minimum confidence score for a PII detection to be flagged (0.0–1.0)
    PII_CONFIDENCE_THRESHOLD: float = 0.7

    # ── Document Classification ───────────────────────────────────────
    # Whether to use LLM for classification (True) or rule-based only (False)
    CLASSIFIER_USE_LLM: bool = False
    CLASSIFIER_LLM_CONFIDENCE_THRESHOLD: float = 0.6

    # ── Prompt Injection ─────────────────────────────────────────────
    PROMPT_INJECTION_ENABLED: bool = True

    # ── Verification ─────────────────────────────────────────────────
    VERIFICATION_ENABLED: bool = True
    VERIFICATION_MODEL: str = "gemini-2.5-flash"

    # ── CORS ─────────────────────────────────────────────────────────
    FRONTEND_URL: str = "*"

    # ── Rate Limiting ─────────────────────────────────────────────────
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_QUERY: str = "30/minute"
    RATE_LIMIT_HEALTH: str = "60/minute"
    RATE_LIMIT_AUTH: str = "20/minute"

    # ── Monitoring ───────────────────────────────────────────────────
    PROMETHEUS_ENABLED: bool = True
    OTEL_ENABLED: bool = False
    OTEL_ENDPOINT: Optional[str] = None

    # ── Self-Healing ─────────────────────────────────────────────────
    # How often the self-healer background job runs (seconds)
    SELF_HEALER_INTERVAL_SECONDS: int = 900   # 15 minutes
    # Jobs stuck in "processing" for longer than this are marked failed
    STUCK_JOB_THRESHOLD_SECONDS: int = 1800  # 30 minutes

    # ── Memory ───────────────────────────────────────────────────────
    # Max short-term message turns before compression kicks in
    MEMORY_MAX_SHORT_TERM_TURNS: int = 20
    # Sessions inactive beyond this many days are expired
    MEMORY_SESSION_EXPIRE_DAYS: int = 30

    # ── Evaluation CI ─────────────────────────────────────────────────
    # Cache evaluation judge results to avoid redundant Gemini calls
    EVAL_JUDGE_CACHE_ENABLED: bool = True
    # Number of questions to run in --quick mode
    EVAL_QUICK_MODE_COUNT: int = 10
    # Maximum regression tolerance before CI fails (fraction, e.g. 0.05 = 5%)
    EVAL_REGRESSION_TOLERANCE: float = 0.05


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()


# Module-level shortcut used by non-DI code paths
settings = get_settings()
