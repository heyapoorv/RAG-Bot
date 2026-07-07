"""
MongoDB connection — single client singleton with all enterprise collection handles.
All collections are namespaced to support multi-tenant hierarchy:
  Organization → Workspace → Collection → Document
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _create_client() -> MongoClient:
    client = MongoClient(
        settings.MONGODB_URI,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
        maxPoolSize=50,
        retryWrites=True,
    )
    # Verify connection on startup
    client.admin.command("ping")
    logger.info("MongoDB connection established", extra={"uri": settings.MONGODB_URI.split("@")[-1]})
    return client


# ── Singleton client & database ─────────────────────────────────────────────
_client: MongoClient = _create_client()
db: Database = _client[settings.MONGODB_DB_NAME]

# ── Core Auth Collections ─────────────────────────────────────────────────────
users_collection: Collection = db["users"]

# ── Multi-Tenant Hierarchy ─────────────────────────────────────────────────────
organizations_collection: Collection = db["organizations"]
workspaces_collection: Collection = db["workspaces"]
collections_collection: Collection = db["collections"]  # document collections within workspaces

# ── Document Management ────────────────────────────────────────────────────────
documents_collection: Collection = db["documents"]
ingestion_jobs_collection: Collection = db["ingestion_jobs"]
pii_scan_results_collection: Collection = db["pii_scan_results"]

# ── Chat & Session ─────────────────────────────────────────────────────────────
chat_history_collection: Collection = db["chat_history"]
long_term_memory_collection: Collection = db["long_term_memory"]

# ── Cache ─────────────────────────────────────────────────────────────────────
cache_collection: Collection = db["semantic_cache"]

# ── Analytics & Observability ─────────────────────────────────────────────────
analytics_collection: Collection = db["analytics"]
evaluation_runs_collection: Collection = db["evaluation_runs"]
service_health_collection: Collection = db["service_health"]

# ── Configuration & Audit ─────────────────────────────────────────────────────
config_collection: Collection = db["system_config"]
audit_collection: Collection = db["audit_logs"]

# ── Search ────────────────────────────────────────────────────────────────────
bm25_corpus_collection: Collection = db["bm25_corpus"]

# ── Circuit Breaker State ─────────────────────────────────────────────────────
circuit_breaker_collection: Collection = db["circuit_breaker_state"]

# ── Evaluation Judge Cache ────────────────────────────────────────────────────
eval_judge_cache_collection: Collection = db["eval_judge_cache"]


def ensure_indexes() -> None:
    """Create all necessary indexes — safe to call on every startup."""

    # ── Users ─────────────────────────────────────────────────────────────────
    users_collection.create_index([("username", ASCENDING)], unique=True)
    users_collection.create_index([("email", ASCENDING)], sparse=True, unique=True)
    users_collection.create_index([("org_id", ASCENDING)])
    users_collection.create_index([("workspace_ids", ASCENDING)])

    # ── Organizations ─────────────────────────────────────────────────────────
    organizations_collection.create_index([("org_id", ASCENDING)], unique=True)
    organizations_collection.create_index([("name", ASCENDING)])

    # ── Workspaces ────────────────────────────────────────────────────────────
    workspaces_collection.create_index([("workspace_id", ASCENDING)], unique=True)
    workspaces_collection.create_index([("org_id", ASCENDING)])
    workspaces_collection.create_index([("org_id", ASCENDING), ("name", ASCENDING)])

    # ── Collections (document groups) ─────────────────────────────────────────
    collections_collection.create_index([("collection_id", ASCENDING)], unique=True)
    collections_collection.create_index([("workspace_id", ASCENDING)])
    collections_collection.create_index([("workspace_id", ASCENDING), ("name", ASCENDING)])

    # ── Documents ─────────────────────────────────────────────────────────────
    documents_collection.create_index([("document_id", ASCENDING)], unique=True)
    documents_collection.create_index([("workspace_id", ASCENDING)])
    documents_collection.create_index([("collection_id", ASCENDING)])
    documents_collection.create_index([("sha256_fingerprint", ASCENDING)])  # dedup
    documents_collection.create_index(
        [("org_id", ASCENDING), ("workspace_id", ASCENDING), ("collection_id", ASCENDING)]
    )
    documents_collection.create_index([("ingestion_status", ASCENDING)])

    # ── Ingestion Jobs ────────────────────────────────────────────────────────
    ingestion_jobs_collection.create_index([("job_id", ASCENDING)], unique=True)
    ingestion_jobs_collection.create_index([("workspace_id", ASCENDING)])
    ingestion_jobs_collection.create_index([("status", ASCENDING)])
    ingestion_jobs_collection.create_index([("created_at", DESCENDING)])
    # Self-healer: find stuck jobs quickly
    ingestion_jobs_collection.create_index(
        [("status", ASCENDING), ("created_at", ASCENDING)]
    )

    # ── PII Scan Results ──────────────────────────────────────────────────────
    pii_scan_results_collection.create_index([("document_id", ASCENDING)])
    pii_scan_results_collection.create_index([("workspace_id", ASCENDING)])
    pii_scan_results_collection.create_index([("scanned_at", DESCENDING)])

    # ── Chat History ─────────────────────────────────────────────────────────
    chat_history_collection.create_index([("session_id", ASCENDING)], unique=True)
    chat_history_collection.create_index([("workspace_id", ASCENDING)])
    chat_history_collection.create_index([("user_id", ASCENDING)])
    chat_history_collection.create_index([("active_document_ids", ASCENDING)])
    chat_history_collection.create_index([("active_collection_id", ASCENDING)])
    chat_history_collection.create_index([("last_active_at", DESCENDING)])

    # ── Long-Term Memory ──────────────────────────────────────────────────────
    long_term_memory_collection.create_index([("user_id", ASCENDING)])
    long_term_memory_collection.create_index([("workspace_id", ASCENDING)])
    long_term_memory_collection.create_index(
        [("user_id", ASCENDING), ("workspace_id", ASCENDING)]
    )

    # ── Semantic Cache ────────────────────────────────────────────────────────
    cache_collection.create_index([("workspace_id", ASCENDING)])
    cache_collection.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
    cache_collection.create_index(
        [("workspace_id", ASCENDING), ("created_at", DESCENDING)]
    )

    # ── Analytics ────────────────────────────────────────────────────────────
    analytics_collection.create_index([("timestamp", DESCENDING)])
    analytics_collection.create_index([("workspace_id", ASCENDING), ("timestamp", DESCENDING)])
    analytics_collection.create_index([("org_id", ASCENDING), ("timestamp", DESCENDING)])
    analytics_collection.create_index([("verified", ASCENDING)])
    analytics_collection.create_index([("cache_hit", ASCENDING)])

    # ── Evaluation Runs ───────────────────────────────────────────────────────
    evaluation_runs_collection.create_index([("run_id", ASCENDING)], unique=True)
    evaluation_runs_collection.create_index([("created_at", DESCENDING)])

    # ── Service Health ────────────────────────────────────────────────────────
    service_health_collection.create_index([("checked_at", DESCENDING)])
    service_health_collection.create_index([("service", ASCENDING), ("checked_at", DESCENDING)])

    # ── Config ───────────────────────────────────────────────────────────────
    config_collection.create_index([("key", ASCENDING)], unique=True)

    # ── Audit Logs ────────────────────────────────────────────────────────────
    audit_collection.create_index([("timestamp", DESCENDING)])
    audit_collection.create_index([("user", ASCENDING), ("timestamp", DESCENDING)])
    audit_collection.create_index([("org_id", ASCENDING), ("timestamp", DESCENDING)])
    audit_collection.create_index([("action", ASCENDING)])

    # ── BM25 Corpus ───────────────────────────────────────────────────────────
    bm25_corpus_collection.create_index(
        [("workspace_id", ASCENDING), ("term", ASCENDING)], unique=True
    )
    bm25_corpus_collection.create_index([("workspace_id", ASCENDING)])

    # ── Circuit Breaker ───────────────────────────────────────────────────────
    circuit_breaker_collection.create_index([("service_name", ASCENDING)], unique=True)

    # ── Eval Judge Cache ──────────────────────────────────────────────────────
    eval_judge_cache_collection.create_index([("question_hash", ASCENDING)], unique=True)
    eval_judge_cache_collection.create_index(
        [("created_at", ASCENDING)], expireAfterSeconds=86400 * 7  # 7-day TTL
    )

    logger.info("MongoDB indexes ensured")