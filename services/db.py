"""
MongoDB connection — single client singleton with all collection handles.
"""
from pymongo import MongoClient
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
    )
    # Verify connection on startup
    client.admin.command("ping")
    logger.info("MongoDB connection established", extra={"uri": settings.MONGODB_URI.split("@")[-1]})
    return client


# ── Singleton client & database ─────────────────────────────────────────────
_client: MongoClient = _create_client()
db: Database = _client[settings.MONGODB_DB_NAME]

# ── Collections ──────────────────────────────────────────────────────────────
analytics_collection: Collection = db["analytics"]
users_collection: Collection = db["users"]
documents_collection: Collection = db["documents"]
chat_history_collection: Collection = db["chat_history"]
cache_collection: Collection = db["semantic_cache"]
config_collection: Collection = db["system_config"]
audit_collection: Collection = db["audit_logs"]
roles_collection: Collection = db["roles"]
bm25_corpus_collection: Collection = db["bm25_corpus"]  # Term → doc-frequency stats
ingestion_jobs_collection: Collection = db["ingestion_jobs"]

# ── Indexes (idempotent) ─────────────────────────────────────────────────────

def ensure_indexes() -> None:
    """Create all necessary indexes — safe to call on every startup."""
    # Analytics
    analytics_collection.create_index([("timestamp", -1)])
    analytics_collection.create_index([("namespace", 1), ("timestamp", -1)])
    analytics_collection.create_index([("verified", 1)])
    analytics_collection.create_index([("cache_hit", 1)])

    # Users
    users_collection.create_index([("username", 1)], unique=True)
    users_collection.create_index([("email", 1)], sparse=True, unique=True)

    # Chat history
    chat_history_collection.create_index([("session_id", 1)], unique=True)
    chat_history_collection.create_index([("namespace", 1)])

    # Semantic cache
    cache_collection.create_index([("namespace", 1)])
    cache_collection.create_index([("expires_at", 1)], expireAfterSeconds=0)

    # Config
    config_collection.create_index([("key", 1)], unique=True)

    # Audit logs
    audit_collection.create_index([("timestamp", -1)])
    audit_collection.create_index([("user", 1), ("timestamp", -1)])

    # BM25 corpus stats
    bm25_corpus_collection.create_index([("namespace", 1), ("term", 1)], unique=True)
    bm25_corpus_collection.create_index([("namespace", 1)])

    # Ingestion Jobs
    ingestion_jobs_collection.create_index([("job_id", 1)], unique=True)
    ingestion_jobs_collection.create_index([("namespace", 1)])

    # Phase 3 — document-scoped session index
    chat_history_collection.create_index([("active_document_ids", 1)])
    chat_history_collection.create_index([("active_collection_id", 1)])

    logger.info("MongoDB indexes ensured")