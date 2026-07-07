"""
Dynamic system configuration CRUD.
Admins can update RAG pipeline parameters that apply live without restart.
All changes are versioned and audited.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from services.auth_service import require_role, log_audit_event
from services.db import config_collection
from models.schemas import SystemConfigSchema, ConfigUpdateRequest, ConfigResponse
from config import settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

CONFIG_KEY = "rag_system_config"


def _get_or_default() -> dict:
    """Load current config from DB or return defaults from settings."""
    doc = config_collection.find_one({"key": CONFIG_KEY})
    if doc:
        return doc
    return {
        "key": CONFIG_KEY,
        "config": SystemConfigSchema().model_dump(),
        "version": 0,
        "updated_at": None,
        "updated_by": "system",
    }


def _apply_to_settings(cfg: SystemConfigSchema) -> None:
    """
    Apply config values to the running settings object so they take
    effect without restart (for in-process consumers).
    """
    settings.CHUNK_PARENT_SIZE = cfg.chunk_parent_size
    settings.CHUNK_CHILD_SIZE = cfg.chunk_child_size
    settings.CHUNK_OVERLAP_SENTENCES = cfg.chunk_overlap_sentences
    settings.DEFAULT_TOP_K = cfg.default_top_k
    settings.CACHE_SIMILARITY_THRESHOLD = cfg.cache_threshold
    settings.CACHE_ENABLED = cfg.cache_enabled
    settings.RERANKER_MODE = cfg.reranker_mode
    settings.RERANKER_MODEL = cfg.reranker_model
    settings.VERIFICATION_ENABLED = cfg.verification_enabled
    settings.GENERATION_TEMPERATURE = cfg.generation_temperature
    settings.GEMINI_GENERATION_MODEL = cfg.generation_model
    settings.CONFIDENCE_THRESHOLD = cfg.confidence_threshold


# ── GET current config ────────────────────────────────────────────────────────

@router.get("/", response_model=ConfigResponse)
def get_config(user=Depends(require_role("analyst"))):
    """Retrieve the current system configuration (analyst+ role)."""
    doc = _get_or_default()
    return ConfigResponse(
        config=SystemConfigSchema(**doc["config"]),
        updated_at=doc.get("updated_at"),
        updated_by=doc.get("updated_by"),
        version=doc.get("version", 0),
    )


# ── UPDATE config ─────────────────────────────────────────────────────────────

@router.put("/", response_model=ConfigResponse)
def update_config(
    payload: ConfigUpdateRequest,
    admin=Depends(require_role("admin")),
):
    """
    Update system configuration (admin+ role).
    Changes apply immediately to the running process.
    """
    doc = _get_or_default()
    new_version = doc.get("version", 0) + 1

    now = datetime.now(timezone.utc)
    new_doc = {
        "key": CONFIG_KEY,
        "config": payload.config.model_dump(),
        "version": new_version,
        "updated_at": now,
        "updated_by": admin["username"],
        "reason": payload.reason,
    }

    config_collection.replace_one(
        {"key": CONFIG_KEY},
        new_doc,
        upsert=True,
    )

    # Apply live
    _apply_to_settings(payload.config)

    log_audit_event(
        "config_update",
        admin["username"],
        detail={
            "version": new_version,
            "reason": payload.reason,
            "changes": payload.config.model_dump(),
        },
    )

    logger.info(
        "System config updated",
        extra={"version": new_version, "by": admin["username"]},
    )

    return ConfigResponse(
        config=payload.config,
        updated_at=now,
        updated_by=admin["username"],
        version=new_version,
    )


# ── ROLLBACK ──────────────────────────────────────────────────────────────────

@router.post("/rollback")
def rollback_config(admin=Depends(require_role("admin"))):
    """
    Rollback to default configuration values.
    """
    default_cfg = SystemConfigSchema()
    now = datetime.now(timezone.utc)
    doc = _get_or_default()
    new_version = doc.get("version", 0) + 1

    config_collection.replace_one(
        {"key": CONFIG_KEY},
        {
            "key": CONFIG_KEY,
            "config": default_cfg.model_dump(),
            "version": new_version,
            "updated_at": now,
            "updated_by": admin["username"],
            "reason": "Rollback to defaults",
        },
        upsert=True,
    )

    _apply_to_settings(default_cfg)
    log_audit_event("config_rollback", admin["username"])

    return {"message": "Configuration rolled back to defaults.", "version": new_version}


# ── CONFIG HISTORY ────────────────────────────────────────────────────────────

@router.get("/history")
def config_history(admin=Depends(require_role("admin"))):
    """Return audit trail for config changes."""
    from services.db import audit_collection
    logs = list(
        audit_collection.find(
            {"action": {"$in": ["config_update", "config_rollback"]}},
            {"_id": 0},
        ).sort("timestamp", -1).limit(50)
    )
    return logs
