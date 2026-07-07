"""
services/audit.py — Structured Audit Logging Service

All security-relevant events are written to the append-only 'audit_logs' collection.
This is the single source of truth for:
  - Authentication events (login, logout, failed attempts)
  - Authorization events (access denied, role changes)
  - Data access events (document upload, query, delete)
  - Configuration changes (system config updates)
  - Ingestion events (document blocked by PII/virus)
  - Admin actions

The audit collection is write-only from the application.
No audit records are ever updated or deleted by the application.
Deletion requires out-of-band database access (super_admin + DBA).

Schema per record:
  action:     str   — event type (login_success, document_blocked_pii, etc.)
  user:       str   — username (or "system" for automated events)
  org_id:     str   — organization ID
  workspace_id: str — workspace ID (if applicable)
  target:     str   — object of the action (filename, document_id, etc.)
  detail:     dict  — structured context (no PII values, only types/counts)
  success:    bool  — True if action succeeded
  ip_address: str   — request IP (if available)
  request_id: str   — X-Request-ID header value
  timestamp:  float — epoch seconds (UTC)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def log_audit_event(
    action: str,
    user: str,
    target: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    success: bool = True,
    org_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    """
    Write an immutable audit record to MongoDB.

    This function NEVER raises — if the write fails, it logs the error
    but does NOT affect the calling request.

    Args:
        action:       Event type identifier (snake_case).
        user:         Username or "system".
        target:       Primary object of the action (filename, URL path, etc.).
        detail:       Structured context dict. Must NOT contain raw PII values.
        success:      Whether the action succeeded.
        org_id:       Organization scope.
        workspace_id: Workspace scope.
        ip_address:   Client IP address.
        request_id:   Request trace ID.
    """
    record = {
        "action":       action,
        "user":         user,
        "org_id":       org_id,
        "workspace_id": workspace_id,
        "target":       target,
        "detail":       detail or {},
        "success":      success,
        "ip_address":   ip_address,
        "request_id":   request_id,
        "timestamp":    datetime.now(timezone.utc).timestamp(),
    }

    try:
        from services.db import audit_collection
        audit_collection.insert_one(record)
    except Exception as exc:
        logger.error(
            "Audit log write failed",
            extra={"action": action, "user": user, "error": str(exc)},
        )


# ── Convenience helpers for common audit events ───────────────────────────────

def audit_login_success(username: str, ip: Optional[str] = None, request_id: Optional[str] = None) -> None:
    log_audit_event("login_success", username, ip_address=ip, request_id=request_id)


def audit_login_failure(username: str, ip: Optional[str] = None, request_id: Optional[str] = None) -> None:
    log_audit_event("login_failure", username, success=False, ip_address=ip, request_id=request_id)


def audit_access_denied(
    username: str, path: str, reason: str,
    org_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    log_audit_event(
        "access_denied", username, target=path,
        detail={"reason": reason},
        success=False, org_id=org_id, request_id=request_id,
    )


def audit_document_uploaded(
    username: str, filename: str,
    workspace_id: Optional[str] = None,
    org_id: Optional[str] = None,
    job_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    log_audit_event(
        "document_uploaded", username, target=filename,
        detail={"job_id": job_id},
        workspace_id=workspace_id, org_id=org_id, request_id=request_id,
    )


def audit_document_blocked_pii(
    username: str, filename: str,
    entity_types: list,
    workspace_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> None:
    log_audit_event(
        "document_blocked_pii", username, target=filename,
        detail={"entity_types": entity_types},
        success=False, workspace_id=workspace_id, org_id=org_id,
    )


def audit_document_blocked_virus(
    username: str, filename: str,
    threat_name: str,
    workspace_id: Optional[str] = None,
    org_id: Optional[str] = None,
) -> None:
    log_audit_event(
        "document_blocked_virus", username, target=filename,
        detail={"threat_name": threat_name},
        success=False, workspace_id=workspace_id, org_id=org_id,
    )


def audit_config_changed(
    username: str, changed_keys: list,
    reason: Optional[str] = None,
    request_id: Optional[str] = None,
) -> None:
    log_audit_event(
        "config_changed", username,
        detail={"changed_keys": changed_keys, "reason": reason},
        request_id=request_id,
    )


def audit_query(
    username: str,
    workspace_id: Optional[str] = None,
    org_id: Optional[str] = None,
    session_id: Optional[str] = None,
    intent: Optional[str] = None,
    cache_hit: bool = False,
    request_id: Optional[str] = None,
) -> None:
    log_audit_event(
        "query", username,
        detail={"session_id": session_id, "intent": intent, "cache_hit": cache_hit},
        workspace_id=workspace_id, org_id=org_id, request_id=request_id,
    )
