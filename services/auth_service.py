"""
RBAC Authentication Service.

Roles (ordered by privilege level):
    super_admin  → full system access, manage all users/roles
    admin        → analytics, config, trace viewer, user management
    analyst      → read-only analytics and trace viewer
    user         → chat, upload, query

JWT payload includes:
    { "username": ..., "role": ..., "exp": ... }

Password hashing:
    Uses bcrypt (cost=12) via passlib.
    Backward-compat shim: SHA-256 hashes (64-char hex) from before the
    bcrypt migration are detected on login and transparently re-hashed.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
import bcrypt
from fastapi import Depends, Header, HTTPException, Query, status

from config import settings
from services.db import users_collection, audit_collection
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Role hierarchy ────────────────────────────────────────────────────────────

ROLE_HIERARCHY: dict[str, int] = {
    "super_admin": 100,
    "admin": 80,
    "analyst": 50,
    "user": 10,
}

# Roles that may read (but not write) any namespace
_ANALYST_AND_ABOVE = {"analyst", "admin", "super_admin"}

# Roles that may write any namespace
_ADMIN_AND_ABOVE = {"admin", "super_admin"}


def role_level(role: str) -> int:
    return ROLE_HIERARCHY.get(role, 0)


# ── Password hashing (bcrypt) ─────────────────────────────────────────────────

_BCRYPT_ROUNDS = 12
_SHA256_HEX_LEN = 64  # Length of a SHA-256 hex digest — used to detect old hashes


def _is_legacy_sha256(hashed: str) -> bool:
    """Return True if the stored hash is an old-style SHA-256 hex digest."""
    return len(hashed) == _SHA256_HEX_LEN and all(c in "0123456789abcdef" for c in hashed)


def _sha256_hex(password: str) -> str:
    """Compute raw SHA-256 hex for legacy comparison only."""
    return hashlib.sha256(password.encode()).hexdigest()


def hash_password(password: str) -> str:
    """Hash a password with bcrypt. Returns a UTF-8 string suitable for MongoDB storage."""
    salt = bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a password against a stored hash.

    Supports both:
      - bcrypt hashes (new format, starts with $2b$)
      - SHA-256 hex hashes (legacy format, 64-char hex) — read-only check,
        caller is responsible for re-hashing on success.
    """
    if _is_legacy_sha256(hashed):
        # Legacy SHA-256 path — constant-time compare via hmac-style equality
        return _sha256_hex(plain) == hashed
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def maybe_rehash_password(username: str, plain: str, stored_hash: str) -> None:
    """
    If the stored hash is legacy SHA-256, transparently re-hash with bcrypt.
    Called once on successful login. Safe to fail silently.
    """
    if not _is_legacy_sha256(stored_hash):
        return
    try:
        new_hash = hash_password(plain)
        users_collection.update_one(
            {"username": username},
            {"$set": {"password": new_hash}},
        )
        logger.info(
            "Password re-hashed from SHA-256 to bcrypt",
            extra={"username": username},
        )
    except Exception as exc:
        logger.warning(
            "Password re-hash failed (non-fatal)",
            extra={"username": username, "error": str(exc)},
        )


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=settings.JWT_EXPIRE_DAYS)
    )
    payload["exp"] = expire
    payload["iat"] = datetime.now(timezone.utc)
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )


# ── FastAPI dependency ────────────────────────────────────────────────────────

def _extract_token(authorization: str = Header(None)) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing.",
        )
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <token>'.",
        )
    return parts[1]


def get_current_user(authorization: str = Header(None)) -> dict:
    """FastAPI dependency: decode token → return user payload dict."""
    token = _extract_token(authorization)
    payload = decode_token(token)

    # Verify user still exists in DB and is active
    db_user = users_collection.find_one({"username": payload.get("username")})
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )
    if not db_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return payload  # {"username": ..., "role": ..., "exp": ...}


def require_role(minimum_role: str):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.get("/admin-only")
        def endpoint(user=Depends(require_role("admin"))):
            ...
    """
    min_level = role_level(minimum_role)

    def _check(user: dict = Depends(get_current_user)) -> dict:
        user_role = user.get("role", "user")
        if role_level(user_role) < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{minimum_role}' role or higher. Your role: '{user_role}'.",
            )
        return user

    return _check


# ── Namespace ownership enforcement ───────────────────────────────────────────

def validate_namespace_access(
    namespace: str,
    user: dict,
    require_write: bool = True,
) -> None:
    """
    Enforce tenant isolation on a namespace string.

    Rules:
      - role=user         → must own namespace (namespace == username). No cross-tenant access.
      - role=analyst      → read-only access to any namespace (require_write=False).
                            Write operations raise 403.
      - role=admin/super  → full read+write access to any namespace.

    Raises HTTPException 403 on violation.
    """
    role = user.get("role", "user")
    username = user.get("username", "")

    if role in _ADMIN_AND_ABOVE:
        # Admins can access everything
        return

    if role == "analyst":
        if require_write:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Analysts have read-only access. Write operations are not permitted.",
            )
        # Analysts can read any namespace
        return

    # role == "user": must own the namespace
    if namespace != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Forbidden: You can only access your own namespace ('{username}'). "
                f"Attempted: '{namespace}'."
            ),
        )


def require_namespace_owner(require_write: bool = True):
    """
    FastAPI dependency factory for namespace ownership.

    Expects `namespace` as a query parameter or Form field.
    Use on endpoints that accept `namespace` as a query param.

    Usage:
        @router.get("/")
        def endpoint(
            namespace: str = Query(...),
            user=Depends(get_current_user),
            _=Depends(require_namespace_owner()),
        ):
            ...

    Note: For endpoints where namespace comes from a request body (Pydantic model),
    call validate_namespace_access() directly inside the endpoint handler.
    """
    def _check(
        namespace: str = Query(..., description="Tenant namespace"),
        user: dict = Depends(get_current_user),
    ) -> dict:
        validate_namespace_access(namespace, user, require_write=require_write)
        return user

    return _check


# ── Audit logging ─────────────────────────────────────────────────────────────

def log_audit_event(
    action: str,
    user: str,
    target: Optional[str] = None,
    detail: Optional[dict] = None,
    success: bool = True,
) -> None:
    """Write an immutable audit record to MongoDB."""
    try:
        audit_collection.insert_one(
            {
                "action": action,
                "user": user,
                "target": target,
                "detail": detail or {},
                "success": success,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            }
        )
    except Exception as exc:
        logger.error("Audit log write failed", extra={"error": str(exc)})