"""
Auth routes: register, login, refresh, profile, user management.
Full RBAC role assignment on registration/admin actions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from services.auth_service import (
    hash_password,
    verify_password,
    maybe_rehash_password,
    create_token,
    get_current_user,
    require_role,
    log_audit_event,
    decode_token,
)
from services.db import users_collection
from models.schemas import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenResponse,
    UserProfile,
)
from config import settings
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=dict, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterRequest):
    """Register a new user. Default role: 'user'."""
    if users_collection.find_one({"username": payload.username}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists.",
        )

    if payload.email:
        if users_collection.find_one({"email": payload.email}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use.",
            )

    # Only store the email key when a value is provided.
    # A sparse unique index on `email` tolerates multiple documents
    # with the field absent, but rejects duplicate non-null values.
    new_user: dict = {
        "username": payload.username,
        "password": hash_password(payload.password),
        "role": "user",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    if payload.email:
        new_user["email"] = payload.email

    users_collection.insert_one(new_user)

    logger.info("User registered", extra={"username": payload.username})
    log_audit_event("register", payload.username, detail={"email": payload.email})

    return {"message": "Account created successfully."}


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
def login(payload: UserLoginRequest):
    """Authenticate user and return JWT with role claim."""
    db_user = users_collection.find_one({"username": payload.username})

    if not db_user or not verify_password(payload.password, db_user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    if not db_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact an administrator.",
        )

    # Transparently upgrade legacy SHA-256 hash to bcrypt on next login
    maybe_rehash_password(payload.username, payload.password, db_user["password"])

    role = db_user.get("role", "user")
    token = create_token({"username": payload.username, "role": role})

    logger.info("User logged in", extra={"username": payload.username, "role": role})
    log_audit_event("login", payload.username)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRE_DAYS * 86400,
        role=role,
    )


# ── Profile ───────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserProfile)
def me(user: dict = Depends(get_current_user)):
    """Return the current user's profile."""
    db_user = users_collection.find_one({"username": user["username"]})
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserProfile(
        username=db_user["username"],
        email=db_user.get("email"),
        role=db_user.get("role", "user"),
        created_at=db_user.get("created_at"),
    )


# ── Admin: list users ─────────────────────────────────────────────────────────

@router.get("/users", response_model=List[UserProfile])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    admin: dict = Depends(require_role("admin")),
):
    """List all users (admin+ only)."""
    cursor = users_collection.find(
        {},
        {"_id": 0, "password": 0},
    ).skip(skip).limit(limit)

    return [
        UserProfile(
            username=u["username"],
            email=u.get("email"),
            role=u.get("role", "user"),
            created_at=u.get("created_at"),
        )
        for u in cursor
    ]


# ── Admin: set role ───────────────────────────────────────────────────────────

class RoleUpdate(BaseModel):
    role: str = Field(..., pattern="^(super_admin|admin|analyst|user)$")


@router.patch("/users/{username}/role")
def set_user_role(
    username: str,
    payload: RoleUpdate,
    admin: dict = Depends(require_role("super_admin")),
):
    """Assign a role to a user (super_admin only)."""
    result = users_collection.update_one(
        {"username": username},
        {"$set": {"role": payload.role}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found.")

    log_audit_event(
        "role_change",
        admin["username"],
        target=username,
        detail={"new_role": payload.role},
    )

    return {"message": f"Role updated to '{payload.role}' for user '{username}'."}


# ── Admin: deactivate user ────────────────────────────────────────────────────

@router.patch("/users/{username}/deactivate")
def deactivate_user(
    username: str,
    admin: dict = Depends(require_role("admin")),
):
    """Deactivate a user account (admin+)."""
    result = users_collection.update_one(
        {"username": username},
        {"$set": {"is_active": False}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found.")

    log_audit_event("deactivate_user", admin["username"], target=username)
    return {"message": f"User '{username}' deactivated."}