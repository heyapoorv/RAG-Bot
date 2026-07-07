"""
Dashboard routes.
Exposes basic overview metrics for the authenticated user.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from services.auth_service import get_current_user, role_level
from services.db import analytics_collection
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/overview")
def overview(user: dict = Depends(get_current_user)):
    """
    Get user overview metrics.
    Analysts/admins see global counts.
    Standard users see only counts for their namespace.
    """
    u_role = user.get("role", "user")
    u_name = user.get("username")

    query_filter: dict = {}
    if role_level(u_role) < role_level("analyst"):
        # standard users scoped to their own username namespace
        query_filter["namespace"] = u_name

    total = analytics_collection.count_documents(query_filter)

    verified_filter = {**query_filter, "verified": True}
    verified = analytics_collection.count_documents(verified_filter)

    return {
        "total_queries": total,
        "verified": verified,
    }