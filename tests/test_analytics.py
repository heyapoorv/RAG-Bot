"""
Tests for analytics queries, aggregation, and permissions.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@patch("routes.analytics.analytics_collection")
def test_get_summary_analytics_admin(mock_coll, app_client, admin_headers):
    # Mock aggregation pipeline result
    mock_coll.aggregate.return_value = [
        {
            "total_queries": 10,
            "avg_latency": 250.5,
            "cache_hits": 4,
            "verified_answers": 8,
            "failures": 1,
        }
    ]

    response = app_client.get(
        "/analytics/summary?days=7",
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_queries"] == 10
    assert data["avg_latency_ms"] == 250.5
    assert data["cache_hit_rate"] == 0.4
    assert data["verification_rate"] == 0.8
    assert data["failure_rate"] == 0.1


@patch("routes.analytics.analytics_collection")
def test_get_summary_analytics_user_scoping_violation(mock_coll, app_client, auth_headers):
    # Standard user requests namespace different from their own (testuser)
    response = app_client.get(
        "/analytics/summary?namespace=other_namespace",
        headers=auth_headers,
    )
    assert response.status_code == 403
    assert "Forbidden" in response.json()["detail"]


@patch("routes.analytics.analytics_collection")
def test_get_summary_analytics_user_scoping_allowed(mock_coll, app_client, auth_headers):
    mock_coll.aggregate.return_value = []
    response = app_client.get(
        "/analytics/summary?namespace=testuser",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_queries"] == 0
