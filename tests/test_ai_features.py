"""
Tests for AI intelligence features: summarization, clauses, comparison.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@patch("services.ai_features._gemini")
@patch("services.ai_features._get_document_chunks")
def test_summarize_success(mock_chunks, mock_gemini_call, app_client, admin_headers):
    """Admin can summarize a document in any namespace."""
    mock_chunks.return_value = ["Sample sentence chunk."]
    mock_gemini_call.return_value = '{"summary": "Short summary", "key_topics": ["topic1"]}'

    response = app_client.post(
        "/documents/summarize",
        json={"document_id": "doc123", "namespace": "ns_test", "max_length": 200},
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "Short summary"
    assert data["key_topics"] == ["topic1"]


@patch("services.ai_features._gemini")
@patch("services.ai_features._get_document_chunks")
def test_summarize_user_own_namespace(mock_chunks, mock_gemini_call, app_client, auth_headers):
    """A regular user can only summarize documents in their own namespace (== username)."""
    mock_chunks.return_value = ["Sample sentence chunk."]
    mock_gemini_call.return_value = '{"summary": "My summary", "key_topics": ["my_topic"]}'

    # auth_headers fixture uses username="testuser", so namespace must equal "testuser"
    response = app_client.post(
        "/documents/summarize",
        json={"document_id": "doc123", "namespace": "testuser", "max_length": 200},
        headers=auth_headers,
    )
    assert response.status_code == 200


@patch("services.ai_features._gemini")
@patch("services.ai_features._get_document_chunks")
def test_summarize_user_wrong_namespace_forbidden(mock_chunks, mock_gemini_call, app_client, auth_headers):
    """A regular user cannot access another user's namespace."""
    mock_chunks.return_value = []
    response = app_client.post(
        "/documents/summarize",
        json={"document_id": "doc123", "namespace": "other_user", "max_length": 200},
        headers=auth_headers,
    )
    assert response.status_code == 403


@patch("services.ai_features._gemini")
@patch("services.ai_features._get_document_chunks")
def test_clause_extraction_success(mock_chunks, mock_gemini_call, app_client, admin_headers):
    """Admin can extract clauses from any namespace."""
    mock_chunks.return_value = ["Contract terms paragraph."]
    mock_gemini_call.return_value = (
        '[{"title": "Term", "type": "Duration", "text": "2 years", "risk_level": "low"}]'
    )

    response = app_client.post(
        "/documents/clauses?document_id=doc123&namespace=ns_test",
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_clauses"] == 1
    assert data["clauses"][0]["title"] == "Term"

