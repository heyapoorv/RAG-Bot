"""
Tests for reranker options and execution logic.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from services.reranker import rerank_chunks, preload_reranker


def test_rerank_empty_chunks():
    result = rerank_chunks("query", [])
    assert result == []


@patch("services.reranker.embed_texts")
def test_local_rerank_scores(mock_embed):
    # Setup mocks
    # Query embedding and 2 chunk embeddings
    mock_embed.side_effect = [
        [[1.0, 0.0]],  # query
        [[0.9, 0.1], [0.1, 0.9]],  # chunks (first matches query more)
    ]

    chunks = [
        {"chunk_text": "Close match doc"},
        {"chunk_text": "Far match doc"},
    ]

    result = rerank_chunks(
        query="test query",
        chunks=chunks,
        top_n=2,
        mode="local",
    )

    assert len(result) == 2
    assert result[0]["chunk_text"] == "Close match doc"
    assert "rerank_score" in result[0]
    assert result[0]["rerank_score"] > result[1]["rerank_score"]


@patch("services.reranker._load_cross_encoder")
def test_cross_encoder_fallback_on_error(mock_load):
    # Simulate loading model works but prediction fails
    mock_model = MagicMock()
    mock_model.predict.side_effect = Exception("GPU out of memory")
    mock_load.return_value = mock_model

    chunks = [{"chunk_text": "Doc 1"}, {"chunk_text": "Doc 2"}]

    with patch("services.reranker._local_rerank") as mock_local:
        mock_local.return_value = chunks
        res = rerank_chunks(
            query="test",
            chunks=chunks,
            top_n=2,
            mode="cross_encoder",
            model_name="dummy",
        )
        assert res == chunks
        mock_local.assert_called_once()
