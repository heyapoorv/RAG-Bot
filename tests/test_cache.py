"""
Tests for persistent semantic cache system.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from services.cache import (
    get_cached_answer,
    store_cached_answer,
    invalidate_namespace,
    get_cache_stats,
)


def test_cache_stats_accumulation():
    stats = get_cache_stats()
    # Reset stats
    stats.hits = 0
    stats.misses = 0
    stats.stores = 0

    assert stats.hit_rate == 0.0

    stats.hits = 2
    stats.misses = 2
    assert stats.hit_rate == 0.5


def test_cache_retrieval_flow_disabled():
    with patch("services.cache.settings") as mock_settings:
        mock_settings.CACHE_ENABLED = False
        res = get_cached_answer("some question")
        assert res is None


@patch("services.cache.embed_texts")
@patch("services.cache.cache_collection")
def test_mongo_cache_hit(mock_coll, mock_embed):
    # Setup mocks
    mock_embed.return_value = [[1.0, 0.0]]
    # Simulate doc in db with high similarity
    mock_coll.find.return_value = [
        {
            "embedding": [0.99, 0.01],
            "response": {"answer": "Target Answer"},
            "question": "test question",
        }
    ]

    with patch("services.cache.settings") as mock_settings:
        mock_settings.CACHE_ENABLED = True
        mock_settings.CACHE_SIMILARITY_THRESHOLD = 0.90
        mock_settings.REDIS_URL = None  # Force MongoDB route

        ans = get_cached_answer("test question")
        assert ans == {"answer": "Target Answer"}


@patch("services.cache.embed_texts")
@patch("services.cache.cache_collection")
def test_mongo_cache_miss_low_similarity(mock_coll, mock_embed):
    mock_embed.return_value = [[1.0, 0.0]]
    mock_coll.find.return_value = [
        {
            "embedding": [0.1, 0.9],
            "response": {"answer": "Different Answer"},
            "question": "another query",
        }
    ]

    with patch("services.cache.settings") as mock_settings:
        mock_settings.CACHE_ENABLED = True
        mock_settings.CACHE_SIMILARITY_THRESHOLD = 0.90
        mock_settings.REDIS_URL = None

        ans = get_cached_answer("test question")
        assert ans is None
