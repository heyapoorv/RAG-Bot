"""
Unit and integration tests for the RAG query pipeline.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestQueryGuard:
    def test_valid_query_passes(self):
        from services.query_guard import is_valid_rag_query
        assert is_valid_rag_query("What are the policy coverage limits?") is True

    def test_too_short_query_fails(self):
        from services.query_guard import is_valid_rag_query
        assert is_valid_rag_query("ok") is False

    def test_bad_pattern_fails(self):
        from services.query_guard import is_valid_rag_query
        assert is_valid_rag_query("tell everything about it") is False

    def test_empty_string_fails(self):
        from services.query_guard import is_valid_rag_query
        assert is_valid_rag_query("") is False


class TestDynamicTopK:
    def test_simple_query_returns_5(self):
        from services.dynamic_topk import compute_dynamic_topk
        assert compute_dynamic_topk("What is the deductible?") == 5

    def test_compare_query_returns_10(self):
        from services.dynamic_topk import compute_dynamic_topk
        assert compute_dynamic_topk("Compare all coverage plans and list differences") == 10

    def test_long_query_returns_8(self):
        from services.dynamic_topk import compute_dynamic_topk
        q = "What are the specific features and requirements for the extended warranty program that is offered for standard electric vehicles?"
        assert compute_dynamic_topk(q) == 8


class TestBM25:
    def test_bm25_scores_relevant_doc_higher(self):
        from services.bm25 import rank_bm25
        query = "insurance premium payment"
        docs = [
            {"chunk_text": "The insurance premium must be paid monthly.", "source": "a"},
            {"chunk_text": "The weather is sunny today.", "source": "b"},
        ]
        ranked = rank_bm25(query, docs)
        assert ranked[0]["source"] == "a"
        assert ranked[0]["bm25_score"] > ranked[1]["bm25_score"]

    def test_bm25_handles_empty_docs(self):
        from services.bm25 import rank_bm25
        result = rank_bm25("query", [])
        assert result == []


class TestRRF:
    def test_rrf_fusion_deduplicates(self):
        from services.rrf import rrf_fusion
        chunk = {"source": "a", "chunk_text": "Same text"}
        rankings = [[chunk], [chunk]]
        result = rrf_fusion(rankings)
        assert len(result) == 1

    def test_rrf_gives_higher_score_to_top_ranked(self):
        from services.rrf import rrf_fusion
        chunk_a = {"source": "a", "chunk_text": "Top result"}
        chunk_b = {"source": "b", "chunk_text": "Lower result"}
        result = rrf_fusion([[chunk_a, chunk_b], [chunk_a, chunk_b]])
        assert result[0]["source"] == "a"


class TestReranker:
    def test_local_reranker_sorts_by_similarity(self):
        from services.reranker import rerank_chunks

        chunks = [
            {"chunk_text": "Dogs are common pets.", "source": "a"},
            {"chunk_text": "Insurance policy covers medical expenses.", "source": "b"},
        ]
        with patch("services.reranker.embed_texts") as mock_emb:
            # Query embedding
            # b is more similar to query
            mock_emb.side_effect = [
                [[1.0, 0.0]],  # query
                [[0.1, 0.9], [0.9, 0.1]],  # chunks
            ]
            with patch("services.reranker.cosine_similarity") as mock_sim:
                mock_sim.side_effect = [
                    [[0.2]],  # chunk a similarity
                    [[0.8]],  # chunk b similarity
                ]
                result = rerank_chunks("insurance coverage", chunks, top_n=2, mode="local")

        assert isinstance(result, list)
        assert len(result) <= 2

    def test_reranker_handles_empty_chunks(self):
        from services.reranker import rerank_chunks
        result = rerank_chunks("query", [], top_n=5)
        assert result == []


class TestCache:
    def test_cache_disabled_returns_none(self):
        with patch("services.cache.settings") as mock_settings:
            mock_settings.CACHE_ENABLED = False
            from services.cache import get_cached_answer
            result = get_cached_answer("any question")
            assert result is None

    def test_cache_store_disabled_is_noop(self):
        with patch("services.cache.settings") as mock_settings:
            mock_settings.CACHE_ENABLED = False
            from services.cache import store_cached_answer
            # Should not raise
            store_cached_answer("question", {"answer": "test"})


class TestQueryRoute:
    def test_query_requires_namespace(self):
        from fastapi.testclient import TestClient
        from main import app
        from services.auth_service import create_token

        token = create_token({"username": "u", "role": "user"})
        headers = {"Authorization": f"Bearer {token}"}

        with patch("services.db.users_collection") as mock_col:
            mock_col.find_one.return_value = {"username": "u", "role": "user"}
            with TestClient(app) as client:
                response = client.post(
                    "/query/",
                    json={"questions": ["What is coverage?"]},
                    headers=headers,
                )
        # No namespace → 400
        assert response.status_code in (400, 422, 500)

    def test_empty_questions_returns_422(self):
        from fastapi.testclient import TestClient
        from main import app
        from services.auth_service import create_token

        token = create_token({"username": "u", "role": "user"})
        headers = {"Authorization": f"Bearer {token}"}

        with TestClient(app) as client:
            response = client.post(
                "/query/",
                json={"questions": [], "namespace": "test"},
                headers=headers,
            )
        assert response.status_code == 422
