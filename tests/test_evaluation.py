import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os

def test_calculate_metrics():
    from evaluation.evaluate_retrieval import calculate_metrics
    
    retrieved_chunks = [
        {"source": "policy.txt", "page": 1, "chunk_text": "text1"},
        {"source": "policy.txt", "page": 2, "chunk_text": "text2"},
        {"source": "other.txt", "page": 1, "chunk_text": "text3"}
    ]
    
    # Expected page is 1
    metrics = calculate_metrics(retrieved_chunks, "policy.txt", [1], k=5)
    
    assert metrics["precision"] == 0.2  # 1 hit out of 5 slots
    assert metrics["recall"] == 1.0  # page 1 retrieved
    assert metrics["rr"] == 1.0  # first hit is at index 0 (1/1)
    
    # Expected page is 2
    metrics_page2 = calculate_metrics(retrieved_chunks, "policy.txt", [2], k=5)
    assert metrics_page2["precision"] == 0.2  # 1 hit
    assert metrics_page2["recall"] == 1.0  # page 2 retrieved
    assert metrics_page2["rr"] == 0.5  # first hit is at index 1 (1/2)

def test_citation_correctness():
    from evaluation.evaluate_generation import calculate_citation_correctness
    
    # Test perfect match
    cited = {"1", "2"}
    expected = [1, 2]
    assert calculate_citation_correctness(cited, expected) == 1.0
    
    # Test partial mismatch
    cited = {"1", "3"}
    expected = [1, 2]
    assert calculate_citation_correctness(cited, expected) == 0.5

@pytest.mark.anyio
@pytest.mark.asyncio
async def test_ensure_sample_ingested():
    from evaluation.utils import EVAL_NAMESPACE
    with patch("services.db.documents_collection") as mock_col, \
         patch("services.vectordb.index") as mock_pinecone:

        # Mock: document already exists in MongoDB
        mock_col.find_one.return_value = {"document_id": "policy.txt"}

        # Mock: Pinecone already has vectors for the correct namespace
        mock_pinecone.describe_index_stats.return_value = {
            "namespaces": {
                EVAL_NAMESPACE: {"vector_count": 10}
            }
        }

        from evaluation.utils import ensure_sample_ingested
        # Should return immediately — both conditions met
        await ensure_sample_ingested()

        mock_col.find_one.assert_called_once()
