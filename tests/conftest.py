"""
pytest configuration and hermetic mocking.
Mocks all external services (Pinecone, MongoDB, Gemini) at the sys.modules level
before any application imports can trigger network requests.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# ── Hermetic Mocks (MUST run before any local imports) ───────────────────────

# 1. Mock Pinecone
mock_pinecone_lib = MagicMock()
mock_pc_instance = MagicMock()
mock_index = MagicMock()
mock_pc_instance.Index.return_value = mock_index
mock_pinecone_lib.Pinecone.return_value = mock_pc_instance
sys.modules["pinecone"] = mock_pinecone_lib

# 2. Mock pymongo package and submodules
mock_pymongo_lib = MagicMock()
mock_pymongo_lib.__path__ = []
mock_mongo_client = MagicMock()
# Make admin ping command succeed
mock_mongo_client.admin.command.return_value = {"ok": 1.0}
mock_pymongo_lib.MongoClient.return_value = mock_mongo_client
sys.modules["pymongo"] = mock_pymongo_lib

mock_collection_module = MagicMock()
mock_collection_module.Collection = MagicMock
sys.modules["pymongo.collection"] = mock_collection_module

mock_database_module = MagicMock()
sys.modules["pymongo.database"] = mock_database_module

# Mock pymongo.UpdateOne used by bm25.py bulk_write
mock_pymongo_lib.UpdateOne = MagicMock()

# 3. Mock google.genai
mock_genai_lib = MagicMock()
mock_genai_client = MagicMock()
mock_genai_lib.Client.return_value = mock_genai_client
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = mock_genai_lib
sys.modules["google"].genai = mock_genai_lib

# 4. Mock sentence-transformers
mock_st_lib = MagicMock()
sys.modules["sentence_transformers"] = mock_st_lib

# 5. Mock bcrypt (tests don't need real hashing)
import hashlib
mock_bcrypt = MagicMock()
mock_bcrypt.gensalt.return_value = b"$2b$12$testsalt"
mock_bcrypt.hashpw.side_effect = lambda pw, salt: hashlib.sha256(pw).hexdigest().encode()
mock_bcrypt.checkpw.side_effect = lambda pw, hashed: hashlib.sha256(pw).hexdigest().encode() == hashed
sys.modules["bcrypt"] = mock_bcrypt

# ── Configure Test Environment ────────────────────────────────────────────────
os.environ.update(
    {
        "ENVIRONMENT": "development",
        "MONGODB_URI": "mongodb://localhost:27017",
        "MONGODB_DB_NAME": "test_dociintel",
        "GOOGLE_API_KEY": "test-key",
        "PINECONE_API_KEY": "test-key",
        "PINECONE_INDEX_NAME": "newrag",
        "JWT_SECRET": "test-secret-32-characters-minimum-x",
        "CACHE_ENABLED": "false",
        "VERIFICATION_ENABLED": "false",
        "PROMETHEUS_ENABLED": "false",
    }
)

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pinecone_mock():
    """Access the mocked Pinecone index."""
    return mock_index


@pytest.fixture(scope="session")
def gemini_mock():
    """Access the mocked Gemini client."""
    return mock_genai_client


@pytest.fixture(scope="session")
def mongo_mock():
    """Access the mocked MongoClient."""
    return mock_mongo_client


@pytest.fixture(scope="session", autouse=True)
def mock_embedding_globally():
    """Mock embed_texts globally to avoid real embedding calls."""
    with patch("services.embedding.embed_texts") as mock_fn:
        mock_fn.return_value = [[0.1] * 768]  # match GEMINI_EMBED_DIM=768
        yield mock_fn


@pytest.fixture
def test_user_payload():
    return {"username": "testuser", "role": "user", "exp": 9999999999}


@pytest.fixture
def test_admin_payload():
    return {"username": "admin", "role": "admin", "exp": 9999999999}


@pytest.fixture
def test_analyst_payload():
    return {"username": "analyst", "role": "analyst", "exp": 9999999999}


@pytest.fixture
def auth_headers(test_user_payload):
    from services.auth_service import create_token
    token = create_token(test_user_payload)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(test_admin_payload):
    from services.auth_service import create_token
    token = create_token(test_admin_payload)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def analyst_headers(test_analyst_payload):
    from services.auth_service import create_token
    token = create_token(test_analyst_payload)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def app_client():
    """FastAPI TestClient with all mocks applied."""
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as client:
        yield client

# Helper utility to avoid duplicate patch code in test files
from unittest.mock import patch
