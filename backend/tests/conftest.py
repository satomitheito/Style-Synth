"""
Shared pytest fixtures for backend tests.
"""

import io
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def mock_db_connection():
    """
    Mock asyncpg connection with common database operations.
    """
    mock_conn = AsyncMock()

    # Default return values
    mock_conn.fetchval.return_value = 1  # Default item_id
    mock_conn.fetch.return_value = []    # Default empty list
    mock_conn.execute.return_value = "OK"

    return mock_conn


@pytest.fixture
def mock_get_db(mock_db_connection):
    """
    Override the get_db dependency for FastAPI tests.
    """
    async def override_get_db():
        yield mock_db_connection
    return override_get_db


# =============================================================================
# S3 Fixtures
# =============================================================================

@pytest.fixture
def mock_s3_client():
    """
    Mock boto3 S3 client.
    """
    mock_client = MagicMock()
    mock_client.put_object.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
    mock_client.generate_presigned_url.return_value = "https://bucket.s3.amazonaws.com/key?sig=xxx"
    mock_client.get_bucket_location.return_value = {"LocationConstraint": "us-east-1"}
    return mock_client


# =============================================================================
# ML Model Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_vector():
    """
    Generate a mock 2048-dimensional embedding.
    """
    np.random.seed(42)
    return np.random.randn(2048).astype(np.float32)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_image_bytes():
    """
    Generate a minimal valid PNG image as bytes.
    """
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_pil_image():
    """
    Generate a PIL Image object for testing.
    """
    return Image.new("RGB", (224, 224), color="blue")


@pytest.fixture
def sample_wardrobe_items():
    """
    Mock wardrobe items from database.
    """
    return [
        {
            "item_id": 1,
            "image_url": "s3://bucket/wardrobe/item1.jpg",
            "category": "top",
            "metadata": {"subcategory": "t-shirt", "colors": ["red"], "occasions": ["casual"]}
        },
        {
            "item_id": 2,
            "image_url": "s3://bucket/wardrobe/item2.jpg",
            "category": "bottom",
            "metadata": {"subcategory": "jeans", "colors": ["blue"], "occasions": ["casual"]}
        },
    ]


@pytest.fixture
def sample_saved_outfits():
    """
    Mock saved outfits from database.
    """
    from datetime import datetime
    return [
        {
            "outfit_id": "uuid-1",
            "items": [1, 2],
            "occasion": "casual",
            "season": "summer",
            "name": "Summer Casual",
            "created_at": datetime.now()
        },
    ]


# =============================================================================
# FastAPI Test Client
# =============================================================================

@pytest.fixture
def test_client(mock_db_connection, mock_embedding_vector):
    """
    FastAPI TestClient with all dependencies mocked.
    """
    from starlette.testclient import TestClient
    from backend.app.main import app
    from backend.app.database.connection import get_db

    # Override database dependency
    async def override_get_db():
        yield mock_db_connection

    app.dependency_overrides[get_db] = override_get_db

    # Create patches for external services
    with patch('backend.app.api.endpoints.upload_file_to_s3', new_callable=AsyncMock) as mock_upload:
        with patch('backend.app.api.endpoints.get_presigned_url') as mock_presigned:
            with patch('backend.app.api.endpoints.compute_embedding') as mock_embed:
                with patch('backend.app.api.endpoints.classify_image') as mock_classify:
                    with patch('backend.app.api.endpoints.find_similar_items', new_callable=AsyncMock) as mock_similar:
                        # Configure mocks
                        mock_upload.return_value = "s3://test-bucket/wardrobe/test.jpg"
                        mock_presigned.side_effect = lambda uri, exp=3600: f"https://presigned/{uri}"
                        mock_embed.return_value = mock_embedding_vector
                        mock_classify.return_value = {
                            "embedding": mock_embedding_vector,
                            "label": "T-shirt",
                            "confidence": 0.95,
                            "nearest_index": 42
                        }
                        mock_similar.return_value = [
                            {"item_id": 1, "distance": 0.1},
                            {"item_id": 2, "distance": 0.2},
                        ]

                        # Store mocks for access in tests
                        # Use transport for newer httpx compatibility
                        with TestClient(app) as client:
                            client.mock_db = mock_db_connection
                            client.mock_upload = mock_upload
                            client.mock_presigned = mock_presigned
                            client.mock_embed = mock_embed
                            client.mock_classify = mock_classify
                            client.mock_similar = mock_similar

                            yield client

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
def test_client_with_recommender(mock_db_connection, mock_embedding_vector):
    """
    FastAPI TestClient with recommender also mocked (for outfit tests).
    """
    from starlette.testclient import TestClient
    from backend.app.main import app
    from backend.app.database.connection import get_db

    # Override database dependency
    async def override_get_db():
        yield mock_db_connection

    app.dependency_overrides[get_db] = override_get_db

    # Create mock recommender
    mock_recommender = MagicMock()
    mock_recommender.recommend_outfits = AsyncMock(return_value=[
        {"items": [1, 2], "score": 0.85, "num_items": 2, "ml_powered": True},
        {"items": [3, 4], "score": 0.80, "num_items": 2, "ml_powered": False},
    ])

    with patch('backend.app.api.endpoints.recommender', mock_recommender):
        with patch('backend.app.api.endpoints.get_presigned_url') as mock_presigned:
            mock_presigned.side_effect = lambda uri, exp=3600: f"https://presigned/{uri}"

            with TestClient(app) as client:
                client.mock_db = mock_db_connection
                client.mock_recommender = mock_recommender
                client.mock_presigned = mock_presigned

                yield client

    # Clean up
    app.dependency_overrides.clear()


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
