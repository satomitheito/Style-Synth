"""
Unit tests for embedding service functions.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from PIL import Image


@pytest.mark.unit
class TestComputeEmbedding:
    """Tests for compute_embedding function."""

    def test_compute_embedding_returns_2048_dims(self, sample_pil_image, mock_embedding_vector):
        """Test that compute_embedding returns a 2048-D vector."""
        with patch('backend.app.services.embedding_service.embedding_model') as mock_model:
            with patch('backend.app.services.embedding_service.ML_INFERENCE_TIME') as mock_time:
                import torch

                # Configure mock to return correct shape tensor
                mock_output = torch.zeros(1, 2048, 1, 1)
                mock_model.return_value = mock_output
                mock_time.labels.return_value.observe = MagicMock()

                from backend.app.services.embedding_service import compute_embedding

                result = compute_embedding(sample_pil_image)

                assert isinstance(result, np.ndarray)
                assert result.shape == (2048,)


@pytest.mark.unit
class TestClassifyImage:
    """Tests for classify_image function."""

    def test_classify_image_returns_label(self, sample_pil_image, mock_embedding_vector):
        """Test that classify_image returns expected structure."""
        with patch('backend.app.services.embedding_service.compute_embedding') as mock_embed:
            with patch('backend.app.services.embedding_service.find_nearest_neighbor') as mock_nn:
                with patch('backend.app.services.embedding_service.ML_INFERENCE_TIME') as mock_time:
                    # Configure mocks
                    mock_embed.return_value = mock_embedding_vector
                    mock_nn.return_value = ("T-shirt", 0.95, 42)
                    mock_time.labels.return_value.observe = MagicMock()

                    from backend.app.services.embedding_service import classify_image

                    result = classify_image(sample_pil_image)

                    assert "embedding" in result
                    assert "label" in result
                    assert "confidence" in result
                    assert "nearest_index" in result
                    assert result["label"] == "T-shirt"
                    assert result["confidence"] == 0.95
                    assert result["nearest_index"] == 42


@pytest.mark.unit
class TestFindSimilarItems:
    """Tests for find_similar_items function."""

    @pytest.mark.asyncio
    async def test_find_similar_items_returns_list(self, mock_db_connection, mock_embedding_vector):
        """Test that find_similar_items queries database and returns results."""
        # Configure mock to return sample results
        mock_db_connection.fetch.return_value = [
            {"item_id": 1, "distance": 0.1},
            {"item_id": 2, "distance": 0.2},
            {"item_id": 3, "distance": 0.3},
        ]

        from backend.app.services.embedding_service import find_similar_items

        result = await find_similar_items(
            vector=mock_embedding_vector,
            conn=mock_db_connection,
            limit=5
        )

        assert len(result) == 3
        assert result[0]["item_id"] == 1
        assert result[0]["distance"] == 0.1
        mock_db_connection.fetch.assert_called_once()


@pytest.mark.unit
class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation is correct."""
        from backend.app.services.embedding_service import cosine_similarity

        # Test with identical vectors (should be 1.0)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[1.0, 0.0, 0.0]])

        result = cosine_similarity(a, b)
        assert np.isclose(result[0], 1.0, atol=1e-6)

        # Test with orthogonal vectors (should be 0.0)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[0.0, 1.0, 0.0]])

        result = cosine_similarity(a, b)
        assert np.isclose(result[0], 0.0, atol=1e-6)

        # Test with opposite vectors (should be -1.0)
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([[-1.0, 0.0, 0.0]])

        result = cosine_similarity(a, b)
        assert np.isclose(result[0], -1.0, atol=1e-6)
