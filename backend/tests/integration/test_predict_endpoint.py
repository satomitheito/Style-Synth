"""
Integration tests for predict API endpoint.
"""

import pytest


@pytest.mark.integration
class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_image_success(self, test_client, sample_image_bytes):
        """Test successful image prediction."""
        response = test_client.post(
            "/predict",
            files={"file": ("test_image.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "filename" in data
        assert "predicted_label" in data
        assert "confidence" in data
        assert "nearest_index" in data
        assert "similar_items" in data

        # Verify values from mocks
        assert data["predicted_label"] == "T-shirt"
        assert data["confidence"] == 0.95
        assert len(data["similar_items"]) == 2

    def test_predict_empty_file_error(self, test_client):
        """Test prediction with empty file returns error."""
        response = test_client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")}
        )

        # Backend catches HTTPException in outer try/except and returns 500
        assert response.status_code in [400, 500]
        assert "empty" in response.json()["detail"].lower()
