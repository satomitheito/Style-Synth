"""
Integration tests for outfit API endpoints.
"""

import pytest
from datetime import datetime


@pytest.mark.integration
class TestGenerateOutfits:
    """Tests for POST /outfits/generate endpoint."""

    def test_generate_outfits_returns_recommendations(self, test_client_with_recommender):
        """Test generating outfit recommendations."""
        response = test_client_with_recommender.post(
            "/outfits/generate",
            json={"occasion": "casual", "season": "summer"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["occasion"] == "casual"
        assert data["season"] == "summer"
        assert "count" in data
        assert "outfits" in data
        assert len(data["outfits"]) == 2  # From mock


@pytest.mark.integration
class TestSaveOutfit:
    """Tests for POST /outfits/save endpoint."""

    def test_save_outfit_success(self, test_client_with_recommender):
        """Test saving an outfit."""
        test_client_with_recommender.mock_db.execute.return_value = "INSERT"

        response = test_client_with_recommender.post(
            "/outfits/save",
            json={
                "items": [1, 2, 3],
                "occasion": "casual",
                "season": "summer",
                "name": "Beach Day"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "outfit_id" in data


@pytest.mark.integration
class TestGetSavedOutfits:
    """Tests for GET /outfits/saved endpoint."""

    def test_get_saved_outfits(self, test_client_with_recommender, sample_wardrobe_items):
        """Test getting saved outfits with enriched item data."""
        # Mock saved outfits
        saved_outfits = [
            {
                "outfit_id": "uuid-123",
                "items": [1, 2],
                "occasion": "casual",
                "season": "summer",
                "name": "Summer Casual",
                "created_at": datetime.now()
            }
        ]

        # Configure mock DB - first call returns outfits, second returns wardrobe items
        test_client_with_recommender.mock_db.fetch.side_effect = [
            saved_outfits,
            sample_wardrobe_items
        ]

        response = test_client_with_recommender.get("/outfits/saved")

        assert response.status_code == 200
        data = response.json()

        assert "saved_outfits" in data
        assert len(data["saved_outfits"]) == 1

        outfit = data["saved_outfits"][0]
        assert outfit["outfit_id"] == "uuid-123"
        assert outfit["name"] == "Summer Casual"
        assert outfit["occasion"] == "casual"
        assert outfit["season"] == "summer"
        assert "items" in outfit


@pytest.mark.integration
class TestDeleteOutfit:
    """Tests for DELETE /outfits/{outfit_id} endpoint."""

    def test_delete_outfit(self, test_client_with_recommender):
        """Test deleting an outfit."""
        test_client_with_recommender.mock_db.execute.return_value = "DELETE 1"

        response = test_client_with_recommender.delete("/outfits/uuid-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["deleted_outfit_id"] == "uuid-123"
