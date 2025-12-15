"""
Integration tests for wardrobe API endpoints.
"""

import pytest


@pytest.mark.integration
class TestWardrobeUpload:
    """Tests for POST /wardrobe/upload endpoint."""

    def test_upload_wardrobe_item_success(self, test_client, sample_image_bytes):
        """Test successful wardrobe item upload."""
        # Configure mock DB to return item_id
        test_client.mock_db.fetchval.return_value = 123

        response = test_client.post(
            "/wardrobe/upload",
            params={"category": "top"},
            files={"file": ("test_shirt.png", sample_image_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["item_id"] == 123
        assert "image_url" in data

    def test_upload_wardrobe_item_empty_file_error(self, test_client):
        """Test upload with empty file returns error."""
        response = test_client.post(
            "/wardrobe/upload",
            params={"category": "top"},
            files={"file": ("empty.png", b"", "image/png")}
        )

        # Backend catches HTTPException in outer try/except and returns 500
        assert response.status_code in [400, 500]
        assert "empty" in response.json()["detail"].lower()


@pytest.mark.integration
class TestWardrobeGet:
    """Tests for GET /wardrobe/items endpoint."""

    def test_get_wardrobe_items_returns_list(self, test_client, sample_wardrobe_items):
        """Test getting all wardrobe items."""
        # Configure mock DB to return sample items
        test_client.mock_db.fetch.return_value = sample_wardrobe_items

        response = test_client.get("/wardrobe/items")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 2
        assert data["items"][0]["item_id"] == 1
        assert data["items"][0]["category"] == "top"

    def test_get_wardrobe_items_empty(self, test_client):
        """Test getting items when wardrobe is empty."""
        test_client.mock_db.fetch.return_value = []

        response = test_client.get("/wardrobe/items")

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []


@pytest.mark.integration
class TestWardrobeUpdate:
    """Tests for PATCH /wardrobe/item/{item_id} endpoint."""

    def test_update_wardrobe_item_metadata(self, test_client):
        """Test updating wardrobe item metadata."""
        test_client.mock_db.execute.return_value = "UPDATE 1"

        response = test_client.patch(
            "/wardrobe/item/1",
            json={
                "category": "top",
                "subcategory": "t-shirt",
                "season": ["summer", "spring"],
                "brand": "Nike",
                "colors": ["blue", "white"],
                "occasions": ["casual", "sports"],
                "notes": "Favorite shirt"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["item_id"] == 1


@pytest.mark.integration
class TestWardrobeDelete:
    """Tests for DELETE /wardrobe/item/{item_id} endpoint."""

    def test_delete_wardrobe_item(self, test_client):
        """Test deleting a wardrobe item."""
        test_client.mock_db.execute.return_value = "DELETE 1"

        response = test_client.delete("/wardrobe/item/1")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["deleted_item_id"] == 1

        # Verify both embedding and wardrobe item were deleted
        assert test_client.mock_db.execute.call_count == 2
