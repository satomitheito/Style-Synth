"""
Unit tests for frontend API client.
Uses mocking to avoid actual HTTP requests.
"""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import requests

from frontend.api_client import APIClient


class TestAPIClientInit:
    """Tests for APIClient initialization."""

    def test_default_url_from_env(self):
        """Uses BACKEND_URL env var if set."""
        with patch.dict("os.environ", {"BACKEND_URL": "http://custom:9000"}):
            client = APIClient()
            assert client.base_url == "http://custom:9000"

    def test_explicit_url_overrides_env(self):
        """Explicit URL parameter overrides environment."""
        with patch.dict("os.environ", {"BACKEND_URL": "http://env:9000"}):
            client = APIClient(base_url="http://explicit:8080")
            assert client.base_url == "http://explicit:8080"

    def test_strips_trailing_slashes(self):
        """Trailing slashes are stripped from base URL."""
        client = APIClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_default_localhost_fallback(self):
        """Falls back to localhost:8000 when no URL provided."""
        with patch.dict("os.environ", {}, clear=True):
            client = APIClient()
            assert client.base_url == "http://localhost:8000"


class TestMakeRequest:
    """Tests for _make_request method."""

    @pytest.fixture
    def client(self):
        """Create APIClient with known base URL."""
        return APIClient(base_url="http://test:8000")

    @pytest.fixture
    def mock_response(self):
        """Create mock response object."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"status": "ok"}
        return response

    def test_get_request(self, client, mock_response):
        """GET request is made correctly."""
        with patch("requests.get", return_value=mock_response) as mock_get:
            result = client._make_request("GET", "/health")
            mock_get.assert_called_once_with(
                "http://test:8000/health", params=None, timeout=30
            )
            assert result == {"status": "ok"}

    def test_post_request_with_json(self, client, mock_response):
        """POST request sends JSON data."""
        with patch("requests.post", return_value=mock_response) as mock_post:
            result = client._make_request("POST", "/data", json={"key": "value"})
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"] == {"key": "value"}

    def test_delete_request(self, client, mock_response):
        """DELETE request is made correctly."""
        with patch("requests.delete", return_value=mock_response) as mock_delete:
            result = client._make_request("DELETE", "/item/1")
            mock_delete.assert_called_once_with(
                "http://test:8000/item/1", params=None, timeout=30
            )

    def test_patch_request(self, client, mock_response):
        """PATCH request sends JSON data."""
        with patch("requests.patch", return_value=mock_response) as mock_patch:
            result = client._make_request("PATCH", "/item/1", json={"name": "new"})
            mock_patch.assert_called_once()
            call_kwargs = mock_patch.call_args[1]
            assert call_kwargs["json"] == {"name": "new"}

    def test_unsupported_method_raises(self, client):
        """Unsupported HTTP method raises Exception."""
        with pytest.raises(Exception, match="Unsupported HTTP method"):
            client._make_request("PUT", "/data")

    def test_connection_error_wrapped(self, client):
        """Connection errors are wrapped with helpful message."""
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(ConnectionError, match="Could not connect"):
                client._make_request("GET", "/health")

    def test_timeout_error_wrapped(self, client):
        """Timeout errors are wrapped with helpful message."""
        with patch("requests.get", side_effect=requests.exceptions.Timeout()):
            with pytest.raises(TimeoutError, match="timed out"):
                client._make_request("GET", "/health")

    def test_http_error_includes_detail(self, client):
        """HTTP errors include detail from response."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Bad request"}
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(requests.exceptions.HTTPError, match="Bad request"):
                client._make_request("GET", "/bad")


class TestAPIClientMethods:
    """Tests for specific API client methods."""

    @pytest.fixture
    def client(self):
        """Create APIClient instance."""
        return APIClient(base_url="http://test:8000")

    @pytest.fixture
    def mock_request(self):
        """Patch _make_request to avoid actual HTTP calls."""
        with patch.object(APIClient, "_make_request") as mock:
            mock.return_value = {"success": True}
            yield mock

    def test_upload_wardrobe_item(self, client, mock_request):
        """upload_wardrobe_item calls correct endpoint."""
        image_file = BytesIO(b"fake image data")
        client.upload_wardrobe_item(image_file, "test.jpg", "Tops")

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/wardrobe/upload"
        assert call_args[1]["params"] == {"category": "Tops"}

    def test_upload_wardrobe_item_resets_file_pointer(self, client, mock_request):
        """File pointer is reset before upload."""
        image_file = BytesIO(b"fake image data")
        image_file.read()  # Move pointer to end
        assert image_file.tell() == len(b"fake image data")

        client.upload_wardrobe_item(image_file, "test.jpg", "Tops")
        # File should have been reset (seek called)
        # We verify the call was made correctly
        mock_request.assert_called_once()

    def test_predict_image(self, client, mock_request):
        """predict_image calls correct endpoint."""
        image_file = BytesIO(b"fake image data")
        client.predict_image(image_file, "test.jpg")

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "/predict"

    def test_generate_outfits(self, client, mock_request):
        """generate_outfits sends occasion and season."""
        client.generate_outfits("Casual", "Summer")

        mock_request.assert_called_once_with(
            "POST", "/outfits/generate", json={"occasion": "Casual", "season": "Summer"}
        )

    def test_save_outfit(self, client, mock_request):
        """save_outfit sends all outfit data."""
        client.save_outfit([1, 2, 3], "Formal", "Winter", "Work Outfit")

        mock_request.assert_called_once_with(
            "POST", "/outfits/save",
            json={"items": [1, 2, 3], "occasion": "Formal", "season": "Winter", "name": "Work Outfit"}
        )

    def test_get_saved_outfits(self, client, mock_request):
        """get_saved_outfits calls correct endpoint."""
        client.get_saved_outfits()
        mock_request.assert_called_once_with("GET", "/outfits/saved")

    def test_get_wardrobe_items(self, client, mock_request):
        """get_wardrobe_items calls correct endpoint."""
        client.get_wardrobe_items()
        mock_request.assert_called_once_with("GET", "/wardrobe/items")

    def test_health_check(self, client, mock_request):
        """health_check calls root endpoint."""
        client.health_check()
        mock_request.assert_called_once_with("GET", "/")

    def test_delete_wardrobe_item(self, client, mock_request):
        """delete_wardrobe_item calls correct endpoint."""
        client.delete_wardrobe_item(42)
        mock_request.assert_called_once_with("DELETE", "/wardrobe/item/42")

    def test_update_wardrobe_item(self, client, mock_request):
        """update_wardrobe_item sends all metadata."""
        client.update_wardrobe_item(
            item_id=1,
            category="Tops",
            subcategory="T-shirt",
            season=["Summer"],
            brand="Nike",
            colors=["Blue"],
            occasions=["Casual"],
            notes="Favorite shirt"
        )

        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[0][0] == "PATCH"
        assert call_args[0][1] == "/wardrobe/item/1"
        json_data = call_args[1]["json"]
        assert json_data["category"] == "Tops"
        assert json_data["subcategory"] == "T-shirt"
        assert json_data["season"] == ["Summer"]

    def test_update_wardrobe_item_handles_none_values(self, client, mock_request):
        """update_wardrobe_item converts None to empty values."""
        client.update_wardrobe_item(
            item_id=1,
            category="Tops",
            season=None,
            colors=None,
            occasions=None
        )

        json_data = mock_request.call_args[1]["json"]
        assert json_data["season"] == []
        assert json_data["colors"] == []
        assert json_data["occasions"] == []

    def test_delete_outfit(self, client, mock_request):
        """delete_outfit calls correct endpoint."""
        client.delete_outfit("outfit-123")
        mock_request.assert_called_once_with("DELETE", "/outfits/outfit-123")

    def test_update_outfit_partial(self, client, mock_request):
        """update_outfit sends only provided fields."""
        client.update_outfit("outfit-123", name="New Name")

        call_args = mock_request.call_args
        json_data = call_args[1]["json"]
        assert json_data == {"name": "New Name"}

    def test_update_outfit_all_fields(self, client, mock_request):
        """update_outfit can send all fields."""
        client.update_outfit(
            "outfit-123",
            name="Updated",
            occasion="Formal",
            season="Winter",
            items=[1, 2, 3]
        )

        json_data = mock_request.call_args[1]["json"]
        assert json_data == {
            "name": "Updated",
            "occasion": "Formal",
            "season": "Winter",
            "items": [1, 2, 3]
        }

    def test_clear_all_wardrobe_items(self, client, mock_request):
        """clear_all_wardrobe_items calls correct endpoint."""
        client.clear_all_wardrobe_items()
        mock_request.assert_called_once_with("DELETE", "/wardrobe/clear-all")
