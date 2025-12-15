"""
Unit tests for S3 service functions.
"""

import io
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestUploadFileToS3:
    """Tests for upload_file_to_s3 function."""

    @pytest.mark.asyncio
    async def test_upload_file_to_s3_returns_s3_uri(self, mock_s3_client):
        """Test that upload returns correct S3 URI format."""
        with patch('backend.app.services.s3_service.s3', mock_s3_client):
            # Need to also patch the metrics to avoid issues
            with patch('backend.app.services.s3_service.AWS_S3_TIME') as mock_time:
                with patch('backend.app.services.s3_service.AWS_S3_CALLS') as mock_calls:
                    with patch('backend.app.services.s3_service.AWS_S3_BYTES') as mock_bytes:
                        # Configure metric mocks
                        mock_time.labels.return_value.observe = MagicMock()
                        mock_calls.labels.return_value.inc = MagicMock()
                        mock_bytes.labels.return_value.inc = MagicMock()

                        from backend.app.services.s3_service import upload_file_to_s3

                        # Create a file-like object
                        file = io.BytesIO(b"test image content")

                        result = await upload_file_to_s3(
                            file=file,
                            bucket="test-bucket",
                            key="wardrobe/test.jpg"
                        )

                        assert result == "s3://test-bucket/wardrobe/test.jpg"
                        mock_s3_client.put_object.assert_called_once()


@pytest.mark.unit
class TestGetPresignedUrl:
    """Tests for get_presigned_url function."""

    def test_get_presigned_url_converts_s3_uri(self, mock_s3_client):
        """Test that S3 URI is converted to presigned URL."""
        with patch('backend.app.services.s3_service.s3', mock_s3_client):
            with patch('backend.app.services.s3_service.boto3') as mock_boto3:
                with patch('backend.app.services.s3_service.AWS_S3_TIME') as mock_time:
                    with patch('backend.app.services.s3_service.AWS_S3_CALLS') as mock_calls:
                        # Configure mocks
                        mock_time.labels.return_value.observe = MagicMock()
                        mock_calls.labels.return_value.inc = MagicMock()

                        # Mock the boto3.client call inside the function
                        mock_client = MagicMock()
                        mock_client.generate_presigned_url.return_value = "https://bucket.s3.amazonaws.com/key?sig=xxx"
                        mock_boto3.client.return_value = mock_client

                        from backend.app.services.s3_service import get_presigned_url

                        result = get_presigned_url("s3://my-bucket/wardrobe/image.jpg")

                        assert result.startswith("https://")
                        assert "s3.amazonaws.com" in result or "presigned" in result.lower() or result.startswith("https://")

    def test_get_presigned_url_returns_non_s3_unchanged(self):
        """Test that non-S3 URIs are returned unchanged."""
        from backend.app.services.s3_service import get_presigned_url

        # Test with regular HTTPS URL
        https_url = "https://example.com/image.jpg"
        result = get_presigned_url(https_url)
        assert result == https_url

        # Test with None
        result = get_presigned_url(None)
        assert result is None

        # Test with empty string
        result = get_presigned_url("")
        assert result == ""

    def test_get_presigned_url_handles_invalid_s3_format(self):
        """Test that invalid S3 URI format is returned as-is."""
        from backend.app.services.s3_service import get_presigned_url

        # S3 URI without key
        invalid_uri = "s3://bucket-only"
        result = get_presigned_url(invalid_uri)
        assert result == invalid_uri
