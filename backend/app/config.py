"""
Application configuration using Pydantic BaseSettings.

Loads values automatically from:
- environment variables
- .env file (if present)

Central location for:
- Database settings
- AWS S3 settings
- File storage
- Model configuration
"""

from pydantic import BaseSettings


class Settings(BaseSettings):
    # ----------------------------------------------------
    # ENVIRONMENT
    # ----------------------------------------------------
    environment: str = "development"

    # ----------------------------------------------------
    # DATABASE (asyncpg)
    # Example: postgresql://user:password@host:5432/dbname
    # ----------------------------------------------------
    database_url: str = "postgresql://postgres:postgres@localhost:5432/appdb"

    # ----------------------------------------------------
    # AWS S3 CONFIGURATION
    # These must match your AWS setup.
    # ----------------------------------------------------
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Buckets
    s3_bucket_documents: str = "myapp-documents"
    s3_bucket_images: str = "myapp-images"
    s3_bucket_models: str = "myapp-models"

    # ----------------------------------------------------
    # MODEL / EMBEDDINGS CONFIG
    # ----------------------------------------------------
    embedding_path: str = "/mnt/data/fashion_mnist_resnet50_embeddings.npy"
    label_path: str = "/mnt/data/fashion_mnist_labels.npy"
    embedding_dim: int = 2048   # Adjust if needed

    # ----------------------------------------------------
    # FILE UPLOAD LIMITS
    # ----------------------------------------------------
    max_upload_mb: int = 20

    # ----------------------------------------------------
    # SECURITY
    # ----------------------------------------------------
    secret_key: str = "CHANGE_ME"

    class Config:
        env_file = ".env"                 # Load from .env automatically
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
