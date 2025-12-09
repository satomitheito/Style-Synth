from pydantic_settings import BaseSettings
from pathlib import Path

# Get the backend directory (parent of app directory)
BACKEND_DIR = Path(__file__).parent.parent
# Also check project root (parent of backend directory)
PROJECT_ROOT = BACKEND_DIR.parent

# Try backend/.env first, then root/.env
ENV_FILE_PATH = None
if (BACKEND_DIR / ".env").exists():
    ENV_FILE_PATH = BACKEND_DIR / ".env"
elif (PROJECT_ROOT / ".env").exists():
    ENV_FILE_PATH = PROJECT_ROOT / ".env"
else:
    # Default to backend/.env (will create if needed)
    ENV_FILE_PATH = BACKEND_DIR / ".env"

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str
    DB_PORT: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME:str

    # AWS
    AWS_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str

    #Buckets
    S3_BUCKET_DOCUMENTS: str
    S3_BUCKET_IMAGES: str

    class Config:
        env_file = str(ENV_FILE_PATH)
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
print(f"DEBUG: Loading .env from: {ENV_FILE_PATH}")
print(f"DEBUG: .env file exists: {ENV_FILE_PATH.exists()}")
print("DEBUG SETTINGS:", settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY)
