from pydantic_settings import BaseSettings
from pathlib import Path

APP_LOGGER_NAME = "cloud_classifier_app"

class Settings(BaseSettings):
    MINIO_ENDPOINT_URL: str = "http://127.0.0.1:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "m1n104dm1n"
    MINIO_BUCKET_NAME: str = "clouds"
    MINIO_SECURE: bool = False # True if MinIO endpoint is HTTPS

    JAVA_INTERNAL_API_URL: str = "http://localhost:8080/api/internal" # Example
    INTERNAL_API_KEY: str = "your-strong-internal-api-key" # For Python -> Java internal calls

    # For local file repo fallback if MinIO is not configured or for images/predictions
    # if not using MinIO for those.
    LOCAL_STORAGE_BASE_PATH: Path = Path("model_src/data/") # Default local storage

    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True # This is for the FastAPI app log, PipelineExecutor has its own
    APP_LOG_FILE_PATH: Path = Path("logs/fastapi_app.log")

    LOG_TO_FILE_PIPELINE: bool = True  # Controls if "ImgClassPipe" (pipeline logger) writes to a file
    LOG_LEVEL_PIPELINE: str = "INFO"  # Log level for "ImgClassPipe"
    PIPELINE_LOG_FILE_PATH: Path = Path("logs/pipeline_main.log")  # Default path for "ImgClassPipe" log file

    # Executor defaults that might come from config
    DEFAULT_IMG_SIZE_H: int = 224
    DEFAULT_IMG_SIZE_W: int = 224
    DEFAULT_MAX_EPOCHS: int = 10 # Default for quick experiments if not specified

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

# Ensure log directory exists for FastAPI app log
if settings.LOG_TO_FILE:
    settings.APP_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Ensure local storage base path exists
settings.LOCAL_STORAGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
