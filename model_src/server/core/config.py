from pathlib import Path

from pydantic_settings import BaseSettings

APP_LOGGER_NAME = "cloud_classifier_app"

class Settings(BaseSettings):
    MINIO_ENDPOINT_URL: str = "http://127.0.0.1:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "m1n104dm1n"
    MINIO_BUCKET_NAME: str = "clouds"
    MINIO_SECURE: bool = False

    JAVA_INTERNAL_API_URL: str = "http://localhost:8080/api/internal"
    INTERNAL_API_KEY: str = "your-strong-internal-api-key"

    LOCAL_STORAGE_BASE_PATH: Path = Path("model_src/data/")

    # Logging configuration
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    APP_LOG_FILE_PATH: Path = Path("logs/fastapi_app.log")

    LOG_TO_FILE_PIPELINE: bool = True
    LOG_LEVEL_PIPELINE: str = "INFO"
    PIPELINE_LOG_FILE_PATH: Path = Path("logs/pipeline_main.log")

    # Executor defaults
    DEFAULT_IMG_SIZE_H: int = 224
    DEFAULT_IMG_SIZE_W: int = 224
    DEFAULT_MAX_EPOCHS: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()

if settings.LOG_TO_FILE:
    settings.APP_LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

settings.LOCAL_STORAGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
