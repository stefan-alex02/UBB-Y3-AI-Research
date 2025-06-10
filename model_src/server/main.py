import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import experiments, images, predictions
from .core import task_queue
from .core.config import settings, APP_LOGGER_NAME
from .ml.logger_utils import setup_logger as setup_pipeline_logger, EnhancedFormatter, \
    logger_name_global  # Import your logger_name_global
from .persistence import load_minio_repository, load_file_repository, ArtifactRepository  # Your existing imports

# --- FastAPI Application Logger Setup ---
# This logger is for FastAPI/Uvicorn specific messages IF you want to customize them,
# or for general application-level logs separate from the pipeline.
fastapi_app_logger = logging.getLogger(APP_LOGGER_NAME)
fastapi_app_logger.setLevel(settings.LOG_LEVEL.upper())
app_console_handler = logging.StreamHandler()
app_console_handler.setFormatter(EnhancedFormatter(use_colors=True)) # Your custom formatter
fastapi_app_logger.addHandler(app_console_handler)
fastapi_app_logger.propagate = False # Often good to prevent double logging if root is configured

# The uvicorn.error and uvicorn.access loggers will continue to use their own default formatters
# unless you explicitly configure them with different formatters.

# --- Repository Initialization ---
artifact_repo_instance: ArtifactRepository = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact_repo_instance
    fastapi_app_logger.info("FastAPI application startup sequence initiated...") # Uses your new logger
    try:
        artifact_repo_instance = load_minio_repository(
            logger=fastapi_app_logger, # Pass this app logger for repo init messages
            bucket_name=settings.MINIO_BUCKET_NAME,
        )
        fastapi_app_logger.info(f"MinIORepository initialized for bucket: {settings.MINIO_BUCKET_NAME}")
    except Exception as e:
        fastapi_app_logger.warning(f"Failed to initialize MinIORepository: {e}. Falling back to LocalFileSystemRepository.")
        artifact_repo_instance = load_file_repository(
            logger=fastapi_app_logger,
            repo_base_path=settings.LOCAL_STORAGE_BASE_PATH
        )
        fastapi_app_logger.info(f"LocalFileSystemRepository initialized at: {settings.LOCAL_STORAGE_BASE_PATH}")

    # Setup the PIPELINE's logger (ImgClassPipe)
    # This is YOUR main ML pipeline logger, configured by your setup_pipeline_logger
    setup_pipeline_logger(
        name=logger_name_global, # e.g., "ImgClassPipe" from your ml.config or logger_utils
        log_dir=settings.APP_LOG_FILE_PATH.parent if settings.LOG_TO_FILE_PIPELINE else None, # Use a distinct config
        log_filename="pipeline_main.log" if settings.LOG_TO_FILE_PIPELINE else "pipeline_console.log",
        level=settings.LOG_LEVEL_PIPELINE.upper() # Use a distinct config
    )
    # Example additions to your core/config.py:
    # LOG_TO_FILE_PIPELINE: bool = True
    # LOG_LEVEL_PIPELINE: str = "INFO"


    app.state.artifact_repo = artifact_repo_instance
    fastapi_app_logger.info("FastAPI lifespan startup complete. Artifact repo attached to app state.")
    # Start the experiment queue processor
    task_queue.ensure_processor_is_running()
    yield
    fastapi_app_logger.info("FastAPI application shutdown sequence initiated...")
    fastapi_app_logger.info("FastAPI application shutdown sequence initiated...")
    # Any cleanup for artifact_repo_instance if needed
    fastapi_app_logger.info("FastAPI lifespan shutdown complete.")


app = FastAPI(lifespan=lifespan, title="Cloud Image Classification ML API")

# CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify your frontend URL: e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler (optional, but good for consistent error responses)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    fastapi_app_logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


# Include routers
# Example of protecting a router with API key
# app.include_router(experiments.router, prefix="/experiments", dependencies=[Depends(get_api_key)])
app.include_router(experiments.router, prefix="/experiments") # Unprotected for now for easier dev
app.include_router(images.router, prefix="/images")
app.include_router(predictions.router, prefix="/predictions")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Example usage of the pipeline logger in an endpoint:
from .ml.logger_utils import logger as pipeline_logger # Import your specific pipeline logger

@app.get("/test-pipeline-log")
async def test_log():
    pipeline_logger.info("This is a test message from the pipeline logger via an endpoint.")
    fastapi_app_logger.info("This is a test message from the FastAPI app logger via an endpoint.")
    return {"message": "Logs sent"}
