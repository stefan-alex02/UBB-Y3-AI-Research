import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import experiments, images, predictions
from .core import task_queue
from .core.config import settings, APP_LOGGER_NAME
from .ml.logger_utils import setup_logger as setup_pipeline_logger, EnhancedFormatter, \
    logger_name_global
from .persistence import load_minio_repository, load_file_repository, ArtifactRepository

fastapi_app_logger = logging.getLogger(APP_LOGGER_NAME)
fastapi_app_logger.setLevel(settings.LOG_LEVEL.upper())
app_console_handler = logging.StreamHandler()
app_console_handler.setFormatter(EnhancedFormatter(use_colors=True))
fastapi_app_logger.addHandler(app_console_handler)
fastapi_app_logger.propagate = False


# Repository Initialization
artifact_repo_instance: ArtifactRepository = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact_repo_instance
    fastapi_app_logger.info("FastAPI application startup sequence initiated...")
    try:
        artifact_repo_instance = load_minio_repository(
            logger=fastapi_app_logger,
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

    setup_pipeline_logger(
        name=logger_name_global,
        log_dir=settings.APP_LOG_FILE_PATH.parent if settings.LOG_TO_FILE_PIPELINE else None,
        log_filename="pipeline_main.log" if settings.LOG_TO_FILE_PIPELINE else "pipeline_console.log",
        level=settings.LOG_LEVEL_PIPELINE.upper()
    )


    app.state.artifact_repo = artifact_repo_instance
    fastapi_app_logger.info("FastAPI lifespan startup complete. Artifact repo attached to app state.")
    task_queue.ensure_processor_is_running()
    yield
    fastapi_app_logger.info("FastAPI application shutdown sequence initiated...")
    fastapi_app_logger.info("FastAPI application shutdown sequence initiated...")
    fastapi_app_logger.info("FastAPI lifespan shutdown complete.")


app = FastAPI(lifespan=lifespan, title="Cloud Image Classification ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    fastapi_app_logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


app.include_router(experiments.router, prefix="/experiments")
app.include_router(images.router, prefix="/images")
app.include_router(predictions.router, prefix="/predictions")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

from .ml.logger_utils import logger as pipeline_logger

@app.get("/test-pipeline-log")
async def test_log():
    pipeline_logger.info("This is a test message from the pipeline logger via an endpoint.")
    fastapi_app_logger.info("This is a test message from the FastAPI app logger via an endpoint.")
    return {"message": "Logs sent"}
