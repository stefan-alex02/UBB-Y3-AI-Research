from fastapi import APIRouter, Depends, HTTPException, Request as FastAPIRequest
from typing import List, Optional
import logging
from pathlib import PurePath
import io

from starlette.responses import StreamingResponse

from ..services import prediction_service as service
from .utils import RunPredictionRequest, PredictionRunResponse, ArtifactNode
# from app.main import artifact_repo_instance # Avoid global
from ..persistence import MinIORepository

import logging
from ..core.config import APP_LOGGER_NAME # Import the consistent name

logger = logging.getLogger(APP_LOGGER_NAME) # Use the same name

router = APIRouter()


@router.post("/run", response_model=PredictionRunResponse)
async def run_prediction_endpoint(
        config: RunPredictionRequest,
        fast_api_request: FastAPIRequest  # To access app.state
):
    try:
        predictions = await service.run_prediction(fast_api_request, config)
        return PredictionRunResponse(predictions=predictions, message="Predictions completed successfully.")
    except HTTPException as he:  # Re-raise HTTPExceptions from the service
        raise he
    except ValueError as ve:  # Specific value errors from service
        logger.warning(f"Prediction value error for user {config.username}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error running prediction for user {config.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction execution failed: {e}")


@router.get("/{username}/{image_id}/{experiment_id_of_model}/artifacts/list", response_model=List[ArtifactNode])
async def list_prediction_artifacts_api(
        username: str, image_id: str, experiment_id_of_model: str,
        fast_api_request: FastAPIRequest,
        path: Optional[str] = ""  # Query parameter for sub-path, e.g., "plots"
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")
    try:
        clean_sub_path = path.lstrip('/').lstrip('\\') if path else ""
        logger.info(
            f"Listing prediction artifacts for user {username}, image {image_id}, model_exp {experiment_id_of_model}, sub-path: '{clean_sub_path}'")
        nodes = service.list_artifacts_for_prediction(
            artifact_repo, username, image_id, experiment_id_of_model, clean_sub_path
        )
        return nodes
    except Exception as e:
        logger.error(f"Error listing prediction artifacts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list prediction artifacts: {str(e)}")


@router.get("/{username}/{image_id}/{experiment_id_of_model}/artifacts/content/{artifact_path:path}")
async def get_prediction_artifact_content_api(
        username: str, image_id: str, experiment_id_of_model: str, artifact_path: str,
        fast_api_request: FastAPIRequest
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")
    logger.info(
        f"Fetching prediction artifact content: predictions/{username}/{image_id}/{experiment_id_of_model}/{artifact_path}")
    try:
        file_bytes = service.get_prediction_artifact_content_bytes(
            artifact_repo, username, image_id, experiment_id_of_model, artifact_path
        )
        if file_bytes is None:
            raise HTTPException(status_code=404, detail="Prediction artifact content not found.")

        media_type = "application/octet-stream"
        # Use the same get_artifact_type_from_filename helper from prediction_service
        file_type_from_name = service.get_artifact_type_from_filename(artifact_path)

        if file_type_from_name == "json":
            media_type = "application/json; charset=utf-8"
        elif file_type_from_name == "image":
            ext = artifact_path.split('.')[-1].lower()
            if ext == "png":
                media_type = "image/png"
            elif ext in ["jpg", "jpeg"]:
                media_type = "image/jpeg"
            else:
                media_type = "application/octet-stream"
        elif file_type_from_name == "log":
            media_type = "text/plain; charset=utf-8"
        elif file_type_from_name == "csv":
            media_type = "text/csv; charset=utf-8"

        return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prediction artifact content for .../{artifact_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction artifact content: {str(e)}")

# TODO: DELETE endpoint for predictions (all artifacts for a specific image_id + experiment_id_of_model)
