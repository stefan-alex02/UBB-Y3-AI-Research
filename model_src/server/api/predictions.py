from fastapi import APIRouter, Depends, HTTPException, Request as FastAPIRequest
from typing import List, Optional
import logging
from pathlib import PurePath
import io

from starlette.responses import StreamingResponse

from ..services import prediction_service
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
        predictions = await prediction_service.run_prediction(fast_api_request, config)
        return PredictionRunResponse(predictions=predictions, message="Predictions completed successfully.")
    except HTTPException as he:  # Re-raise HTTPExceptions from the service
        raise he
    except ValueError as ve:  # Specific value errors from service
        logger.warning(f"Prediction value error for user {config.username}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error running prediction for user {config.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction execution failed: {e}")


@router.get("/{username}/{image_id}/{experiment_id_of_model}/artifacts", response_model=List[ArtifactNode])
async def list_prediction_artifacts_endpoint(
        username: str, image_id: str, experiment_id_of_model: str,
        fast_api_request: FastAPIRequest,  # To access app.state
        prefix: Optional[str] = None  # Optional sub-prefix, e.g., "plots"
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")
    if not isinstance(artifact_repo, MinIORepository):
        raise HTTPException(status_code=501,
                            detail="Artifact listing currently only supported for S3-like repositories.")

    base_prediction_prefix = f"predictions/{username}/{image_id}/{experiment_id_of_model}/"
    current_list_prefix = base_prediction_prefix
    if prefix:
        clean_prefix = prefix.lstrip('/')
        if clean_prefix and not clean_prefix.endswith('/'):
            clean_prefix += '/'
        current_list_prefix = base_prediction_prefix + clean_prefix

    logger.info(f"Listing prediction artifacts for prefix: {current_list_prefix}")
    try:
        listed_content = artifact_repo.list_objects_in_prefix(prefix=current_list_prefix, delimiter='/')
        nodes = []
        for subfolder_key in listed_content.get('subfolders', []):
            relative_folder_path = subfolder_key[len(base_prediction_prefix):].strip('/')
            folder_name = PurePath(subfolder_key.strip('/')).name
            nodes.append(ArtifactNode(name=folder_name, path=relative_folder_path, type="folder"))
        for object_key in listed_content.get('objects', []):
            relative_file_path = object_key[len(base_prediction_prefix):]
            file_name = PurePath(object_key).name
            nodes.append(ArtifactNode(name=file_name, path=relative_file_path, type="file"))
        return nodes
    except Exception as e:
        logger.error(f"Error listing prediction artifacts for {current_list_prefix}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list prediction artifacts.")


@router.get("/{username}/{image_id}/{experiment_id_of_model}/artifacts/{artifact_path:path}")
async def get_prediction_artifact_endpoint(
        username: str, image_id: str, experiment_id_of_model: str, artifact_path: str,
        fast_api_request: FastAPIRequest  # To access app.state
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    full_artifact_key = f"predictions/{username}/{image_id}/{experiment_id_of_model}/{artifact_path}"
    logger.info(f"Fetching prediction artifact: {full_artifact_key}")
    try:
        file_bytes = artifact_repo.download_file_to_memory(full_artifact_key)
        if file_bytes is None:
            raise HTTPException(status_code=404, detail="Prediction artifact not found.")

        media_type = "application/octet-stream"
        if artifact_path.lower().endswith(".json"):
            media_type = "application/json"
        elif artifact_path.lower().endswith(".png"):
            media_type = "image/png"
        # Add other types as needed

        return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prediction artifact {full_artifact_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch prediction artifact.")

# TODO: DELETE endpoint for predictions (all artifacts for a specific image_id + experiment_id_of_model)
