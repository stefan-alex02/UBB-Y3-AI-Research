import io
import logging
from pathlib import PurePath
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request as FastAPIRequest
from starlette.responses import StreamingResponse

from .utils import RunPredictionRequest, PredictionRunResponse, ArtifactNode
from ..core.config import APP_LOGGER_NAME
from ..persistence import ArtifactRepository
from ..services import prediction_service as service

logger = logging.getLogger(APP_LOGGER_NAME) # Use the same name

router = APIRouter()


@router.post("/run", response_model=PredictionRunResponse) # Assuming PredictionRunResponse is your Pydantic model
async def run_prediction_endpoint(
    config: RunPredictionRequest, # This is your Pydantic model for the request body
    fast_api_request: FastAPIRequest
):
    try:
        # Call the async wrapper in your service
        predictions_list = await service.run_prediction_async_wrapper(fast_api_request, config)
        return PredictionRunResponse(predictions=predictions_list, message="Predictions completed successfully.")
    except HTTPException as he:
        raise he # Re-raise HTTPExceptions to let FastAPI handle them
    except Exception as e:
        logger.error(f"API Error running prediction for user {config.username}: {e}", exc_info=True)
        # This will be caught by your global exception handler if not an HTTPException
        raise HTTPException(status_code=500, detail=f"Prediction submission failed: {str(e)}")


@router.get("/{username}/{image_id}/{prediction_id}/artifacts/list", response_model=List[ArtifactNode])
async def list_prediction_artifacts_api(
        username: str, image_id: str, prediction_id: str,
        fast_api_request: FastAPIRequest,
        path: Optional[str] = ""  # Query parameter for sub-path, e.g., "plots"
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")
    try:
        clean_sub_path = path.lstrip('/').lstrip('\\') if path else ""
        logger.info(
            f"Listing prediction artifacts for user {username}, image {image_id}, model_exp {prediction_id}, sub-path: '{clean_sub_path}'")
        nodes = service.list_artifacts_for_prediction(
            artifact_repo, username, image_id, prediction_id, clean_sub_path
        )
        return nodes
    except Exception as e:
        logger.error(f"Error listing prediction artifacts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list prediction artifacts: {str(e)}")


@router.get("/{username}/{image_id}/{prediction_id}/artifacts/content/{artifact_path:path}")
async def get_prediction_artifact_content_api(
        username: str, image_id: str, prediction_id: str, artifact_path: str,
        fast_api_request: FastAPIRequest
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")
    logger.info(
        f"Fetching prediction artifact content: predictions/{username}/{image_id}/{prediction_id}/{artifact_path}")
    try:
        file_bytes = service.get_prediction_artifact_content_bytes(
            artifact_repo, username, image_id, prediction_id, artifact_path
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


@router.delete("/{username}/{image_id}/{prediction_id}")
async def delete_prediction_api(
        username: str,
        image_id: str,
        prediction_id: str,
        fast_api_request: FastAPIRequest
):
    logger.info(
        f"Received request to delete prediction artifacts for user {username}, image {image_id}, model_exp {prediction_id}")
    artifact_repo: ArtifactRepository = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    prediction_prefix = str((PurePath("predictions") / username / image_id / prediction_id).as_posix()) + "/"

    success = artifact_repo.delete_objects_by_prefix(prediction_prefix)
    if success:
        return {
            "message": f"Prediction artifacts for image {image_id}, model_exp {prediction_id} deletion process initiated."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete prediction artifacts.")
