import io
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request as FastAPIRequest
from fastapi.responses import StreamingResponse

from .utils import RunExperimentRequest, ExperimentRunResponse, ArtifactNode
from ..core.config import APP_LOGGER_NAME  # Import the consistent name
from ..services import experiment_service
from ..services.experiment_service import get_artifact_type_from_filename


logger = logging.getLogger(APP_LOGGER_NAME) # Use the same name

router = APIRouter()


@router.post("/run", response_model=ExperimentRunResponse)
async def run_experiment_endpoint(
        request: RunExperimentRequest,
        background_tasks: BackgroundTasks,  # FastAPI's built-in
        fast_api_request: FastAPIRequest  # To access app.state
):
    try:
        # The service function now handles calling Java to update status to RUNNING
        # and the background task itself handles COMPLETED/FAILED updates.
        response_data = await experiment_service.start_experiment(fast_api_request, request, background_tasks)
        return response_data
    except ValueError as ve:  # e.g., dataset not found
        logger.error(f"Validation error running experiment {request.experiment_run_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error submitting experiment {request.experiment_run_id}: {e}", exc_info=True)
        # Optionally call Java to mark as FAILED if submission itself fails critically
        # await experiment_service.update_experiment_status_in_java(request.experiment_run_id, "FAILED", end_time=True)
        raise HTTPException(status_code=500, detail="Failed to submit experiment for execution.")


@router.get("/{dataset_name}/{model_type}/{experiment_run_id}/artifacts/list", response_model=List[ArtifactNode])
async def list_experiment_artifacts_endpoint(
        dataset_name: str, model_type: str, experiment_run_id: str,
        fast_api_request: FastAPIRequest,
        path: Optional[str] = ""  # Query parameter for sub-path, defaults to root
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    try:
        # Sanitize path: remove leading slashes, ensure it's a relative path concept
        clean_sub_path = path.lstrip('/').lstrip('\\') if path else ""
        logger.info(f"Listing artifacts for experiment {experiment_run_id}, sub-path: '{clean_sub_path}'")

        nodes = experiment_service.list_artifacts_for_experiment(
            artifact_repo, dataset_name, model_type, experiment_run_id, clean_sub_path
        )
        return nodes
    except Exception as e:
        logger.error(f"Error listing artifacts for {experiment_run_id} at path '{path}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list experiment artifacts: {str(e)}")


@router.get("/{dataset_name}/{model_type}/{experiment_run_id}/artifacts/content/{artifact_path:path}")
async def get_experiment_artifact_content_endpoint(  # Renamed for clarity
        dataset_name: str, model_type: str, experiment_run_id: str, artifact_path: str,
        fast_api_request: FastAPIRequest
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    logger.info(
        f"Fetching artifact content: experiments/{dataset_name}/{model_type}/{experiment_run_id}/{artifact_path}")
    try:
        file_bytes = experiment_service.get_experiment_artifact_content_bytes(
            artifact_repo, dataset_name, model_type, experiment_run_id, artifact_path
        )
        if file_bytes is None:
            raise HTTPException(status_code=404, detail="Artifact content not found.")

        # Determine media type based on artifact_path extension
        media_type = "application/octet-stream"
        file_type_from_name = get_artifact_type_from_filename(artifact_path)  # Use your helper

        if file_type_from_name == "json":
            media_type = "application/json"
        elif file_type_from_name == "image":  # Generic image
            ext = artifact_path.split('.')[-1].lower()
            if ext == "png":
                media_type = "image/png"
            elif ext in ["jpg", "jpeg"]:
                media_type = "image/jpeg"
            elif ext == "gif":
                media_type = "image/gif"
            elif ext == "svg":
                media_type = "image/svg+xml"
            else:
                media_type = "application/octet-stream"  # Fallback for unknown image types
        elif file_type_from_name == "log":
            media_type = "text/plain; charset=utf-8"
        elif file_type_from_name == "csv":
            media_type = "text/csv; charset=utf-8"

        return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching artifact content for .../{artifact_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch artifact content: {str(e)}")

# TODO: Add DELETE /experiments/{dataset_name}/{model_type}/{experiment_run_id}
# This would call a service function that lists ALL objects (recursively) under the experiment's base prefix
# and then deletes them one by one or using a bulk delete if the S3 client supports it.
