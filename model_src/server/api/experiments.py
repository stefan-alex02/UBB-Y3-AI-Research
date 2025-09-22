import io
import logging
from pathlib import PurePath
from typing import List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request as FastAPIRequest
from fastapi.responses import StreamingResponse

from .utils import RunExperimentRequest, ExperimentRunResponse, ArtifactNode
from ..core.config import APP_LOGGER_NAME
from ..services import experiment_service
from ..services.experiment_service import get_artifact_type_from_filename

logger = logging.getLogger(APP_LOGGER_NAME)

router = APIRouter()


@router.post("/run", response_model=ExperimentRunResponse)
async def run_experiment_endpoint(
        request: RunExperimentRequest,
        background_tasks: BackgroundTasks,
        fast_api_request: FastAPIRequest
):
    try:
        response_data = await experiment_service.start_experiment(fast_api_request, request, background_tasks)
        return response_data
    except ValueError as ve:
        logger.error(f"Validation error running experiment {request.experiment_run_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error submitting experiment {request.experiment_run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit experiment for execution.")


@router.get("/{dataset_name}/{model_type}/{experiment_run_id}/artifacts/list", response_model=List[ArtifactNode])
async def list_experiment_artifacts_endpoint(
        dataset_name: str, model_type: str, experiment_run_id: str,
        fast_api_request: FastAPIRequest,
        path: Optional[str] = ""
):
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    try:
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
async def get_experiment_artifact_content_endpoint(
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

        media_type = "application/octet-stream"
        file_type_from_name = get_artifact_type_from_filename(artifact_path)

        if file_type_from_name == "json":
            media_type = "application/json"
        elif file_type_from_name == "image":
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
                media_type = "application/octet-stream"
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

@router.delete("/{dataset_name}/{model_type}/{experiment_run_id}")
async def delete_experiment_api(
        dataset_name: str,
        model_type: str,
        experiment_run_id: str,
        fast_api_request: FastAPIRequest
):
    logger.info(
        f"Received request to delete experiment: {experiment_run_id} (dataset: {dataset_name}, model: {model_type})")
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    experiment_prefix = str((PurePath("experiments") / dataset_name / model_type / experiment_run_id).as_posix()) + "/"

    success = artifact_repo.delete_objects_by_prefix(experiment_prefix)
    if success:
        return {"message": f"Experiment artifacts for {experiment_run_id} deletion process initiated."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to delete experiment artifacts for {experiment_run_id}.")
