import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request as FastAPIRequest
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional
import io

from ..services import experiment_service
from .utils import RunExperimentRequest, ExperimentRunResponse, ArtifactNode
# from app.main import artifact_repo_instance # Avoid global
from ..persistence import MinIORepository  # For type checking

import logging
from ..core.config import APP_LOGGER_NAME # Import the consistent name

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


@router.get("/{dataset_name}/{model_type}/{experiment_run_id}/artifacts", response_model=List[ArtifactNode])
async def list_experiment_artifacts_endpoint(
        dataset_name: str, model_type: str, experiment_run_id: str,
        fast_api_request: FastAPIRequest,  # To access app.state
        prefix: Optional[str] = None  # Optional sub-prefix within the experiment folder
):
    """
    Lists artifacts (files and subfolders) for a given experiment run.
    The base prefix for the experiment is constructed from path params.
    An optional 'prefix' query param can specify a subfolder within the experiment.
    e.g., /experiments/CCSN/pvit/exp123/artifacts?prefix=single_train_0/plots
    """
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")
    if not isinstance(artifact_repo, MinIORepository):  # Listing is well-defined for MinIO like repos
        # For LocalFileSystemRepository, you'd need to walk the directory.
        raise HTTPException(status_code=501,
                            detail="Artifact listing currently only supported for S3-like repositories.")

    base_experiment_prefix = f"experiments/{dataset_name}/{model_type}/{experiment_run_id}/"
    current_list_prefix = base_experiment_prefix
    if prefix:
        # Sanitize prefix: remove leading slashes, ensure it ends with a slash if it's meant to be a folder
        clean_prefix = prefix.lstrip('/')
        if clean_prefix and not clean_prefix.endswith('/'):
            clean_prefix += '/'
        current_list_prefix = base_experiment_prefix + clean_prefix

    logger.info(f"Listing artifacts for experiment prefix: {current_list_prefix}")
    try:
        # MinIORepository.list_objects_in_prefix should return {'objects': [...], 'subfolders': [...]}
        listed_content = artifact_repo.list_objects_in_prefix(prefix=current_list_prefix, delimiter='/')

        nodes = []
        # Add subfolders
        for subfolder_key in listed_content.get('subfolders', []):
            # subfolder_key is like "experiments/.../run_id/subfolder_path/"
            # We want the name relative to current_list_prefix
            relative_folder_path = subfolder_key[len(base_experiment_prefix):].strip('/')
            folder_name = PurePath(subfolder_key.strip('/')).name
            nodes.append(ArtifactNode(name=folder_name, path=relative_folder_path, type="folder"))

        # Add objects (files)
        for object_key in listed_content.get('objects', []):
            # object_key is like "experiments/.../run_id/path/to/file.txt"
            # We want the name relative to base_experiment_prefix
            relative_file_path = object_key[len(base_experiment_prefix):]
            file_name = PurePath(object_key).name
            nodes.append(ArtifactNode(name=file_name, path=relative_file_path, type="file"))

        return nodes
    except Exception as e:
        logger.error(f"Error listing artifacts for {current_list_prefix}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list experiment artifacts.")


@router.get("/{dataset_name}/{model_type}/{experiment_run_id}/artifacts/{artifact_path:path}")
async def get_experiment_artifact_endpoint(
        dataset_name: str, model_type: str, experiment_run_id: str, artifact_path: str,
        fast_api_request: FastAPIRequest  # To access app.state
):
    """
    Fetches a specific artifact file for an experiment.
    artifact_path is relative to the experiment's run_id folder.
    e.g., single_train_0/plots/learning_curves.png OR executor_run_log.log
    """
    artifact_repo = fast_api_request.app.state.artifact_repo
    if not artifact_repo:
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    full_artifact_key = f"experiments/{dataset_name}/{model_type}/{experiment_run_id}/{artifact_path}"
    logger.info(f"Fetching artifact: {full_artifact_key}")

    try:
        # Assuming download_file_to_memory exists in your ArtifactRepository interface
        file_bytes = artifact_repo.download_file_to_memory(full_artifact_key)
        if file_bytes is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        media_type = "application/octet-stream"  # Default
        if artifact_path.lower().endswith(".json"):
            media_type = "application/json"
        elif artifact_path.lower().endswith(".png"):
            media_type = "image/png"
        elif artifact_path.lower().endswith(".jpg") or artifact_path.lower().endswith(".jpeg"):
            media_type = "image/jpeg"
        elif artifact_path.lower().endswith(".log") or artifact_path.lower().endswith(
                ".txt") or artifact_path.lower().endswith(".csv"):
            media_type = "text/plain"

        return StreamingResponse(io.BytesIO(file_bytes), media_type=media_type)
    except HTTPException:
        raise  # Re-raise 404 if that was the case
    except Exception as e:
        logger.error(f"Error fetching artifact {full_artifact_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch artifact.")

# TODO: Add DELETE /experiments/{...}/{experiment_run_id}
# This would need to list all objects under the experiment prefix and delete them.
# Be careful with MinIO rate limits if there are many objects.
