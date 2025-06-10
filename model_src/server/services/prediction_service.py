import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path, PurePath
import io
from typing import List, Optional

from ..api.utils import RunPredictionRequest, SinglePredictionResult, ArtifactNode
from ..core.config import settings, APP_LOGGER_NAME
from ..ml.config import DATASET_DICT
from ..ml.pipeline import ClassificationPipeline # Your existing import
from ..ml.architectures import ModelType
from ..ml.dataset_utils import ImageDatasetHandler # For default transforms if needed
# from app.main import artifact_repo_instance # Avoid global
from fastapi import Request as FastAPIRequest, HTTPException

from ..persistence import ArtifactRepository, LocalFileSystemRepository, MinIORepository

fastapi_app_logger = logging.getLogger(APP_LOGGER_NAME)
pipeline_logger = logging.getLogger("ImgClassPipe")


def _run_prediction_sync(
    username: str,
    image_id_format_pairs: List, # From Pydantic model
    model_load_details_dict: dict, # From Pydantic model, converted to dict
    experiment_run_id_of_model: str,
    generate_lime: bool,
    lime_num_features: int,
    lime_num_samples: int,
    prob_plot_top_k: int,
    artifact_repo, # Pass the repo
    # Pass dataset_path_for_model_context if needed by pipeline init
    dataset_path_for_model_context: Path
):
    fastapi_app_logger.info(f"SYNC: Starting prediction for user {username} using model from {experiment_run_id_of_model}")
    # Instantiate a lean ClassificationPipeline for prediction
    try:
        pipeline_for_prediction = ClassificationPipeline(
            dataset_path=dataset_path_for_model_context,
            model_type=ModelType(model_load_details_dict["model_type_of_model"]),
            artifact_repository=artifact_repo,
            # Ensure img_size is passed if not handled by load_model re-init
            img_size=(settings.DEFAULT_IMG_SIZE_H, settings.DEFAULT_IMG_SIZE_W)
        )

        # Construct full model path key
        model_prefix = PurePath(
            DATASET_DICT[model_load_details_dict["dataset_name_of_model"]],
            model_load_details_dict["model_type_of_model"],
            model_load_details_dict["experiment_run_id_of_model_producer"]
        )
        full_model_path_or_key = str(model_prefix / model_load_details_dict["relative_model_path_in_experiment"])
        pipeline_for_prediction.load_model(full_model_path_or_key)

        predictions_api_format = pipeline_for_prediction.predict_images(
            image_id_format_pairs=[(p.image_id, p.image_format) for p in image_id_format_pairs],
            experiment_run_id_of_model=experiment_run_id_of_model,
            username=username,
            persist_prediction_artifacts=True,
            plot_level=not prob_plot_top_k == 0,
            generate_lime_explanations=generate_lime,
            lime_num_features_to_show_plot=lime_num_features,
            lime_num_samples_for_explainer=lime_num_samples,
            prob_plot_top_k=prob_plot_top_k if prob_plot_top_k else -1,
        )
        fastapi_app_logger.info(f"SYNC: Finished prediction for user {username}")
        return predictions_api_format
    except FileNotFoundError as e:
        fastapi_app_logger.error(f"SYNC: Model file or config not found for prediction: {e}")
        # Re-raise a specific error that the endpoint can catch and convert to HTTPException
        raise HTTPException(status_code=404, detail=f"Model or required configuration not found: {e}")
    except Exception as e:
        fastapi_app_logger.error(f"SYNC: Unexpected error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed internally: {str(e)}")

# The function called by your FastAPI endpoint becomes async and uses run_in_executor
async def run_prediction_async_wrapper(
        request_fast_api: FastAPIRequest,  # To get artifact_repo and other app state
        config: RunPredictionRequest  # Your Pydantic model
):
    fastapi_app_logger.info(f"ASYNC WRAPPER: Received prediction request for user {config.username}")
    artifact_repo_from_state = request_fast_api.app.state.artifact_repo
    if not artifact_repo_from_state:
        fastapi_app_logger.error("Artifact repository not available for prediction service.")
        raise HTTPException(status_code=500, detail="Artifact repository not configured.")

    # Prepare dataset_path_for_model_context (as you did before)
    dataset_path_for_model_context = settings.LOCAL_STORAGE_BASE_PATH / DATASET_DICT[config.model_load_details.dataset_name_of_model]
    if not dataset_path_for_model_context.exists():
        fastapi_app_logger.error(f"Dataset context path {dataset_path_for_model_context} for model loading not found.")
        raise HTTPException(status_code=400,
                            detail=f"Cannot initialize context for dataset {config.model_load_details.dataset_name_of_model}")

    loop = asyncio.get_event_loop()
    try:
        # Convert Pydantic model to dict for easier passing if _run_prediction_sync expects it
        # or pass the Pydantic model directly if _run_prediction_sync can handle it.
        # For simplicity, let's assume it expects primitive types or dicts for now.
        prediction_results = await loop.run_in_executor(
            None,  # Uses default ThreadPoolExecutor
            _run_prediction_sync,  # The synchronous function
            # Positional arguments for _run_prediction_sync:
            config.username,
            config.image_id_format_pairs,  # Pass the list of Pydantic models
            config.model_load_details.model_dump(),  # Convert Pydantic to dict
            config.experiment_run_id_of_model,
            config.generate_lime,
            config.lime_num_features,
            config.lime_num_samples,
            config.prob_plot_top_k,
            artifact_repo_from_state,
            dataset_path_for_model_context
        )
        return prediction_results  # This is List[SinglePredictionResult] from Pydantic
    except HTTPException as he:  # Re-raise HTTPExceptions from _run_prediction_sync
        raise he
    except Exception as e:
        fastapi_app_logger.error(f"ASYNC WRAPPER: Unexpected error running prediction in executor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction execution failed: {str(e)}")


# Re-use or adapt get_artifact_type_from_filename from experiment_service.py
def get_artifact_type_from_filename(filename: str) -> str:
    name_lower = filename.lower()
    if name_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")): return "image"
    if name_lower.endswith(".json"): return "json"
    if name_lower.endswith((".log", ".txt")): return "log"
    if name_lower.endswith(".csv"): return "csv"
    return "file"

def list_artifacts_for_prediction(
    artifact_repo: ArtifactRepository,
    username: str,
    image_id: str, # Image ID (from DB)
    experiment_id_of_model: str, # Experiment ID that produced the model
    sub_path: str = "" # Relative path within the prediction's folder, e.g., "plots"
) -> List[ArtifactNode]:
    # Base path for this specific prediction's artifacts
    base_prediction_prefix = PurePath("predictions", username, str(image_id), experiment_id_of_model)
    current_scan_prefix = str((base_prediction_prefix / sub_path).as_posix()).strip("/")
    if current_scan_prefix and not current_scan_prefix.endswith("/"):
        current_scan_prefix += "/"

    fastapi_app_logger.info(f"Scanning prediction artifacts under prefix: {current_scan_prefix}")
    nodes: List[ArtifactNode] = []

    if isinstance(artifact_repo, MinIORepository):
        listed_content = artifact_repo.list_objects_in_prefix(prefix=current_scan_prefix, delimiter='/')
        for folder_key in listed_content.get('subfolders', []):
            folder_name = PurePath(folder_key.rstrip('/')).name
            relative_path = str(PurePath(folder_key).relative_to(base_prediction_prefix))
            nodes.append(ArtifactNode(name=folder_name, path=relative_path, type="folder"))
        for file_key in listed_content.get('objects', []):
            if file_key == current_scan_prefix and file_key.endswith('/'): continue
            file_name = PurePath(file_key).name
            if not file_name: continue
            relative_path = str(PurePath(file_key).relative_to(base_prediction_prefix))
            file_type = get_artifact_type_from_filename(file_name)
            nodes.append(ArtifactNode(name=file_name, path=relative_path, type=file_type))

    elif isinstance(artifact_repo, LocalFileSystemRepository):
        scan_dir = Path(artifact_repo.base_path) / current_scan_prefix
        if scan_dir.is_dir():
            for item in sorted(scan_dir.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                relative_path = str(item.relative_to(Path(artifact_repo.base_path) / base_prediction_prefix))
                if item.is_dir():
                    nodes.append(ArtifactNode(name=item.name, path=relative_path, type="folder"))
                elif item.is_file():
                    file_type = get_artifact_type_from_filename(item.name)
                    stat_info = item.stat()
                    nodes.append(ArtifactNode(
                        name=item.name, path=relative_path, type=file_type,
                        size=stat_info.st_size,
                        last_modified=datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
                    ))
    else:
        fastapi_app_logger.warning("Artifact listing for predictions only for MinIO/LocalFileSystem.")

    nodes.sort(key=lambda n: (n.type != 'folder', n.name.lower()))
    return nodes


def get_prediction_artifact_content_bytes(
    artifact_repo: ArtifactRepository,
    username: str,
    image_id: str,
    experiment_id_of_model: str,
    artifact_relative_path: str # e.g., "plots/lime.png" or "prediction_details.json"
) -> Optional[bytes]:
    full_artifact_key = str(PurePath("predictions", username, str(image_id), experiment_id_of_model, artifact_relative_path).as_posix())
    fastapi_app_logger.info(f"Fetching content for prediction artifact: {full_artifact_key}")
    return artifact_repo.download_file_to_memory(full_artifact_key)
