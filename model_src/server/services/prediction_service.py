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


async def run_prediction(
    request_fast_api: FastAPIRequest, # To access app.state.artifact_repo
    config: RunPredictionRequest
) -> List[SinglePredictionResult]:
    fastapi_app_logger.info(f"Received prediction request for user {config.username} using model from {config.experiment_run_id_of_model}")

    artifact_repo_from_state = request_fast_api.app.state.artifact_repo
    if not artifact_repo_from_state:
        fastapi_app_logger.error("Artifact repository not available for prediction service.")
        raise RuntimeError("Artifact repository not configured.")

    # Construct the full model path key for MinIO/local
    # Path is: experiments/{dataset_name_of_model}/{model_type_of_model}/{exp_id_producer}/{relative_model_path}.pt
    model_prefix = PurePath(
        DATASET_DICT[config.model_load_details.dataset_name_of_model],
        config.model_load_details.model_type_of_model,
        config.model_load_details.experiment_run_id_of_model_producer
    )
    full_model_path_or_key = str(model_prefix / config.model_load_details.relative_model_path_in_experiment)
    fastapi_app_logger.info(f"Attempting to load model from derived path/key: {full_model_path_or_key}")


    # Instantiate a lean ClassificationPipeline for prediction
    # It needs a dataset_path for ImageDatasetHandler init, even if not used for loading train data.
    # This path could be a dummy or a common one if not directly relevant for prediction transforms.
    # However, ImageDatasetHandler IS used by predict_images to get class names and eval_transform.
    # So, we need to infer the original dataset path the model was trained on.
    # Let's assume dataset_name_of_model implies the dataset structure under LOCAL_STORAGE_BASE_PATH / "datasets_for_training"
    dataset_path_for_model_context = settings.LOCAL_STORAGE_BASE_PATH / DATASET_DICT[config.model_load_details.dataset_name_of_model]
    if not dataset_path_for_model_context.exists():
        fastapi_app_logger.error(f"Dataset context path {dataset_path_for_model_context} for model loading not found.")
        raise ValueError(f"Cannot initialize pipeline context for dataset {config.model_load_details.dataset_name_of_model}")

    try:
        # Create a temporary pipeline instance for this prediction
        # It doesn't need a full executor context for artifact saving during prediction itself,
        # as predict_images will handle its own artifact paths.
        pipeline_for_prediction = ClassificationPipeline(
            dataset_path=dataset_path_for_model_context, # Context for class names, eval transforms
            model_type=ModelType(config.model_load_details.model_type_of_model), # Ensure it's ModelType enum
            artifact_repository=artifact_repo_from_state,
            experiment_base_key_prefix=None, # Not strictly needed for predict_images' own saving
            img_size=(settings.DEFAULT_IMG_SIZE_H, settings.DEFAULT_IMG_SIZE_W) # Use default or pass from config
        )

        pipeline_for_prediction.load_model(full_model_path_or_key) # This loads .pt and _arch_config.json

        # Call the modified predict_images
        predictions_api_format = pipeline_for_prediction.predict_images(
            image_id_format_pairs=[(p.image_id, p.image_format) for p in config.image_id_format_pairs],
            experiment_run_id_of_model=config.experiment_run_id_of_model,
            username=config.username,
            persist_prediction_artifacts=True, # Always persist for predictions
            plot_level=2 if config.generate_lime else 1, # Show plots if LIME, else just save
            generate_lime_explanations=config.generate_lime,
            lime_num_features_to_show_plot=config.lime_num_features if config.generate_lime else 0,
            lime_num_samples_for_explainer=config.lime_num_samples if config.generate_lime else 0,
            prob_plot_top_k=config.prob_plot_top_k
        )
        return predictions_api_format

    except FileNotFoundError as e:
        fastapi_app_logger.error(f"Model file or its config not found for prediction: {e}")
        raise HTTPException(status_code=404, detail=f"Model or required configuration not found: {e}")
    except RuntimeError as e:
        fastapi_app_logger.error(f"Runtime error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    except Exception as e:
        fastapi_app_logger.error(f"Unexpected error during prediction service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")


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
    current_scan_prefix = str(base_prediction_prefix / sub_path).strip("/")
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
    full_artifact_key = str(PurePath("predictions", username, str(image_id), experiment_id_of_model, artifact_relative_path))
    fastapi_app_logger.info(f"Fetching content for prediction artifact: {full_artifact_key}")
    return artifact_repo.download_file_to_memory(full_artifact_key)
