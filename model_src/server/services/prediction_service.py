import logging
from pathlib import Path, PurePath
import io
from typing import List

from ..api.utils import RunPredictionRequest, SinglePredictionResult
from ..core.config import settings
from ..ml.pipeline import ClassificationPipeline # Your existing import
from ..ml.architectures import ModelType
from ..ml.dataset_utils import ImageDatasetHandler # For default transforms if needed
# from app.main import artifact_repo_instance # Avoid global
from fastapi import Request as FastAPIRequest, HTTPException

logger = logging.getLogger("fastapi_app")
pipeline_logger = logging.getLogger("ImgClassPipe")


async def run_prediction(
    request_fast_api: FastAPIRequest, # To access app.state.artifact_repo
    config: RunPredictionRequest
) -> List[SinglePredictionResult]:
    logger.info(f"Received prediction request for user {config.username} using model from {config.experiment_run_id_of_model}")

    artifact_repo_from_state = request_fast_api.app.state.artifact_repo
    if not artifact_repo_from_state:
        logger.error("Artifact repository not available for prediction service.")
        raise RuntimeError("Artifact repository not configured.")

    # Construct the full model path key for MinIO/local
    # Path is: experiments/{dataset_name_of_model}/{model_type_of_model}/{exp_id_producer}/{relative_model_path}.pt
    model_prefix = PurePath(
        "experiments",
        config.model_load_details.dataset_name_of_model,
        config.model_load_details.model_type_of_model,
        config.model_load_details.experiment_run_id_of_model_producer
    )
    full_model_path_or_key = str(model_prefix / config.model_load_details.relative_model_path_in_experiment)
    logger.info(f"Attempting to load model from derived path/key: {full_model_path_or_key}")


    # Instantiate a lean ClassificationPipeline for prediction
    # It needs a dataset_path for ImageDatasetHandler init, even if not used for loading train data.
    # This path could be a dummy or a common one if not directly relevant for prediction transforms.
    # However, ImageDatasetHandler IS used by predict_images to get class names and eval_transform.
    # So, we need to infer the original dataset path the model was trained on.
    # Let's assume dataset_name_of_model implies the dataset structure under LOCAL_STORAGE_BASE_PATH / "datasets_for_training"
    dataset_path_for_model_context = settings.LOCAL_STORAGE_BASE_PATH / "datasets_for_training" / config.model_load_details.dataset_name_of_model
    if not dataset_path_for_model_context.exists():
        logger.error(f"Dataset context path {dataset_path_for_model_context} for model loading not found.")
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
        logger.error(f"Model file or its config not found for prediction: {e}")
        raise HTTPException(status_code=404, detail=f"Model or required configuration not found: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during prediction service: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.")
