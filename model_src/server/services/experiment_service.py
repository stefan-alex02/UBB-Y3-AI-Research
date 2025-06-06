import logging
import httpx # For making calls to Java internal API
from pathlib import Path

from ..api.utils import RunExperimentRequest, ExperimentMethodParams
from ..core.config import settings
from ..core.background_tasks import run_in_background # Or use FastAPI's BackgroundTasks
from ..ml.pipeline import PipelineExecutor
from ..ml.architectures import ModelType # Your existing import
from ..ml.config import AugmentationStrategy # Your existing import
# from app.main import artifact_repo_instance # Avoid global if possible
from fastapi import Request as FastAPIRequest, BackgroundTasks  # To access app.state


logger = logging.getLogger("fastapi_app")
pipeline_logger = logging.getLogger("ImgClassPipe") # Your pipeline's logger


async def update_experiment_status_in_java(experiment_run_id: str, status: str, end_time: bool = False):
    url = f"{settings.JAVA_INTERNAL_API_URL}/experiments/{experiment_run_id}/status"
    payload = {"status": status}
    if end_time:
        from datetime import datetime, timezone
        payload["endTime"] = datetime.now(timezone.utc).isoformat()

    headers = {"X-Internal-API-Key": settings.INTERNAL_API_KEY, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully updated experiment {experiment_run_id} status to {status} in Java.")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error updating status for {experiment_run_id} to {status}: {e.response.text}")
    except Exception as e:
        logger.error(f"Failed to update status for {experiment_run_id} to {status}: {e}")


def _execute_pipeline_task(
    executor_params: dict,
    experiment_run_id: str, # For status updates
    artifact_repo # Pass the initialized repo
):
    """Synchronous function to be run in background thread."""
    try:
        pipeline_logger.info(f"Background task started for experiment: {experiment_run_id}")
        # This function will run in a separate thread, so it can block.
        # Update status to RUNNING (synchronously if possible, or log if http call fails)
        # For simplicity, we'll make the update call synchronous within this blocking task.
        # A more robust way is to have this task only focus on ML and use a separate mechanism for notifications.

        # Sync HTTP call example (if httpx is too complex for background sync thread)
        import requests
        url_running = f"{settings.JAVA_INTERNAL_API_URL}/experiments/{experiment_run_id}/status"
        payload_running = {"status": "RUNNING"}
        headers_running = {"X-Internal-API-Key": settings.INTERNAL_API_KEY, "Content-Type": "application/json"}
        try:
            requests.put(url_running, json=payload_running, headers=headers_running, timeout=10).raise_for_status()
            pipeline_logger.info(f"Updated experiment {experiment_run_id} status to RUNNING in Java (from background task).")
        except Exception as e_status_running:
            pipeline_logger.error(f"Failed to update status to RUNNING for {experiment_run_id} (from background task): {e_status_running}")


        # Initialize and run the executor
        # The executor will setup its own logger file path based on the artifact_repo type
        executor = PipelineExecutor(**executor_params, artifact_repository=artifact_repo)
        final_results = executor.run()
        pipeline_logger.info(f"PipelineExecutor finished for experiment: {experiment_run_id}. Results: {final_results.keys()}")

        # Update status to COMPLETED
        payload_completed = {"status": "COMPLETED", "endTime": True}
        try:
            requests.put(url_running, json=payload_completed, headers=headers_running, timeout=10).raise_for_status()
            pipeline_logger.info(f"Updated experiment {experiment_run_id} status to COMPLETED in Java.")
        except Exception as e_status_completed:
            pipeline_logger.error(f"Failed to update status to COMPLETED for {experiment_run_id}: {e_status_completed}")

    except Exception as e:
        pipeline_logger.critical(f"CRITICAL error during PipelineExecutor run for experiment {experiment_run_id}: {e}", exc_info=True)
        # Update status to FAILED
        try:
            import requests # Ensure requests is imported if not already
            url_failed = f"{settings.JAVA_INTERNAL_API_URL}/experiments/{experiment_run_id}/status"
            payload_failed = {"status": "FAILED", "endTime": True} # endTime: True to set end_time
            headers_failed = {"X-Internal-API-Key": settings.INTERNAL_API_KEY, "Content-Type": "application/json"}
            requests.put(url_failed, json=payload_failed, headers=headers_failed, timeout=10).raise_for_status()
            pipeline_logger.info(f"Updated experiment {experiment_run_id} status to FAILED in Java.")
        except Exception as e_status_failed:
            pipeline_logger.error(f"Failed to update status to FAILED for {experiment_run_id}: {e_status_failed}")
    finally:
        pipeline_logger.info(f"Background task finished for experiment: {experiment_run_id}")


async def start_experiment(
    request: FastAPIRequest, # For accessing app.state.artifact_repo
    config: RunExperimentRequest,
    background_tasks_fastapi: BackgroundTasks # FastAPI's built-in
):
    logger.info(f"Received request to start experiment: {config.experiment_run_id} for dataset {config.dataset_name}")

    dataset_root_path_on_server = settings.LOCAL_STORAGE_BASE_PATH / "datasets_for_training" # Example path
    dataset_path = dataset_root_path_on_server / config.dataset_name
    if not dataset_path.exists():
        logger.error(f"Dataset path {dataset_path} not found on server for experiment {config.experiment_run_id}")
        # Optionally, call Java to mark as FAILED immediately
        await update_experiment_status_in_java(config.experiment_run_id, "FAILED", end_time=True)
        raise ValueError(f"Dataset {config.dataset_name} not found at configured location.")

    methods_sequence_for_executor = []
    for method_param in config.methods_sequence:
        methods_sequence_for_executor.append((method_param.method_name, method_param.params))

    img_h = config.img_size_h if config.img_size_h else settings.DEFAULT_IMG_SIZE_H
    img_w = config.img_size_w if config.img_size_w else settings.DEFAULT_IMG_SIZE_W

    aug_strat_enum = AugmentationStrategy.DEFAULT_STANDARD # Default
    if config.augmentation_strategy_override:
        try:
            aug_strat_enum = AugmentationStrategy(config.augmentation_strategy_override)
        except ValueError:
            logger.warning(f"Invalid augmentation_strategy_override: {config.augmentation_strategy_override}. Using default.")


    executor_params = {
        "dataset_path": dataset_path,
        "model_type": config.model_type,
        "experiment_base_key_prefix": "experiments", # This is the root prefix in MinIO/local
        "methods": methods_sequence_for_executor,
        "img_size": (img_h, img_w),
        "save_main_log_file": True, # Executor will save its detailed log
        # Pass the external_run_id which PipelineExecutor will use for its conceptual_experiment_run_name
        # and thus for its artifact paths within the 'experiments' prefix.
        "conceptual_experiment_run_name": config.experiment_run_id,
        "max_epochs": settings.DEFAULT_MAX_EPOCHS, # Default, can be overridden by method params
        "results_detail_level": 2, # Default detail level for results
        "plot_level": 1,           # Default plot level (save, no show)
        "use_offline_augmented_data": config.offline_augmentation if config.offline_augmentation is not None else False,
        "augmentation_strategy": aug_strat_enum,
        # Any other pipeline-level defaults can be set here
    }
    # Parameters like 'save_model' are typically per-method in the methods_sequence

    # Get the artifact_repo instance from app.state
    artifact_repo_from_state = request.app.state.artifact_repo

    # Use FastAPI's BackgroundTasks for simplicity
    background_tasks_fastapi.add_task(
        _execute_pipeline_task,
        executor_params=executor_params,
        experiment_run_id=config.experiment_run_id,
        artifact_repo=artifact_repo_from_state # Pass the repo instance
    )

    logger.info(f"Experiment {config.experiment_run_id} submitted to background execution.")
    return {"experiment_run_id": config.experiment_run_id, "message": "Experiment submitted for execution.", "status": "SUBMITTED_TO_PYTHON"}
