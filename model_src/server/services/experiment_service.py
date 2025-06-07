# from app.main import artifact_repo_instance # Avoid global if possible

import logging
from pathlib import Path

import httpx  # For asynchronous HTTP calls from the main FastAPI thread (e.g., initial FAILED if dataset not found)
import requests  # For synchronous HTTP calls from the background task
from fastapi import Request as FastAPIRequest, BackgroundTasks, HTTPException

# Assuming these are your Pydantic models from app/api/utils.py
from ..api.utils import RunExperimentRequest
from ..core.config import settings, APP_LOGGER_NAME
from ..ml.config import AugmentationStrategy, DATASET_DICT
# Assuming your ML pipeline imports
from ..ml.pipeline import PipelineExecutor

fastapi_app_logger = logging.getLogger(APP_LOGGER_NAME) # Your FastAPI app logger
pipeline_logger = logging.getLogger("ImgClassPipe") # Your pipeline's logger


def update_experiment_in_java_sync(
    experiment_run_id: str,
    status: str,
    model_relative_path: str = None,
    error_message: str = None,
    set_end_time: bool = False
):
    """
    Synchronously updates the experiment status (and optionally model_path/error) in the Java backend.
    To be called from the background ML task.
    """
    url = f"{settings.JAVA_INTERNAL_API_URL}/experiments/{experiment_run_id}/update" # Changed endpoint to /update
    payload = {"status": status.upper()} # Ensure status is uppercase for enum matching in Java

    if model_relative_path:
        payload["model_relative_path"] = model_relative_path
    if error_message:
        payload["error_message"] = error_message
    if set_end_time:
        payload["set_end_time"] = True # Java will set current time if this is true

    headers = {"X-Internal-API-Key": settings.INTERNAL_API_KEY, "Content-Type": "application/json"}
    source_logger = pipeline_logger if status in ["RUNNING", "COMPLETED", "FAILED"] else fastapi_app_logger

    try:
        response = requests.put(url, json=payload, headers=headers, timeout=360) # TODO timeout back to 15
        response.raise_for_status()
        source_logger.info(f"Successfully updated experiment {experiment_run_id} to {status} in Java. Payload: {payload}")
    except requests.exceptions.HTTPError as e:
        source_logger.error(f"HTTP error updating Java status for {experiment_run_id} to {status}: {e.response.status_code} - {e.response.text}. Payload: {payload}")
    except requests.exceptions.RequestException as e:
        source_logger.error(f"RequestException updating Java status for {experiment_run_id} to {status}: {e}. Payload: {payload}")
    except Exception as e: # Catch any other unexpected error during the update
        source_logger.error(f"Unexpected error updating Java status for {experiment_run_id} to {status}: {e}. Payload: {payload}", exc_info=True)


def _execute_pipeline_task(
    executor_params: dict,
    experiment_run_id: str,
    artifact_repo # Pass the initialized repo
):
    """Synchronous function to be run in background thread for the ML pipeline."""
    pipeline_logger.info(f"Background task started for experiment: {experiment_run_id}")
    final_model_path = None
    error_during_execution = None

    try:
        # 1. Update status to RUNNING
        update_experiment_in_java_sync(experiment_run_id, status="RUNNING")

        # 2. Initialize and run the executor
        executor = PipelineExecutor(**executor_params, artifact_repository=artifact_repo)
        final_results = executor.run() # This is a blocking call
        pipeline_logger.info(f"PipelineExecutor finished for experiment: {experiment_run_id}.")

        # Try to extract the saved model path from the results of the *last successful method that might save a model*
        # This logic depends on how `PipelineExecutor.run()` structures `final_results`
        # and how individual pipeline methods return `saved_model_path`.
        # Assuming `final_results` is a dict where keys are "methodName_idx"
        for method_result_key in reversed(list(final_results.keys())): # Check last methods first
            result_data = final_results[method_result_key]
            if isinstance(result_data, dict):
                # Check for common keys where model path might be stored
                # Note: The model path from PipelineExecutor should be RELATIVE to the experiment_run_id folder
                # e.g., "single_train_0/model_name.pt"
                if result_data.get("saved_model_path"):
                    # The path returned by _save_results is an S3 key or full local path.
                    # We need to make it relative to the current_executor_run_artifacts_prefix for storage in Java DB.
                    # current_executor_run_artifacts_prefix is like "experiments/DATASET/MODEL_TYPE/RUN_ID"
                    # saved_model_path from _save_results is like "experiments/DATASET/MODEL_TYPE/RUN_ID/method_0/model.pt"
                    full_saved_path = result_data["saved_model_path"]
                    base_prefix_to_strip = executor.current_executor_run_artifacts_prefix # Executor stores this
                    if full_saved_path.startswith(base_prefix_to_strip + "/"):
                        final_model_path = full_saved_path[len(base_prefix_to_strip) + 1:]
                    else: # Fallback if path doesn't match expected prefix structure
                        final_model_path = Path(full_saved_path).name # Just the filename as a last resort
                    pipeline_logger.info(f"Extracted model relative path for DB: {final_model_path} from full path: {full_saved_path}")
                    break # Found a model path

        # 3. Update status to COMPLETED
        update_experiment_in_java_sync(
            experiment_run_id,
            status="COMPLETED",
            model_relative_path=final_model_path,
            set_end_time=True
        )

    except Exception as e:
        pipeline_logger.critical(f"CRITICAL error during PipelineExecutor run for experiment {experiment_run_id}: {e}", exc_info=True)
        error_during_execution = str(e)
        # Update status to FAILED
        update_experiment_in_java_sync(
            experiment_run_id,
            status="FAILED",
            error_message=error_during_execution,
            set_end_time=True
        )
    finally:
        pipeline_logger.info(f"Background task processing finished for experiment: {experiment_run_id}")


async def start_experiment(
    request_fast_api: FastAPIRequest, # Renamed from 'request' to avoid conflict
    config: RunExperimentRequest,     # This is your Pydantic model
    background_tasks_fastapi: BackgroundTasks
):
    fastapi_app_logger.info(f"Received request to start experiment (system ID: {config.experiment_run_id}) for dataset '{config.dataset_name}'")

    dataset_root_path_on_server = settings.LOCAL_STORAGE_BASE_PATH
    dataset_path = dataset_root_path_on_server / DATASET_DICT[config.dataset_name]
    if not dataset_path.exists():
        error_msg = f"Dataset '{config.dataset_name}' not found at {dataset_path} for experiment {config.experiment_run_id}"
        fastapi_app_logger.error(error_msg)
        # Update Java to FAILED status immediately if dataset is missing (from main thread)
        # Use a separate async update function for calls from FastAPI main thread
        await update_experiment_in_java_async(
            config.experiment_run_id, "FAILED", error_message=error_msg, set_end_time=True
        )
        raise HTTPException(status_code=400, detail=error_msg) # FastAPI will convert ValueError to 422, this makes it 400

    methods_sequence_for_executor = []
    for method_param_api in config.methods_sequence: # method_param_api is ExperimentMethodParams (Pydantic)
        python_method_kwargs = {}
        # The 'params' field from Pydantic model is already a dict[str, Any]
        if method_param_api.params:
            python_method_kwargs['params'] = method_param_api.params

        # Map other top-level controls from API model to Python method kwargs
        # Ensure these field names match your ExperimentMethodParams Pydantic model
        if method_param_api.save_model is not None:
            python_method_kwargs['save_model'] = method_param_api.save_model
        if method_param_api.save_best_model is not None:
            python_method_kwargs['save_best_model'] = method_param_api.save_best_model
        if method_param_api.plot_level is not None:
            python_method_kwargs['plot_level'] = method_param_api.plot_level
        if method_param_api.results_detail_level is not None:
            python_method_kwargs['results_detail_level'] = method_param_api.results_detail_level
        if method_param_api.cv is not None:
            python_method_kwargs['cv'] = method_param_api.cv
        if method_param_api.outer_cv is not None:
            python_method_kwargs['outer_cv'] = method_param_api.outer_cv
        if method_param_api.inner_cv is not None:
            python_method_kwargs['inner_cv'] = method_param_api.inner_cv
        if method_param_api.scoring is not None:
            python_method_kwargs['scoring'] = method_param_api.scoring
        if method_param_api.method_search_type is not None: # Pydantic model uses 'method_search_type'
            python_method_kwargs['method'] = method_param_api.method_search_type # Python pipeline method uses 'method'
        if method_param_api.n_iter is not None:
            python_method_kwargs['n_iter'] = method_param_api.n_iter
        if method_param_api.evaluate_on is not None:
            python_method_kwargs['evaluate_on'] = method_param_api.evaluate_on
        if method_param_api.val_split_ratio is not None:
            python_method_kwargs['val_split_ratio'] = method_param_api.val_split_ratio # Pass as 'val_split_ratio'
        if method_param_api.use_best_params_from_step is not None:
            python_method_kwargs['use_best_params_from_step'] = method_param_api.use_best_params_from_step

        methods_sequence_for_executor.append(
            (method_param_api.method_name, python_method_kwargs)
        )

    img_h = config.img_size_h if config.img_size_h is not None else settings.DEFAULT_IMG_SIZE_H
    img_w = config.img_size_w if config.img_size_w is not None else settings.DEFAULT_IMG_SIZE_W

    aug_strat_enum = AugmentationStrategy.DEFAULT_STANDARD
    if config.augmentation_strategy_override:
        try:
            aug_strat_enum = AugmentationStrategy(config.augmentation_strategy_override)
        except ValueError:
            fastapi_app_logger.warning(f"Invalid augmentation_strategy_override: {config.augmentation_strategy_override}. Using default.")

    executor_params = {
        "dataset_path": dataset_path,
        "model_type": config.model_type, # This should be the string value for ModelType enum
        "experiment_base_key_prefix": "experiments",
        "methods": methods_sequence_for_executor,
        "img_size": (img_h, img_w),
        "save_main_log_file": True,
        "conceptual_experiment_run_name": config.experiment_run_id, # This is the Java-generated unique ID
        "max_epochs": settings.DEFAULT_MAX_EPOCHS,
        "results_detail_level": 2, # Global default, can be overridden per method
        "plot_level": 1,           # Global default, can be overridden per method
        "use_offline_augmented_data": config.offline_augmentation if config.offline_augmentation is not None else False,
        "augmentation_strategy": aug_strat_enum,
        # Add other **kwargs for PipelineExecutor if they come from RunExperimentRequest
        # e.g., "force_flat_for_fixed_cv": config.force_flat,
    }

    artifact_repo_from_state = request_fast_api.app.state.artifact_repo
    if not artifact_repo_from_state:
        error_msg = f"Artifact repository not configured on Python server. Cannot start experiment {config.experiment_run_id}."
        fastapi_app_logger.error(error_msg)
        await update_experiment_in_java_async( # Use async version here
            config.experiment_run_id, "FAILED", error_message=error_msg, set_end_time=True
        )
        raise HTTPException(status_code=500, detail=error_msg)


    background_tasks_fastapi.add_task(
        _execute_pipeline_task,
        executor_params=executor_params,
        experiment_run_id=config.experiment_run_id,
        artifact_repo=artifact_repo_from_state
    )

    fastapi_app_logger.info(f"Experiment (system ID: {config.experiment_run_id}) submitted to background execution.")
    # This response goes back to Java's PythonApiService
    return {"experiment_run_id": config.experiment_run_id, "message": "Experiment task submitted to Python background processing.", "status": "SUBMITTED_TO_PYTHON"}


async def update_experiment_in_java_async(
    experiment_run_id: str,
    status: str,
    model_relative_path: str = None,
    error_message: str = None,
    set_end_time: bool = False
):
    """
    Asynchronously updates the experiment status (and optionally model_path/error) in the Java backend.
    To be called from the main FastAPI async thread.
    """
    url = f"{settings.JAVA_INTERNAL_API_URL}/experiments/{experiment_run_id}/update"
    payload = {"status": status.upper()}
    if model_relative_path:
        payload["modelRelativePath"] = model_relative_path
    if error_message:
        payload["errorMessage"] = error_message
    if set_end_time:
        payload["setEndTime"] = True

    headers = {"X-Internal-API-Key": settings.INTERNAL_API_KEY, "Content-Type": "application/json"}
    source_logger = fastapi_app_logger # Use app logger for calls from main thread

    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(url, json=payload, headers=headers, timeout=360) # TODO timeout back to 15
            response.raise_for_status()
            source_logger.info(f"Successfully (async) updated experiment {experiment_run_id} to {status} in Java. Payload: {payload}")
    except httpx.HTTPStatusError as e:
        source_logger.error(f"HTTP error (async) updating Java status for {experiment_run_id} to {status}: {e.response.status_code} - {e.response.text}. Payload: {payload}")
    except httpx.RequestError as e:
        source_logger.error(f"RequestException (async) updating Java status for {experiment_run_id} to {status}: {e}. Payload: {payload}")
    except Exception as e:
        source_logger.error(f"Unexpected error (async) updating Java status for {experiment_run_id} to {status}: {e}. Payload: {payload}", exc_info=True)
