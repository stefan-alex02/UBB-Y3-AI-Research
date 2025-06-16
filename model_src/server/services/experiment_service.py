import logging
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Optional, List

import httpx
import requests
from fastapi import Request as FastAPIRequest, BackgroundTasks, HTTPException

from ..api.utils import RunExperimentRequest, ArtifactNode
from ..core import task_queue
from ..core.config import settings, APP_LOGGER_NAME
from ..ml.config import AugmentationStrategy, DATASET_DICT
from ..ml.pipeline import PipelineExecutor
from ..ml.pipeline.executor import ExecutorRunFailedError
from ..persistence import ArtifactRepository, LocalFileSystemRepository, MinIORepository

fastapi_app_logger = logging.getLogger(APP_LOGGER_NAME)
pipeline_logger = logging.getLogger("ImgClassPipe")


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
    url = f"{settings.JAVA_INTERNAL_API_URL}/experiments/{experiment_run_id}/update"
    payload = {"status": status.upper()}

    if model_relative_path:
        payload["model_relative_path"] = model_relative_path
    if error_message:
        payload["error_message"] = error_message
    if set_end_time:
        payload["set_end_time"] = True

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
    except Exception as e:
        source_logger.error(f"Unexpected error updating Java status for {experiment_run_id} to {status}: {e}. Payload: {payload}", exc_info=True)


def _execute_pipeline_task(
        executor_params: dict,
        experiment_run_id: str,
        artifact_repo
):
    pipeline_logger.info(f"Background task started for experiment: {experiment_run_id}")
    final_model_path = None
    error_message_for_java = "Experiment execution failed."

    try:
        update_experiment_in_java_sync(experiment_run_id, status="RUNNING")

        executor = PipelineExecutor(**executor_params, artifact_repository=artifact_repo)
        final_results = executor.run()

        pipeline_logger.info(f"PipelineExecutor run completed for experiment: {experiment_run_id}.")

        if final_results:
            for method_op_id_key in reversed(list(final_results.keys())):
                result_data = final_results.get(method_op_id_key)
                if isinstance(result_data, dict):
                    full_artifact_path_or_key = result_data.get("saved_model_path")

                    if full_artifact_path_or_key:
                        base_prefix_to_strip = executor.current_executor_run_artifacts_prefix

                        normalized_full_path = PurePath(full_artifact_path_or_key).as_posix()
                        normalized_base_prefix = PurePath(base_prefix_to_strip).as_posix()

                        if normalized_full_path.startswith(normalized_base_prefix + "/"):
                            final_model_path_for_db = normalized_full_path[len(normalized_base_prefix) + 1:]
                        else:
                            path_parts = PurePath(normalized_full_path).parts
                            if len(path_parts) >= 2:
                                final_model_path_for_db = str(PurePath(path_parts[-2]) / path_parts[-1])
                            else:
                                final_model_path_for_db = PurePath(normalized_full_path).name
                            pipeline_logger.warning(
                                f"Could not strip base prefix '{normalized_base_prefix}' from '{normalized_full_path}'. Using fallback relative path: '{final_model_path_for_db}'")

                        pipeline_logger.info(
                            f"Extracted model relative path for DB: {final_model_path_for_db} from full path: {full_artifact_path_or_key}")
                        break

        update_experiment_in_java_sync(
            experiment_run_id,
            status="COMPLETED",
            model_relative_path=final_model_path_for_db,
            set_end_time=True
        )

    except ExecutorRunFailedError as erf_err:
        pipeline_logger.error(
            f"ExecutorRunFailedError for experiment {experiment_run_id} in method '{erf_err.failed_method_id}': {erf_err}",
            exc_info=False
        )
        if erf_err.original_exception:
            pipeline_logger.error(f"Original exception type: {type(erf_err.original_exception).__name__}")
        error_message_for_java = f"Failed at method '{erf_err.failed_method_id}': {str(erf_err.original_exception) or erf_err.message}"
        update_experiment_in_java_sync(
            experiment_run_id,
            status="FAILED",
            error_message=error_message_for_java[:1000],
            set_end_time=True
        )
    except Exception as e:
        pipeline_logger.critical(
            f"CRITICAL unexpected error during background task for experiment {experiment_run_id}: {e}",
            exc_info=True
        )
        error_message_for_java = f"Unexpected critical error: {str(e)}"
        update_experiment_in_java_sync(
            experiment_run_id,
            status="FAILED",
            error_message=error_message_for_java[:1000],
            set_end_time=True
        )
    finally:
        pipeline_logger.info(f"Background task processing finished for experiment: {experiment_run_id}")


async def start_experiment(
    request_fast_api: FastAPIRequest,
    config: RunExperimentRequest,
    background_tasks_fastapi: BackgroundTasks
):
    fastapi_app_logger.info(
        f"Received request to start experiment (system ID: {config.experiment_run_id}) "
        f"for dataset '{config.dataset_name}' with model '{config.model_type}'"
    )

    dataset_root_path_on_server = settings.LOCAL_STORAGE_BASE_PATH
    dataset_path = dataset_root_path_on_server / DATASET_DICT[config.dataset_name]
    if not dataset_path.exists():
        error_msg = f"Dataset '{config.dataset_name}' not found at {dataset_path} for experiment {config.experiment_run_id}"
        fastapi_app_logger.error(error_msg)
        await update_experiment_in_java_async(config.experiment_run_id, "FAILED", error_message=error_msg, set_end_time=True)
        raise HTTPException(status_code=400, detail=error_msg)

    methods_sequence_for_executor = []
    for method_api_config in config.methods_sequence:
        python_method_kwargs = {}

        if method_api_config.method_name in ['non_nested_grid_search', 'nested_grid_search']:
            if method_api_config.param_grid is not None:
                python_method_kwargs['param_grid'] = method_api_config.param_grid
        elif method_api_config.method_name == 'cv_model_evaluation':
            if method_api_config.params is not None:
                python_method_kwargs['params'] = method_api_config.params
            if method_api_config.use_best_params_from_step is not None:
                python_method_kwargs['use_best_params_from_step'] = method_api_config.use_best_params_from_step
        elif method_api_config.method_name == 'single_train':
            if method_api_config.params is not None:
                python_method_kwargs['params'] = method_api_config.params

        if method_api_config.save_model is not None:
            python_method_kwargs['save_model'] = method_api_config.save_model
        if method_api_config.save_best_model is not None:
            python_method_kwargs['save_best_model'] = method_api_config.save_best_model
        if method_api_config.plot_level is not None:
            python_method_kwargs['plot_level'] = method_api_config.plot_level
        if method_api_config.results_detail_level is not None:
            python_method_kwargs['results_detail_level'] = method_api_config.results_detail_level
        if method_api_config.cv is not None:
            python_method_kwargs['cv'] = method_api_config.cv
        if method_api_config.outer_cv is not None:
            python_method_kwargs['outer_cv'] = method_api_config.outer_cv
        if method_api_config.inner_cv is not None:
            python_method_kwargs['inner_cv'] = method_api_config.inner_cv
        if method_api_config.scoring is not None:
            python_method_kwargs['scoring'] = method_api_config.scoring
        if method_api_config.method_search_type is not None:
            python_method_kwargs['method'] = method_api_config.method_search_type
        if method_api_config.n_iter is not None:
            python_method_kwargs['n_iter'] = method_api_config.n_iter
        if method_api_config.evaluate_on is not None:
            python_method_kwargs['evaluate_on'] = method_api_config.evaluate_on
        if method_api_config.val_split_ratio is not None:
            python_method_kwargs['val_split_ratio'] = method_api_config.val_split_ratio
        if method_api_config.method_name == 'cv_model_evaluation' and method_api_config.use_best_params_from_step is not None:
            python_method_kwargs['use_best_params_from_step'] = method_api_config.use_best_params_from_step

        methods_sequence_for_executor.append(
            (method_api_config.method_name, python_method_kwargs)
        )

    img_h = config.img_size_h if config.img_size_h is not None else settings.DEFAULT_IMG_SIZE_H
    img_w = config.img_size_w if config.img_size_w is not None else settings.DEFAULT_IMG_SIZE_W

    aug_strat_enum = AugmentationStrategy.DEFAULT_STANDARD
    if config.augmentation_strategy_override:
        try: aug_strat_enum = AugmentationStrategy(config.augmentation_strategy_override)
        except ValueError: fastapi_app_logger.warning(f"Invalid aug_strat_override: {config.augmentation_strategy_override}")

    executor_params = {
        "dataset_path": dataset_path,
        "model_type": config.model_type,
        "experiment_base_key_prefix": "experiments",
        "methods": methods_sequence_for_executor,
        "img_size": (img_h, img_w),
        "save_main_log_file": True,
        "conceptual_experiment_run_name": config.experiment_run_id,
        "random_seed_override": config.random_seed,
        "max_epochs": settings.DEFAULT_MAX_EPOCHS,
        "results_detail_level": 2,
        "plot_level": 1,
        "use_offline_augmented_data": config.offline_augmentation if config.offline_augmentation is not None else False,
        "augmentation_strategy": aug_strat_enum,
        "test_split_ratio_if_flat": config.test_split_ratio_if_flat,
        "force_flat_for_fixed_cv": config.force_flat_for_fixed_cv if config.force_flat_for_fixed_cv is not None else False,
    }

    artifact_repo_from_state = request_fast_api.app.state.artifact_repo
    if not artifact_repo_from_state:
        error_msg = f"Artifact repository not configured. Cannot start experiment {config.experiment_run_id}."
        fastapi_app_logger.error(error_msg)
        await update_experiment_in_java_async(config.experiment_run_id, "FAILED", error_message=error_msg, set_end_time=True)
        raise HTTPException(status_code=500, detail=error_msg)

    await task_queue.add_experiment_to_queue(
        _execute_pipeline_task,
        args=(),
        kwargs={
            "executor_params": executor_params,
            "experiment_run_id": config.experiment_run_id,
            "artifact_repo": artifact_repo_from_state
        }
    )

    fastapi_app_logger.info(f"Experiment (system ID: {config.experiment_run_id}) added to execution queue.")

    return {"experiment_run_id": config.experiment_run_id,
            "message": "Experiment task queued for sequential execution.", "status": "QUEUED_IN_PYTHON"}

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
    source_logger = fastapi_app_logger

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


def get_artifact_type_from_filename(filename: str) -> str:
    name_lower = filename.lower()
    if name_lower.endswith(".png") or name_lower.endswith(".jpg") or \
       name_lower.endswith(".jpeg") or name_lower.endswith(".gif") or name_lower.endswith(".svg"):
        return "image"
    if name_lower.endswith(".json"):
        return "json"
    if name_lower.endswith(".log") or name_lower.endswith(".txt"):
        return "log"
    if name_lower.endswith(".csv"):
        return "csv"
    if name_lower.endswith(".pt") or name_lower.endswith(".pth"):
        return "model"
    return "file"


def list_artifacts_for_experiment(
    artifact_repo: ArtifactRepository,
    dataset_name: str,
    model_type: str,
    experiment_run_id: str,
    sub_path: str = ""
) -> List[ArtifactNode]:
    base_experiment_prefix = PurePath("experiments", DATASET_DICT[dataset_name], model_type, experiment_run_id)
    current_scan_prefix = str((base_experiment_prefix / sub_path).as_posix()).strip("/")
    if current_scan_prefix and not current_scan_prefix.endswith("/"):
        current_scan_prefix += "/"

    fastapi_app_logger.info(f"Scanning artifacts under prefix: {current_scan_prefix}")
    nodes: List[ArtifactNode] = []

    if isinstance(artifact_repo, MinIORepository):
        listed_content = artifact_repo.list_objects_in_prefix(prefix=current_scan_prefix, delimiter='/')

        for folder_key in listed_content.get('subfolders', []):
            folder_name = PurePath(folder_key.rstrip('/')).name
            relative_path = str((PurePath(folder_key).relative_to(base_experiment_prefix)).as_posix())
            nodes.append(ArtifactNode(name=folder_name, path=relative_path, type="folder"))

        for file_key in listed_content.get('objects', []):
            if file_key == current_scan_prefix and file_key.endswith('/'):
                continue
            file_name = PurePath(file_key).name
            if not file_name:
                continue
            relative_path = str((PurePath(file_key).relative_to(base_experiment_prefix)).as_posix())
            file_type = get_artifact_type_from_filename(file_name)
            nodes.append(ArtifactNode(name=file_name, path=relative_path, type=file_type))

    elif isinstance(artifact_repo, LocalFileSystemRepository):
        scan_dir = Path(artifact_repo.base_path) / current_scan_prefix
        if scan_dir.is_dir():
            for item in sorted(scan_dir.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                relative_path = str((item.relative_to(Path(artifact_repo.base_path) / base_experiment_prefix)).as_posix())
                if item.is_dir():
                    nodes.append(ArtifactNode(name=item.name, path=relative_path, type="folder"))
                elif item.is_file():
                    file_type = get_artifact_type_from_filename(item.name)
                    stat_info = item.stat()
                    nodes.append(ArtifactNode(
                        name=item.name,
                        path=relative_path,
                        type=file_type,
                        size=stat_info.st_size,
                        last_modified=datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc)
                    ))
    else:
        fastapi_app_logger.warning("Artifact listing only implemented for MinIO and LocalFileSystem repositories.")

    nodes.sort(key=lambda n: (n.type != 'folder', n.name.lower()))
    return nodes


def get_experiment_artifact_content_bytes(
    artifact_repo: ArtifactRepository,
    dataset_name: str,
    model_type: str,
    experiment_run_id: str,
    artifact_relative_path: str
) -> Optional[bytes]:

    if artifact_relative_path.endswith(".pt") or artifact_relative_path.endswith(".pth"):
        fastapi_app_logger.warning(f"Attempted to fetch model file content directly: {artifact_relative_path}. Use /models/{dataset_name}/{model_type}/{experiment_run_id} instead.")
        return None

    full_artifact_key = str((PurePath("experiments", DATASET_DICT[dataset_name], model_type, experiment_run_id, artifact_relative_path)).as_posix())
    fastapi_app_logger.info(f"Fetching content for artifact: {full_artifact_key}")
    return artifact_repo.download_file_to_memory(full_artifact_key)
