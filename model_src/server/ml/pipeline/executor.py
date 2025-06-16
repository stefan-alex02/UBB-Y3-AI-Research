import logging
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from .pipeline import ClassificationPipeline
from ..architectures import ModelType
from ..config import logger_name_global, DEFAULT_IMG_SIZE, RANDOM_SEED as DEFAULT_PIPELINE_SEED, \
    AugmentationStrategy, apply_random_seed
from ..logger_utils import setup_logger, logger
from ...persistence import ArtifactRepository, LocalFileSystemRepository, MinIORepository


class ExecutorRunFailedError(RuntimeError):
    """Custom exception for when one or more methods in the executor fail."""
    def __init__(self, message, failed_method_id=None, original_exception=None, results_so_far=None):
        super().__init__(message)
        self.failed_method_id = failed_method_id
        self.original_exception = original_exception
        self.results_so_far = results_so_far


class PipelineExecutor:
    def __init__(self,
                 dataset_path: Union[str, Path],
                 conceptual_experiment_run_name: Optional[str] = None,
                 random_seed_override: Optional[int] = None,
                 model_type: Union[str, ModelType] = ModelType.CNN,
                 model_load_path: Optional[Union[str, Path]] = None,
                 artifact_repository: Optional[ArtifactRepository] = None,
                 experiment_base_key_prefix: str = "experiments",
                 results_detail_level: int = 1,
                 plot_level: int = 0,
                 save_main_log_file: bool = True,
                 methods: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 augmentation_strategy: Union[
                     str, AugmentationStrategy, Callable, None] = AugmentationStrategy.DEFAULT_STANDARD,
                 show_first_batch_augmentation_default: bool = False,
                 use_offline_augmented_data: bool = False,
                 force_flat_for_fixed_cv: bool = False,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 optimizer__weight_decay: float = 0.01,
                 module__dropout_rate: Optional[float] = None,
                 **kwargs
                 ):

        self.save_main_log_file = save_main_log_file
        self.seed_to_use = random_seed_override if random_seed_override is not None else DEFAULT_PIPELINE_SEED
        apply_random_seed(self.seed_to_use)

        logger.info(f"PipelineExecutor initialized with effective random seed: {self.seed_to_use}")

        _model_type_enum: ModelType
        if isinstance(model_type, str):
            try:
                _model_type_enum = ModelType(model_type)
            except ValueError:
                raise ValueError(f"Invalid model_type: '{model_type}'. Valid: {[mt.value for mt in ModelType]}")
        elif isinstance(model_type, ModelType):
            _model_type_enum = model_type
        else:
            raise TypeError(f"Executor model_type must be str or ModelType, got {type(model_type)}")

        dataset_name_for_path = Path(dataset_path).name

        if conceptual_experiment_run_name:
            self.conceptual_experiment_run_name = conceptual_experiment_run_name
            print(
                f"[Executor Init] Using provided conceptual_experiment_run_name: {self.conceptual_experiment_run_name}")
        else:
            timestamp_init_for_path = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.conceptual_experiment_run_name = f"{timestamp_init_for_path}_seed{self.seed_to_use}"
            print(f"[Executor Init] Generating conceptual_experiment_run_name: {self.conceptual_experiment_run_name}")

        self.current_executor_run_artifacts_prefix = str(PurePath(
            experiment_base_key_prefix,
            dataset_name_for_path,
            _model_type_enum.value,
            self.conceptual_experiment_run_name
        ).as_posix())

        # Configure Logger
        log_file_name_base = f"executor_run_{self.conceptual_experiment_run_name}.log"
        log_dir_for_file_setup: Optional[Path] = None
        self.log_file_local_path: Optional[Path] = None
        self.temp_log_dir: Optional[Path] = None

        if self.save_main_log_file:
            if isinstance(artifact_repository, LocalFileSystemRepository):
                log_dir_for_file_setup = Path(
                    artifact_repository.base_path) / self.current_executor_run_artifacts_prefix
                log_dir_for_file_setup.mkdir(parents=True, exist_ok=True)
                self.log_file_local_path = log_dir_for_file_setup / log_file_name_base
            elif artifact_repository:
                self.temp_log_dir = Path(tempfile.mkdtemp(prefix="pipeline_run_logs_"))
                self.log_file_local_path = self.temp_log_dir / log_file_name_base
                log_dir_for_file_setup = self.temp_log_dir

        setup_logger(
            name=logger_name_global,
            log_dir=log_dir_for_file_setup,
            log_filename=log_file_name_base if log_dir_for_file_setup else "console_only.log",
            level=logging.DEBUG,
            use_colors=True
        )

        if not self.save_main_log_file:
            logger.info("Main executor log file saving is DISABLED for this run.")
        elif self.log_file_local_path and self.log_file_local_path.exists():
            logger.info(f"Local main executor log file for this run (may be temporary): {self.log_file_local_path}")
        elif self.save_main_log_file and not log_dir_for_file_setup:
            logger.warning(
                "Main executor log file saving intended, but no suitable directory/repo context for file. Logging to console only.")

        logger.info(f"--- Starting Executor Run (ID: {self.conceptual_experiment_run_name}) ---")
        logger.info(f"Executor initialized for model '{_model_type_enum.value}' on dataset '{dataset_name_for_path}'")
        if artifact_repository:
            if isinstance(artifact_repository, LocalFileSystemRepository):
                logger.info(f"Local artifact storage base: {artifact_repository.base_path}")
            elif hasattr(artifact_repository, 'bucket_name'):
                logger.info(
                    f"Artifact repository: {type(artifact_repository).__name__} targeting bucket '{getattr(artifact_repository, 'bucket_name')}'")
            logger.info(
                f"Base key/prefix for this executor run's artifacts: {self.current_executor_run_artifacts_prefix}")
        else:
            logger.info("No Artifact Repository provided. Outputs will not be saved persistently by repository.")

        # Initialize Pipeline
        pipeline_init_kwargs = {
            "dataset_path": dataset_path, "model_type": _model_type_enum,
            "model_load_path": model_load_path, "artifact_repository": artifact_repository,
            "experiment_base_key_prefix": self.current_executor_run_artifacts_prefix,
            "results_detail_level": results_detail_level, "plot_level": plot_level,
            "img_size": img_size, "val_split_ratio": val_split_ratio,
            "test_split_ratio_if_flat": test_split_ratio_if_flat,
            "augmentation_strategy": augmentation_strategy,
            "show_first_batch_augmentation_default": show_first_batch_augmentation_default,
            "use_offline_augmented_data": use_offline_augmented_data,
            "force_flat_for_fixed_cv": force_flat_for_fixed_cv,
            "lr": lr, "max_epochs": max_epochs, "batch_size": batch_size, "patience": patience,
            "optimizer__weight_decay": optimizer__weight_decay,
            "module__dropout_rate": module__dropout_rate
        }
        pipeline_init_kwargs.update(kwargs)

        self.pipeline = ClassificationPipeline(**pipeline_init_kwargs)

        self.methods_to_run = methods if methods is not None else []
        self.all_results: Dict[str, Any] = {}
        try:
            self._validate_methods()
        except ValueError as e:
            logger.error(f"Method validation failed: {e}", exc_info=True)
            raise
        if self.methods_to_run:
            logger.info(f"Executor configured to run methods: {', '.join(m[0] for m in self.methods_to_run)}")
        else:
            logger.info("Executor configured with no methods to run.")

    def _validate_methods(self) -> None:
        valid_method_names = ['non_nested_grid_search', 'nested_grid_search', 'cv_model_evaluation', 'single_train',
                              'single_eval', 'load_model', 'predict_images']
        for i, (method_name, params) in enumerate(self.methods_to_run):
            if not isinstance(method_name, str) or method_name not in valid_method_names:
                raise ValueError(f"Invalid method name '{method_name}' at index {i}. Valid: {valid_method_names}")
            if not isinstance(params, dict):
                raise ValueError(f"Parameters for method '{method_name}' at index {i} must be a dict.")
            # Specific checks (can be expanded)
            if 'search' in method_name and 'param_grid' not in params:
                raise ValueError(f"Method '{method_name}' requires 'param_grid'.")
            if method_name == 'load_model' and 'model_path_or_key' not in params:
                raise ValueError(f"Method 'load_model' requires 'model_path_or_key'.")
            if method_name == 'predict_images' and not (
                    'image_tasks_for_pipeline' in params and 'experiment_run_id_of_model' in params):
                raise ValueError(
                    f"Method 'predict_images' requires 'image_tasks_for_pipeline' and 'experiment_run_id_of_model'.")
        logger.debug("Basic method validation successful.")

    def _get_previous_result(self, step_index: int, method_operation_id_key: str) -> Optional[Dict[str, Any]]:
        if step_index < 0 or step_index >= len(self.methods_to_run): return None
        return self.all_results.get(method_operation_id_key)

    def _upload_and_cleanup_log_if_needed(self):
        if self.save_main_log_file and \
                self.pipeline.artifact_repo and \
                self.log_file_local_path and \
                self.log_file_local_path.exists():

            active_logger = logging.getLogger(logger_name_global)
            file_handler_to_close = None
            for handler in active_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler) and \
                        hasattr(handler, 'baseFilename') and \
                        Path(handler.baseFilename).resolve() == self.log_file_local_path.resolve():
                    logger.debug(f"Flushing and closing log handler for: {handler.baseFilename} before upload/cleanup.")
                    handler.flush()
                    handler.close()
                    break

            if isinstance(self.pipeline.artifact_repo, MinIORepository):
                log_artifact_key = str(
                    (PurePath(self.current_executor_run_artifacts_prefix) / self.log_file_local_path.name).as_posix())
                logger.info(
                    f"Attempting to upload main executor log {self.log_file_local_path} to S3 key: {log_artifact_key}")
                saved_log_id = self.pipeline.artifact_repo.upload_file(self.log_file_local_path, log_artifact_key)
                if saved_log_id:
                    logger.info(f"Main executor log uploaded to: {saved_log_id}")
                else:
                    logger.error(f"Failed to upload main executor log: {self.log_file_local_path}")

            if self.temp_log_dir and self.temp_log_dir.exists():
                logger.info(f"Cleaning up temporary log directory: {self.temp_log_dir}")
                try:
                    shutil.rmtree(self.temp_log_dir)
                    self.temp_log_dir = None
                except Exception as e_cleanup:
                    logger.warning(f"Could not cleanup temp log dir {self.temp_log_dir}: {e_cleanup}")

        if self.save_main_log_file:
            if isinstance(self.pipeline.artifact_repo, LocalFileSystemRepository) and self.log_file_local_path:
                logger.info(f"Main executor log file available at: {self.log_file_local_path}")

    def run(self) -> Dict[str, Any]:
        self.all_results: dict = {'executor_run_id': self.conceptual_experiment_run_name, 'seed_used': self.seed_to_use}
        logger.info(f"Starting execution of methods for Executor Run ID: {self.conceptual_experiment_run_name}")
        start_time_total = time.time()
        execution_successful = True
        failed_method_id_for_exception = None
        error_message_for_exception = "Unknown error in method execution"
        original_exception_obj = None
        try:
            for i, (method_name, params_from_sequence) in enumerate(self.methods_to_run):
                method_operation_id = f"{method_name}_{i}"
                logger.info(
                    f"--- Running Method {i + 1}/{len(self.methods_to_run)}: {method_name} (Op ID: {method_operation_id}) ---")

                current_params = params_from_sequence.copy()

                use_best_params_key = 'use_best_params_from_step'
                step_index_to_use_params_from = current_params.pop(use_best_params_key, None)

                user_defined_skorch_hps_for_this_step = current_params.get('params',
                                                                                {}).copy() if method_name != 'non_nested_grid_search' and method_name != 'nested_grid_search' else {}

                if step_index_to_use_params_from is not None and isinstance(step_index_to_use_params_from, int):
                    if not (0 <= step_index_to_use_params_from < i):
                        err_msg = f"Invalid step_index_to_use_params_from '{step_index_to_use_params_from}' for '{method_name}'."

                        raise ExecutorRunFailedError(
                            message=err_msg,
                            failed_method_id=method_operation_id,
                            results_so_far=self.all_results
                        )

                    prev_method_op_id_key = f"{self.methods_to_run[step_index_to_use_params_from][0]}_{step_index_to_use_params_from}"
                    prev_result = self.all_results.get(prev_method_op_id_key)

                    if prev_result and isinstance(prev_result, dict) and 'best_params' in prev_result and isinstance(
                            prev_result['best_params'], dict):
                        best_params_from_prev = prev_result['best_params']
                        logger.info(
                            f"Method '{method_name}': Will use best_params from step {step_index_to_use_params_from} (Op ID: {prev_method_op_id_key}): {best_params_from_prev}")

                        final_skorch_hps = best_params_from_prev.copy()
                        final_skorch_hps.update(user_defined_skorch_hps_for_this_step)

                        current_params['params'] = final_skorch_hps
                        logger.info(f"  Merged/Overridden Skorch HPs for '{method_name}': {final_skorch_hps}")
                    else:
                        err_msg = f"No 'best_params' dict found in results of step {step_index_to_use_params_from}."
                        raise ExecutorRunFailedError(
                            message=err_msg,
                            failed_method_id=method_operation_id,
                            results_so_far=self.all_results
                        )
                elif user_defined_skorch_hps_for_this_step:
                    current_params['params'] = user_defined_skorch_hps_for_this_step

                logger.debug(f"Running method '{method_name}' with effective parameters: {current_params}")
                start_time_method = time.time()
                try:
                    pipeline_method = getattr(self.pipeline, method_name)
                    result = pipeline_method(**current_params)
                    self.all_results[method_operation_id] = result
                    method_duration = time.time() - start_time_method
                    logger.info(
                        f"--- Method {method_name} (Op ID: {method_operation_id}) completed successfully in {method_duration:.2f}s ---")
                except (ValueError, TypeError, AttributeError) as config_err:
                    logger.error(
                        f"!!! Configuration/Usage error in '{method_name}' (Op ID: {method_operation_id}): {config_err}",
                        exc_info=True)
                    self.all_results[method_operation_id] = {"error": str(config_err), "traceback": traceback.format_exc(),
                                                             "status": "FAILED"}
                    failed_method_id_for_exception = method_operation_id
                    error_message_for_exception = str(config_err)
                    original_exception_obj = config_err
                    execution_successful = False
                    break
                except FileNotFoundError as fnf_err:
                    logger.error(f"!!! File not found during '{method_name}' (Op ID: {method_operation_id}): {fnf_err}",
                                 exc_info=True)
                    self.all_results[method_operation_id] = {"error": str(fnf_err), "traceback": traceback.format_exc(),
                                                             "status": "FAILED"}
                    failed_method_id_for_exception = method_operation_id
                    error_message_for_exception = str(fnf_err)
                    original_exception_obj = fnf_err
                    execution_successful = False
                    break
                except RuntimeError as rt_err:
                    logger.error(f"!!! Runtime error during '{method_name}' (Op ID: {method_operation_id}): {rt_err}",
                                 exc_info=True)
                    self.all_results[method_operation_id] = {"error": str(rt_err), "traceback": traceback.format_exc(),
                                                             "status": "FAILED"}
                    failed_method_id_for_exception = method_operation_id
                    error_message_for_exception = str(rt_err)
                    original_exception_obj = rt_err
                    execution_successful = False
                    break
                except Exception as e:
                    logger.critical(
                        f"!!! Unexpected critical error during '{method_name}' (Op ID: {method_operation_id}): {e}",
                        exc_info=True)
                    self.all_results[method_operation_id] = {"error": str(e), "traceback": traceback.format_exc(),
                                                             "status": "FAILED_UNEXPECTED"}
                    failed_method_id_for_exception = method_operation_id
                    error_message_for_exception = str(e)
                    original_exception_obj = e
                    execution_successful = False
                    break

            total_duration = time.time() - start_time_total
            if not execution_successful:
                logger.error(
                    f"Pipeline execution for Run ID {self.conceptual_experiment_run_name} HALTED due to failure in method '{failed_method_id_for_exception}'. Total duration before halt: {total_duration:.2f}s."
                )
                raise ExecutorRunFailedError(
                    message=f"Execution failed at method '{failed_method_id_for_exception}': {error_message_for_exception}",
                    failed_method_id=failed_method_id_for_exception,
                    original_exception=original_exception_obj,
                    results_so_far=self.all_results
                )
            else:
                logger.info(
                    f"Pipeline execution finished successfully in {total_duration:.2f}s for Executor Run ID: {self.conceptual_experiment_run_name}.")

            return self.all_results

        finally:
            logger.info(f"Executor run for {self.conceptual_experiment_run_name} attempting final log operations.")
            self._upload_and_cleanup_log_if_needed()
