import logging
import time
from datetime import datetime
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from .pipeline import ClassificationPipeline
from ..architectures import ModelType  # Assuming ModelType is in architectures or config
from ..config import logger_name_global, DEFAULT_IMG_SIZE, RANDOM_SEED, \
    AugmentationStrategy  # Assuming these are in config
from ..logger_utils import setup_logger, logger  # Import the logger instance
from ...persistence import ArtifactRepository, LocalFileSystemRepository, MinIORepository


class PipelineExecutor:
    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: Union[str, ModelType] = ModelType.CNN,
                 model_load_path: Optional[Union[str, Path]] = None,
                 artifact_repository: Optional[ArtifactRepository] = None,
                 experiment_base_key_prefix: str = "experiments",
                 results_detail_level: int = 1,
                 plot_level: int = 0,
                 methods: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 augmentation_strategy: Union[
                     str, AugmentationStrategy, Callable, None] = AugmentationStrategy.DEFAULT_STANDARD,
                 show_first_batch_augmentation_default: bool = False,  # Added from previous context
                 force_flat_for_fixed_cv: bool = False,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 optimizer__weight_decay: float = 0.01,
                 module__dropout_rate: Optional[float] = None
                 ):

        # --- 1. Determine key identifiers and paths FIRST ---
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
        timestamp_init_for_path = datetime.now().strftime('%Y%m%d_%H%M%S')
        conceptual_experiment_run_name = f"{timestamp_init_for_path}_seed{RANDOM_SEED}"

        current_experiment_run_key_prefix = str(PurePath(
            experiment_base_key_prefix,
            dataset_name_for_path,
            _model_type_enum.value,
            conceptual_experiment_run_name
        ).as_posix())

        # --- 2. Configure Logger NOW ---
        log_file_name_base = f"experiment_run_{conceptual_experiment_run_name}.log"
        log_dir_for_file_setup: Optional[Path] = None
        self.log_file_local_path: Optional[Path] = None  # Initialize attribute
        self.temp_log_dir: Optional[Path] = None  # Initialize attribute

        if isinstance(artifact_repository, LocalFileSystemRepository):
            # For local repo, log file goes into the experiment's specific local folder
            log_dir_for_file_setup = Path(artifact_repository.base_path) / current_experiment_run_key_prefix
            log_dir_for_file_setup.mkdir(parents=True, exist_ok=True)
            self.log_file_local_path = log_dir_for_file_setup / log_file_name_base
        elif artifact_repository:  # MinIO or other non-local repo
            import tempfile  # Keep import local if only used here
            self.temp_log_dir = Path(tempfile.mkdtemp(prefix="pipeline_run_logs_"))
            self.log_file_local_path = self.temp_log_dir / log_file_name_base
            log_dir_for_file_setup = self.temp_log_dir
            # Initial logger info will go to console if file handler not set yet or if this is first setup
            # logger.info(f"Logging temporarily to {self.log_file_local_path} for later S3 upload.")
        else:  # No repo, console only
            self.log_file_local_path = None

        # The global 'logger' instance from logger_utils is (re)configured here.
        # This setup_logger call effectively makes 'logger' usable by all modules that import it.
        setup_logger(
            name=logger_name_global,
            log_dir=log_dir_for_file_setup,
            log_filename=log_file_name_base if log_dir_for_file_setup else "console_only.log",
            level=logging.DEBUG,  # Or your desired default level
            use_colors=True
        )
        # --- Logger is now configured ---

        # Now log executor-level information
        if not artifact_repository:
            logger.info("No Artifact Repository provided. Outputs will not be saved persistently by the repository.")
            logger.info("Logging will be directed to console only for file output (if not already).")
        else:
            logger.info(f"--- Starting Experiment Run (ID: {conceptual_experiment_run_name}) ---")
            logger.info(
                f"Executor initialized for model '{_model_type_enum.value}' on dataset '{dataset_name_for_path}'")
            if isinstance(artifact_repository, LocalFileSystemRepository):
                logger.info(f"Local artifact storage base: {artifact_repository.base_path}")
            elif hasattr(artifact_repository, 'bucket_name'):
                logger.info(
                    f"Artifact repository: {type(artifact_repository).__name__} targeting bucket '{artifact_repository.bucket_name}'")
            logger.info(f"Base key/prefix for this run's artifacts: {current_experiment_run_key_prefix}")
        if self.log_file_local_path and self.log_file_local_path.exists():
            logger.info(f"Local log file for this run (might be temporary): {self.log_file_local_path}")

        # --- 3. Initialize Pipeline (it can now use the configured logger) ---
        self.pipeline = ClassificationPipeline(
            dataset_path=dataset_path, model_type=_model_type_enum, model_load_path=model_load_path,
            artifact_repository=artifact_repository,
            experiment_base_key_prefix=current_experiment_run_key_prefix,
            results_detail_level=results_detail_level, plot_level=plot_level,
            img_size=img_size, val_split_ratio=val_split_ratio,
            test_split_ratio_if_flat=test_split_ratio_if_flat,
            augmentation_strategy=augmentation_strategy,  # Pass it down
            show_first_batch_augmentation_default=show_first_batch_augmentation_default,  # Pass it down
            force_flat_for_fixed_cv=force_flat_for_fixed_cv, lr=lr, max_epochs=max_epochs,
            batch_size=batch_size, patience=patience,
            optimizer__weight_decay=optimizer__weight_decay, module__dropout_rate=module__dropout_rate
        )

        self.methods_to_run = methods if methods is not None else []
        self.all_results: Dict[str, Any] = {}
        try:
            self._validate_methods()
        except ValueError as e:
            logger.error(f"Method validation failed: {e}")
            raise
        if self.methods_to_run:
            method_names = [m[0] for m in self.methods_to_run]
            logger.info(f"Executor configured to run methods: {', '.join(method_names)}")
        else:
            logger.info("Executor configured with no methods to run.")

    def _validate_methods(self) -> None:  # ... (unchanged)
        valid_method_names = ['non_nested_grid_search', 'nested_grid_search', 'cv_model_evaluation', 'single_train',
                              'single_eval', 'load_model', 'predict_images', ];
        for i, (method_name, params) in enumerate(self.methods_to_run):
            if not isinstance(method_name, str) or method_name not in valid_method_names: raise ValueError(
                f"Invalid method name '{method_name}' at index {i}. Valid: {valid_method_names}")
            if not isinstance(params, dict): raise ValueError(
                f"Parameters for method '{method_name}' at index {i} must be a dict.")
            if 'search' in method_name and 'param_grid' not in params: raise ValueError(
                f"Method '{method_name}' requires 'param_grid'.")
            if method_name == 'load_model' and 'model_path_or_key' not in params: raise ValueError(
                f"Method 'load_model' requires 'model_path_or_key'.")
        logger.debug("Basic method validation successful.")

    def _get_previous_result(self, step_index: int) -> Optional[Dict[str, Any]]:  # ... (unchanged)
        if step_index < 0 or step_index >= len(self.methods_to_run): return None
        prev_method_name, _ = self.methods_to_run[step_index];
        run_id = f"{prev_method_name}_{step_index}";
        return self.all_results.get(run_id)

    def run(self) -> Dict[str, Any]:  # ... (unchanged logic, calls pipeline methods)
        self.all_results = {};
        logger.info("Starting execution of pipeline methods...");
        start_time_total = time.time()
        for i, (method_name, params) in enumerate(self.methods_to_run):
            run_id = f"{method_name}_{i}";
            logger.info(f"--- Running Method {i + 1}/{len(self.methods_to_run)}: {method_name} ({run_id}) ---")
            current_params = params.copy();
            use_best_params_key = 'use_best_params_from_step'
            if use_best_params_key in current_params:
                prev_step_index = current_params.pop(use_best_params_key)
                if not isinstance(prev_step_index, int) or prev_step_index >= i: logger.error(
                    f"Invalid prev step index '{prev_step_index}' for '{method_name}'."); self.all_results[run_id] = {
                    "error": f"Invalid '{use_best_params_key}' value."}; break
                logger.info(
                    f"Injecting 'best_params' from step {prev_step_index} ({self.methods_to_run[prev_step_index][0]}) into params for '{method_name}'.")
                prev_result = self._get_previous_result(prev_step_index)
                if prev_result and isinstance(prev_result, dict) and 'best_params' in prev_result and isinstance(
                        prev_result['best_params'], dict):
                    best_params = prev_result['best_params'];
                    logger.info(f"  Injecting best params: {best_params}")
                    if 'params' not in current_params: current_params['params'] = {}
                    if isinstance(current_params['params'], dict):
                        final_nested_params = best_params.copy(); final_nested_params.update(current_params['params']);
                        current_params['params'] = final_nested_params
                    else:
                        logger.error(f"'params' key for step {i} not dict."); self.all_results[run_id] = {
                            "error": "'params' key not dict."}; break
                else:
                    logger.error(f"No 'best_params' in results of step {prev_step_index}."); self.all_results[
                        run_id] = {"error": f"Missing 'best_params' in step {prev_step_index}."}; break
            logger.debug(f"Running with effective parameters: {current_params}");
            start_time_method = time.time()
            try:
                pipeline_method = getattr(self.pipeline, method_name);
                result = pipeline_method(**current_params);
                self.all_results[run_id] = result;
                method_duration = time.time() - start_time_method;
                logger.info(f"--- Method {method_name} ({run_id}) completed successfully in {method_duration:.2f}s ---")
            except ValueError as ve:
                logger.error(f"!!! Config error in '{method_name}': {ve}", exc_info=True); logger.error(
                    f"!!! Check compatibility/params."); self.all_results[run_id] = {"error": str(ve)}; break
            except FileNotFoundError as fnf:
                logger.error(f"!!! File not found during '{method_name}': {fnf}", exc_info=True); self.all_results[
                    run_id] = {"error": str(fnf)}; break
            except RuntimeError as rte:
                logger.error(f"!!! Runtime error during '{method_name}': {rte}", exc_info=True); self.all_results[
                    run_id] = {"error": str(rte)}; break
            except Exception as e:
                logger.critical(f"!!! Unexpected critical error during '{method_name}': {e}", exc_info=True);
                self.all_results[run_id] = {"error": str(e), "traceback": logging.traceback.format_exc()}; break
        total_duration = time.time() - start_time_total
        logger.info(f"Pipeline execution finished in {total_duration:.2f}s.")

        # --- Upload log file if MinIO/S3 repo was used and temp log exists ---
        # Check if artifact_repo exists and if log_file_local_path was set and exists
        if self.pipeline.artifact_repo and \
                hasattr(self, 'log_file_local_path') and self.log_file_local_path and \
                self.log_file_local_path.exists() and \
                self.pipeline.experiment_run_key_prefix:

            # Only try to upload if it's not a LocalFileSystemRepository (which already wrote it to final place)
            # and if it's indeed a MinIORepository (or similar S3-like repo)
            if isinstance(self.pipeline.artifact_repo, MinIORepository):  # More specific check
                log_artifact_key = str(
                    (PurePath(self.pipeline.experiment_run_key_prefix) / self.log_file_local_path.name).as_posix())

                logger.info(
                    f"Attempting to upload log file {self.log_file_local_path} to artifact key: {log_artifact_key}")

                # --- CRUCIAL: Ensure all handlers are flushed and closed, then remove them ---
                active_logger = logging.getLogger(logger_name_global)
                handlers_to_remove = []
                for handler in active_logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        # Check if baseFilename matches the one we want to upload
                        try:
                            # handler.baseFilename might be None if not configured or already closed
                            if handler.baseFilename and Path(
                                    handler.baseFilename).resolve() == self.log_file_local_path.resolve():
                                logger.debug(f"Flushing and closing log handler for: {handler.baseFilename}")
                                handler.flush()
                                handler.close()
                                handlers_to_remove.append(handler)
                        except Exception as e_handler:
                            logger.warning(f"Error processing handler {handler}: {e_handler}")

                for handler in handlers_to_remove:
                    logger.debug(f"Removing handler: {handler}")
                    active_logger.removeHandler(handler)

                # As a final measure, call logging.shutdown() - this flushes and closes all handlers
                # registered by the logging module. This is usually safe at the very end of a script/process.
                logging.shutdown()
                # --- End Handler Management ---

                # Now attempt upload
                saved_log_identifier = self.pipeline.artifact_repo.upload_file(
                    self.log_file_local_path,
                    log_artifact_key
                )

                if saved_log_identifier:
                    logger.info(f"Log file uploaded via repository to: {saved_log_identifier}")
                else:
                    logger.error(f"Failed to upload log file {self.log_file_local_path} via repository.")

            # Clean up temporary log directory if it was created for non-local repos
            if hasattr(self, 'temp_log_dir') and self.temp_log_dir.exists():
                try:
                    import shutil
                    # Add a small delay and retry loop for Windows file lock issues
                    for _ in range(3):  # Retry up to 3 times
                        try:
                            shutil.rmtree(self.temp_log_dir)
                            logger.info(f"Cleaned up temporary log directory: {self.temp_log_dir}")
                            break  # Success
                        except PermissionError as pe:
                            logger.warning(f"PermissionError cleaning temp log dir (will retry): {pe}")
                            time.sleep(0.5)  # Wait a bit
                        except Exception as e_rm:  # Catch other potential errors during rmtree
                            logger.error(f"Error during shutil.rmtree of {self.temp_log_dir}: {e_rm}")
                            break  # Don't retry on other errors
                    else:  # If loop completed without break (all retries failed)
                        logger.error(
                            f"Failed to clean up temporary log directory {self.temp_log_dir} after multiple retries.")

                except Exception as e_clean:  # Catch errors from initial shutil.rmtree attempt or import
                    logger.warning(f"Could not clean up temp log dir {self.temp_log_dir}: {e_clean}")

        elif self.log_file_local_path and self.log_file_local_path.exists():
            logger.info(f"Final log file (local): {self.log_file_local_path}")
        elif not self.log_file_local_path:  # This case means log_dir_for_file_setup was None
            logger.info("No log file was configured for saving (console only or repo setup issue).")

        return self.all_results
