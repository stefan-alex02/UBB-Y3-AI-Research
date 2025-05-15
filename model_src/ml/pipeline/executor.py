import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from .pipeline import ClassificationPipeline
from ..architectures import ModelType
from ..config import logger_name_global, DEFAULT_IMG_SIZE
from ..logger_utils import setup_logger


class PipelineExecutor:
    """
    Executes a sequence of classification pipeline_v1 methods.
    Handles parameter passing and compatibility checks.
    """
    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: Union[str, ModelType] = ModelType.CNN,
                 model_load_path: Optional[Union[str, Path]] = None,
                 results_dir: Optional[Union[str, Path]] = 'results',
                 results_detail_level: int = 1,
                 plot_level: int = 0,
                 methods: List[Tuple[str, Dict[str, Any]]]= None,
                 # Pipeline config params passed down
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 data_augmentation: bool = True,
                 force_flat_for_fixed_cv: bool = False,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 optimizer__weight_decay: float = 0.01,
                 module__dropout_rate: Optional[float] = None
                 ):
        """
            Initializes the Pipeline Executor.

            Args:
                dataset_path: Path to the root of the image dataset.
                model_type: Type of model to use for classification.
                    Options: 'cnn', 'simple_vit', 'flexible_vit', 'diffusion'.
                model_load_path: Optional path to pre-trained model weights to load at pipeline init.
                results_dir: Base directory where experiment results will be saved.
                         If None, no file outputs (logs, JSON, models, plots) will be saved to disk.
                         Results are still returned in memory. Defaults to 'results'.
                results_detail_level: Default level of detail for saving JSON results by the
                                      ClassificationPipeline. This can be overridden for individual
                                      method calls within the `methods` sequence by providing
                                      `results_detail_level_override` in their parameter dictionary.
                    Levels:
                    - 0: No JSON results file is saved. Only summary CSV is updated.
                    - 1 (Basic Summary): Saves key metrics, best parameters, summary CV scores.
                      Excludes verbose lists like full histories, raw prediction arrays, etc.
                    - 2 (Detailed Epoch-Level): Includes Level 1, plus full epoch histories
                      (without batch data), detailed per-class metrics, y_true/pred/score arrays,
                      ROC/PR curve data, and full GridSearchCV cv_results.
                    - 3 (Full Detail including Batch Data): Includes Level 2, and preserves
                      per-batch training data if present in skorch History.
                plot_level: Default level for plotting results after methods run (0-2).
                        0: No plots, 1: Save plots, 2: Save and show plots.
                        Can be overridden per method in the `methods` sequence via
                        `plot_level_override`.
                methods: A list of (method_name, params_dict) tuples defining the sequence
                         of pipeline operations to run.
                img_size: Target image size for transformations (height, width).
                val_split_ratio: Default ratio for train/validation splits.
                test_split_ratio_if_flat: Ratio for train/test split if dataset is FLAT.
                data_augmentation: Whether to use data augmentation for training.
                force_flat_for_fixed_cv: If True, treats FIXED datasets as FLAT for 'full' dataset CV.
                lr: Default learning rate.
                max_epochs: Default maximum training epochs.
                batch_size: Default batch size.
                patience: Default patience for EarlyStopping.
                optimizer__weight_decay: Default weight decay for AdamW.
                module__dropout_rate: Optional default dropout rate for model modules.
        """
        global logger # <<< Access the global logger instance

        # Convert string to ModelType if necessary (or let ClassificationPipeline handle it)
        # For consistency, it's good if PipelineExecutor also expects/handles the Enum.
        _model_type_enum: ModelType
        if isinstance(model_type, str):
            try:
                _model_type_enum = ModelType(model_type)
            except ValueError:
                raise ValueError(
                    f"Invalid model_type string in Executor: '{model_type}'. "
                    f"Valid types are: {[mt.value for mt in ModelType]}"
                )
        elif isinstance(model_type, ModelType):
            _model_type_enum = model_type
        else:
            raise TypeError(
                f"Executor model_type must be string or ModelType enum, got {type(model_type)}"
            )

        # --- Initialize Pipeline FIRST to get experiment_dir ---
        self.pipeline = ClassificationPipeline(
            dataset_path=dataset_path, model_type=_model_type_enum, model_load_path=model_load_path,
            results_dir=results_dir,
            results_detail_level=results_detail_level, plot_level=plot_level,
            img_size=img_size, val_split_ratio=val_split_ratio,
            test_split_ratio_if_flat=test_split_ratio_if_flat, data_augmentation=data_augmentation,
            force_flat_for_fixed_cv=force_flat_for_fixed_cv, lr=lr, max_epochs=max_epochs,
            batch_size=batch_size, patience=patience,
            optimizer__weight_decay=optimizer__weight_decay, module__dropout_rate=module__dropout_rate
        )
        # --- End Pipeline Init ---

        # --- Configure Logger AFTER pipeline init ---
        logger_name = 'ImgClassPipe'
        experiment_log_dir = self.pipeline.experiment_dir # Get dir from pipeline (this could be None)
        logger = setup_logger(
             name=logger_name_global, # Use the globally defined name
             log_dir=experiment_log_dir,
             log_filename=f"experiment_{Path(experiment_log_dir).name}.log" if experiment_log_dir else "experiment_console_only.log",
             level=logging.DEBUG, # Set desired level
             use_colors=True
        )
        if results_dir is None:
            logger.info("Results directory is None. No file outputs will be saved (logs, JSONs, models, plots).")
            logger.info("Logging will be directed to console only.")
        else:
            logger.info(f"--- Starting Experiment Run ---")
            logger.info(
                f"Pipeline Executor initialized for model '{self.pipeline.model_type.value}' on dataset '{Path(dataset_path).name}'")
            logger.info(f"Results base directory: {self.pipeline.experiment_dir}")
        # --- End Logger Config ---

        self.methods_to_run = methods if methods is not None else []
        self.all_results: Dict[str, Any] = {}
        try: # Validate methods after logger is fully set up
             self._validate_methods()
        except ValueError as e:
             logger.error(f"Method validation failed: {e}")
             raise # Re-raise after logging
        if self.methods_to_run:  # Only log if there are methods
            method_names = [m[0] for m in self.methods_to_run]
            logger.info(f"Executor configured to run methods: {', '.join(method_names)}")
        else:
            logger.info("Executor configured with no methods to run.")

    def _validate_methods(self) -> None:
        """ Basic validation of method names and parameter types. """
        valid_method_names = [
            'non_nested_grid_search', 'nested_grid_search', 'cv_model_evaluation',
            'single_train', 'single_eval', 'load_model', 'predict_images',
        ]
        for i, (method_name, params) in enumerate(self.methods_to_run):
            if not isinstance(method_name, str) or method_name not in valid_method_names:
                 raise ValueError(f"Invalid method name '{method_name}' at index {i}. Valid: {valid_method_names}")
            if not isinstance(params, dict):
                 raise ValueError(f"Parameters for method '{method_name}' at index {i} must be a dict.")
            # Specific parameter checks (examples)
            if 'search' in method_name and 'param_grid' not in params:
                 raise ValueError(f"Method '{method_name}' requires 'param_grid'.")
            if method_name == 'load_model' and 'model_path' not in params:
                 raise ValueError(f"Method 'load_model' requires 'model_path'.")
            # Compatibility checks are now mostly done *inside* the methods themselves.
        logger.debug("Basic method validation successful.")

    # Add helper method to get previous results
    def _get_previous_result(self, step_index: int) -> Optional[Dict[str, Any]]:
        """Gets the results dict from a previous step if available."""
        if step_index < 0 or step_index >= len(self.methods_to_run):
            return None
        prev_method_name, _ = self.methods_to_run[step_index]
        run_id = f"{prev_method_name}_{step_index}"
        return self.all_results.get(run_id)

    def run(self) -> Dict[str, Any]:
        """ Executes the configured sequence of pipeline_v1 methods. """
        self.all_results = {}
        logger.info("Starting execution of pipeline_v1 methods...")
        start_time_total = time.time()

        for i, (method_name, params) in enumerate(self.methods_to_run):
            run_id = f"{method_name}_{i}"  # <<< USE THIS AS THE UNIQUE ID
            logger.info(f"--- Running Method {i + 1}/{len(self.methods_to_run)}: {method_name} ({run_id}) ---")

            # --- Parameter Injection Logic ---
            current_params = params.copy()  # Work with a copy
            use_best_params_key = 'use_best_params_from_step'
            if use_best_params_key in current_params:
                prev_step_index = current_params.pop(use_best_params_key)
                if not isinstance(prev_step_index, int) or prev_step_index >= i:
                    logger.error(f"Invalid previous step index '{prev_step_index}' for '{method_name}'.")
                    self.all_results[run_id] = {"error": f"Invalid '{use_best_params_key}' value."}
                    break
                logger.info(
                    f"Injecting 'best_params' from step {prev_step_index} ({self.methods_to_run[prev_step_index][0]}) into params for '{method_name}'.")
                prev_result = self._get_previous_result(prev_step_index)

                if prev_result and isinstance(prev_result, dict) and 'best_params' in prev_result and isinstance(
                        prev_result['best_params'], dict):
                    best_params = prev_result['best_params']
                    logger.info(f"  Injecting best params: {best_params}")

                    # --- MODIFIED MERGING ---
                    # Ensure 'params' dict exists in current_params for methods like cv_model_evaluation
                    if 'params' not in current_params:
                        current_params['params'] = {}  # Create if missing

                    if isinstance(current_params['params'], dict):
                        # Create final nested params: Start with best_params, then overwrite
                        # with any specific 'params' the user provided for this step.
                        final_nested_params = best_params.copy()
                        final_nested_params.update(current_params['params'])  # Apply user overrides
                        current_params['params'] = final_nested_params
                    else:
                        logger.error(f"'params' key in config for step {i} is not a dict. Cannot inject best_params.")
                        self.all_results[run_id] = {"error": "'params' key is not a dictionary."}
                        break
                    # --- END MODIFIED MERGING ---

                else:
                    logger.error(f"Could not find 'best_params' dictionary in results of step {prev_step_index}.")
                    self.all_results[run_id] = {"error": f"Missing 'best_params' in step {prev_step_index} results."}
                    break
            # --- End Parameter Injection Logic ---

            logger.debug(f"Running with effective parameters: {current_params}")
            start_time_method = time.time()

            try:
                pipeline_method = getattr(self.pipeline, method_name)
                # --- Pass run_id to save_results via the method if necessary ---
                # This requires modifying the signature of _save_results and how it's called
                # OR modifying methods to accept run_id if they call save_results internally
                # Let's modify _save_results to accept run_id instead.
                result = pipeline_method(**current_params)  # Call method
                # If the method didn't call _save_results itself (e.g., load_model)
                # we might want to save something here, but usually results are generated
                # by the methods that do computations.

                # Ensure the result is stored correctly
                self.all_results[run_id] = result
                method_duration = time.time() - start_time_method
                logger.info(f"--- Method {method_name} ({run_id}) completed successfully in {method_duration:.2f}s ---")

            except ValueError as ve:  # Catch specific config errors
                # Change exc_info to True here to log the stack trace for ValueError
                logger.error(f"!!! Configuration error in '{method_name}': {ve}", exc_info=True)
                logger.error(
                    f"!!! Check method compatibility with dataset structure (FIXED requires force_flat_for_fixed_cv=True for some methods) or parameters.")
                self.all_results[run_id] = {"error": str(ve)}
                break  # Stop execution on config errors
            except FileNotFoundError as fnf:
                # This already logs the stack trace
                logger.error(f"!!! File not found during '{method_name}': {fnf}", exc_info=True)
                self.all_results[run_id] = {"error": str(fnf)}
                break
            except RuntimeError as rte:  # Catch runtime errors (e.g., CUDA, data loading)
                # This already logs the stack trace
                logger.error(f"!!! Runtime error during '{method_name}': {rte}", exc_info=True)
                self.all_results[run_id] = {"error": str(rte)}
                break
            except Exception as e:  # Catch any other unexpected errors
                # This already logs the stack trace
                logger.critical(f"!!! An unexpected critical error occurred during '{method_name}': {e}", exc_info=True)
                self.all_results[run_id] = {"error": str(e), "traceback": logging.traceback.format_exc()}
                break  # Stop on critical errors

        total_duration = time.time() - start_time_total
        logger.info(f"Pipeline execution finished in {total_duration:.2f}s.")
        return self.all_results
