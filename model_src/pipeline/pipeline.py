import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import copy

from utils import logger, set_seed, setup_logger
from datasets import DatasetHandler
from models import get_model
from adapters import SkorchImageClassifier
from methods import METHOD_REGISTRY


class ClassificationPipeline:
    """
    Orchestrates the image classification workflow, chaining different methods.
    """

    def __init__(
            self,
            dataset_path: str,
            model_name: str,
            output_dir: str = "results",
            image_size: Tuple[int, int] = (224, 224),
            base_model_params: Optional[Dict[str, Any]] = None,  # Params for Skorch adapter (lr, optimizer, etc.)
            model_path_load: Optional[str] = None,  # Path to load a pre-trained skorch model state
            model_path_save: Optional[str] = None,  # Path to save the final model at the end
            seed: int = 42,
            use_gpu: bool = True,
    ):
        """
        Args:
            dataset_path (str): Path to the dataset.
            model_name (str): Name of the model architecture (e.g., 'dummycnn', 'vit').
            output_dir (str): Root directory to save results and logs.
            image_size (Tuple[int, int]): Target image size.
            base_model_params (Optional[Dict[str, Any]]): Base parameters for the SkorchImageClassifier
                                                           (e.g., lr, batch_size, max_epochs, patience).
                                                           These can be overridden by search methods.
            model_path_load (Optional[str]): Path to a '.pt' file to load initial model state
                                              (loads skorch state: model weights, optimizer, history etc.).
            model_path_save (Optional[str]): Path to save the final skorch model state after pipeline execution.
            seed (int): Global random seed.
            use_gpu (bool): Whether to attempt using GPU if available.
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.output_base_dir = output_dir
        self.image_size = image_size
        self.base_model_params = base_model_params if base_model_params else {}
        self.model_path_load = model_path_load
        self.model_path_save = model_path_save
        self.seed = seed
        self.use_gpu = use_gpu

        self.pipeline_steps: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}  # Store results from each step

        # --- Initialization ---
        set_seed(self.seed)
        # Setup logger within the output directory structure
        self.dataset_name = Path(dataset_path).name
        log_dir = os.path.join(self.output_base_dir, self.dataset_name, self.model_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = setup_logger(log_file=log_file, level=logging.DEBUG)

        self.logger.info(f"🚀 Initializing Classification Pipeline...")
        self.logger.info(f"Dataset: {self.dataset_path}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Output Dir: {self.output_base_dir}")
        self.logger.info(f"Seed: {self.seed}")

        # Load Dataset
        try:
            self.dataset_handler = DatasetHandler(self.dataset_path, self.image_size, self.seed)
            self.num_classes = self.dataset_handler.get_num_classes()
            self.class_names = self.dataset_handler.get_classes()
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize DatasetHandler: {e}", exc_info=True)
            raise  # Critical error, stop pipeline

        # Determine device
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
        self.logger.info(f"💻 Device: {self.device.upper()}")

        # Initialize Model Adapter (can be empty, pre-trained torch, or loaded skorch state)
        self.model_adapter: Optional[SkorchImageClassifier] = None
        try:
            self._initialize_model_adapter()
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model adapter: {e}", exc_info=True)
            raise

    def _initialize_model_adapter(self):
        """Initializes or loads the SkorchImageClassifier."""

        # Get transformations
        train_transform, val_transform = self.dataset_handler.get_transforms()

        # --- Separate model-specific params from skorch params ---
        # Default for pretrained if not specified by user
        model_specific_params = self.base_model_params.copy() # Make a copy to modify
        use_pretrained = model_specific_params.pop('pretrained', True) # Use pop to get value AND remove key
        # Add any other model-specific params here if needed and pop them

        # Base skorch parameters
        skorch_params = {
            'lr': 1e-4,
            'batch_size': 32,
            'max_epochs': 20,
            'patience': 5,  # Early stopping patience
            'optimizer': torch.optim.AdamW,
            'criterion': torch.nn.CrossEntropyLoss,
            'device': self.device,
            'train_transform': train_transform,
            'val_transform': val_transform,
            # Add other skorch defaults if needed
        }
        # Update with user-provided base params, EXCLUDING the ones we popped
        skorch_params.update(model_specific_params) # Now `pretrained` is not in here

        if self.model_path_load and os.path.exists(self.model_path_load):
            self.logger.info(f"💾 Loading model state from: {self.model_path_load}")
            # Instantiate underlying torch model first - DON'T use TIMM pretrained if loading state
            pytorch_model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)

            # Instantiate adapter BEFORE loading params
            self.model_adapter = SkorchImageClassifier(
                module=pytorch_model, **skorch_params  # Pass cleaned skorch_params
            )
            try:
                self.model_adapter.load_params(f_params=self.model_path_load)
                self.logger.info("✅ Skorch model state loaded successfully.")
                # Ensure the loaded model is on the correct device
                self.model_adapter.initialize()  # Re-initialize internal components
                self.model_adapter.module_.to(self.device)  # Move torch module
            except Exception as e:
                self.logger.error(f"❌ Failed to load skorch model state from {self.model_path_load}: {e}",
                                  exc_info=True)
                self.logger.warning("⚠️ Proceeding with a newly initialized model.")
                # Fallback: re-initialize with potentially pretrained weights if user intended
                pytorch_model = get_model(self.model_name, num_classes=self.num_classes,
                                          pretrained=use_pretrained)  # Use original intent
                self.model_adapter = SkorchImageClassifier(module=pytorch_model,
                                                           **skorch_params)  # Pass cleaned skorch_params

        else:
            if self.model_path_load:
                self.logger.warning(
                    f"⚠️ Model load path specified but not found: {self.model_path_load}. Initializing a new model.")
            # Initialize a new model
            self.logger.info(f"✨ Initializing NEW model: {self.model_name}...")
            # Use the extracted 'use_pretrained' value
            pytorch_model = get_model(self.model_name, num_classes=self.num_classes, pretrained=use_pretrained)
            self.model_adapter = SkorchImageClassifier(
                module=pytorch_model, **skorch_params  # Pass cleaned skorch_params
            )
            self.logger.info("✅ New model adapter created.")

    def add_method(self, method_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Adds a method step to the pipeline.

        Args:
            method_name (str): The name of the method (e.g., 'single_train', 'non_nested_cv').
                               Must be a key in `methods.METHOD_REGISTRY`.
            config (Optional[Dict[str, Any]]): Configuration specific to this method step.
                                                If None, defaults will be used.
        """
        if method_name not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method name: '{method_name}'. Available methods: {list(METHOD_REGISTRY.keys())}")

        step_config = config if config else {}
        step_config['name'] = method_name  # Ensure name is in config for logging/saving
        # Add dataset/model names to config for consistent saving paths within methods
        step_config['dataset_name'] = self.dataset_name
        step_config['model_name'] = self.model_name

        self.pipeline_steps.append({
            'method_name': method_name,
            'config': step_config,
        })
        self.logger.info(f"➕ Added method to pipeline: {method_name} with config: {step_config}")

    def run(self):
        """Executes the added pipeline steps sequentially."""
        self.logger.info(f"🏁 Starting pipeline execution with {len(self.pipeline_steps)} steps...")
        start_time = time.time()

        current_model_adapter = self.model_adapter  # Start with the initial adapter

        for i, step in enumerate(self.pipeline_steps):
            method_name = step['method_name']
            config = step['config']
            step_number = i + 1

            self.logger.info(f"\n--- Running Step {step_number}/{len(self.pipeline_steps)}: {method_name} ---")

            # --- Compatibility Checks ---
            dataset_structure = self.dataset_handler.get_dataset_structure()
            if method_name == 'cv_evaluation' and dataset_structure == 'FIXED':
                self.logger.error(
                    f"❌ Skipping step {step_number} ('{method_name}'): Incompatible with FIXED dataset structure.")
                self.results[f'step_{step_number}_{method_name}'] = {'status': 'skipped',
                                                                     'reason': 'Incompatible dataset structure'}
                continue
            if method_name == 'single_eval' and dataset_structure == 'FLAT':
                # Warn, but allow trying (get_test_data will return None inside the method)
                self.logger.warning(
                    f"⚠️ Running step {step_number} ('{method_name}') with FLAT dataset. Method expects a test set, which is undefined for FLAT.")
            # Add more checks if needed (e.g., needing a trained model for eval)
            if method_name in ['single_eval'] and (
                    current_model_adapter is None or not current_model_adapter.initialized_):
                self.logger.error(
                    f"❌ Skipping step {step_number} ('{method_name}'): Requires a fitted/initialized model, but none is available.")
                self.results[f'step_{step_number}_{method_name}'] = {'status': 'skipped',
                                                                     'reason': 'Requires fitted model'}
                continue

            # Get the method function
            method_func = METHOD_REGISTRY[method_name]

            # Execute the method
            try:
                # Pass a deep copy of the adapter to search methods to avoid modifying the original unintentionally
                # For train/eval, pass the current adapter directly
                if method_name in ['non_nested_cv', 'nested_cv', 'cv_evaluation']:
                    # These methods might modify or clone the adapter internally
                    # Pass a fresh instance or ensure they handle state correctly
                    # Let's create a fresh instance with the same params for search/CV eval
                    temp_pytorch_model = get_model(self.model_name, self.num_classes,
                                                   pretrained=False)  # Don't reuse weights unless intended
                    adapter_for_method = SkorchImageClassifier(module=temp_pytorch_model,
                                                               **current_model_adapter.get_params(deep=False))
                    adapter_for_method.initialize()  # Must initialize skorch components
                else:
                    # For single_train, single_eval, pass the current adapter
                    adapter_for_method = current_model_adapter

                if adapter_for_method is None:
                    raise RuntimeError("Model adapter is None before executing method.")

                # Execute the method function
                step_result = method_func(
                    model_adapter=adapter_for_method,  # Pass the appropriate adapter instance
                    dataset_handler=self.dataset_handler,
                    method_config=config,
                    output_base_dir=self.output_base_dir,
                    seed=self.seed
                )

                # Store result and potentially update the main model adapter
                result_key = f'step_{step_number}_{method_name}'
                self.results[result_key] = step_result

                if step_result is not None:
                    self.logger.info(f"✅ Step {step_number} ('{method_name}') completed successfully.")
                    # If the method returns a trained model adapter, update the pipeline's current adapter
                    if isinstance(step_result, SkorchImageClassifier):
                        current_model_adapter = step_result
                        self.logger.info(f"🔄 Pipeline model adapter updated by '{method_name}'.")
                    elif isinstance(step_result, dict) and 'error' in step_result:
                        self.logger.error(
                            f"❌ Step {step_number} ('{method_name}') finished with an error: {step_result['error']}")
                    # else: method returned metrics or scores (dict)
                else:
                    # Method failed or returned None explicitly (e.g., skipped due to incompatibility)
                    if f'step_{step_number}_{method_name}' not in self.results:  # Check if reason wasn't already logged
                        self.logger.warning(
                            f"⚠️ Step {step_number} ('{method_name}') did not return expected results or failed.")
                        self.results[result_key] = {'status': 'failed or no return'}


            except Exception as e:
                self.logger.error(f"❌ Critical error during step {step_number} ('{method_name}'): {e}", exc_info=True)
                self.results[f'step_{step_number}_{method_name}'] = {'status': 'critical_error', 'error': str(e)}
                # Optionally stop the pipeline on critical errors
                # break

        # --- Finalization ---
        end_time = time.time()
        self.logger.info(f"\n🏁 Pipeline execution finished in {(end_time - start_time):.2f} seconds.")

        # Save the final model state if requested and available
        if self.model_path_save and current_model_adapter is not None and current_model_adapter.initialized_:
            self.logger.info(f"💾 Saving final model state to: {self.model_path_save}")
            save_dir = os.path.dirname(self.model_path_save)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            try:
                current_model_adapter.save_params(f_params=self.model_path_save)
                self.logger.info("✅ Final model saved successfully.")
            except Exception as e:
                self.logger.error(f"❌ Failed to save final model to {self.model_path_save}: {e}")
        elif self.model_path_save:
            self.logger.warning(
                f"⚠️ Skipping final model save: No valid trained model available or path not specified correctly.")

        # Log all collected results (optional)
        # self.logger.info("\n--- Pipeline Results Summary ---")
        # self.logger.info(json.dumps(self.results, indent=4)) # Can be very verbose

        return self.results  # Return collected results
