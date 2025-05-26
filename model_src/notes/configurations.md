# Comprehensive Guide to Configurable Parameters for the Image Classification Pipeline

This document outlines the various parameters that can be configured when using the `ClassificationPipeline` and `PipelineExecutor`. Parameters can be set as defaults when initializing the `PipelineExecutor` or `ClassificationPipeline`, overridden for specific method calls (like `single_train`, `cv_model_evaluation`) via a `params` dictionary, or included in a `param_grid` for hyperparameter search methods (`non_nested_grid_search`, `nested_grid_search`).

When used in `param_grid` for `GridSearchCV` or `RandomizedSearchCV`, skorch-specific parameters are prefixed (e.g., `optimizer__<param>`, `module__<param>`, `callbacks__<name>__<param>`).

## I. `PipelineExecutor` Initialization Parameters

These are passed when creating an instance of `PipelineExecutor`.

*   `dataset_path: Union[str, Path]`
    *   **Description:** Path to the root of the image dataset.
    *   **Required.**
*   `model_type: Union[str, ModelType]`
    *   **Description:** Type of model architecture to use. Can be a string (e.g., "cnn", "pvit") or a `ModelType` enum member.
    *   **Default:** `ModelType.CNN`
*   `model_load_path: Optional[Union[str, Path]]`
    *   **Description:** Optional path or S3 key to a pre-trained model's state_dict (`.pt` file) to load when the pipeline is initialized.
    *   **Default:** `None`
*   `artifact_repository: Optional[ArtifactRepository]`
    *   **Description:** An instance of an `ArtifactRepository` (e.g., `MinIORepository`, `LocalFileSystemRepository`) for saving/loading experiment artifacts. If `None`, file outputs are generally disabled or fall back to very basic local saving if `experiment_base_key_prefix` is treated as a local path by the pipeline.
    *   **Default:** `None`
*   `experiment_base_key_prefix: str`
    *   **Description:** The base prefix (for S3-like repositories) or base subfolder name (for local file repositories) under which all experiments for this executor instance will be organized. The pipeline will further append `dataset_name/model_type/timestamp_seed/` to this.
    *   **Default:** `"experiments"`
*   `results_detail_level: int`
    *   **Description:** Default level of detail for saving JSON results (0-3). See `ClassificationPipeline` for level details. Can be overridden per method.
    *   **Default:** `1`
*   `plot_level: int`
    *   **Description:** Default level for plotting results (0: none, 1: save, 2: save & show). Can be overridden per method.
    *   **Default:** `0`
*   `methods: Optional[List[Tuple[str, Dict[str, Any]]]]`
    *   **Description:** The sequence of pipeline methods to execute, each with its parameter dictionary.
    *   **Default:** `None` (no methods run by default).
*   `img_size: Tuple[int, int]`
    *   **Description:** Target image size (height, width) for transformations. Passed to `ImageDatasetHandler`.
    *   **Default:** `DEFAULT_IMG_SIZE` (e.g., `(224, 224)`)
*   `val_split_ratio: float`
    *   **Description:** Default ratio for train/validation splits if not specified otherwise. Passed to `ImageDatasetHandler`.
    *   **Default:** `0.2`
*   `test_split_ratio_if_flat: float`
    *   **Description:** Ratio for train/test split if dataset is FLAT. Passed to `ImageDatasetHandler`.
    *   **Default:** `0.2`
*   `augmentation_strategy: Union[str, AugmentationStrategy, Callable, None]`
    *   **Description:** Default data augmentation strategy for training. Can be a string name, `AugmentationStrategy` enum, or a custom `Callable` transform. Passed to `ImageDatasetHandler`.
    *   **Default:** `AugmentationStrategy.DEFAULT_STANDARD`
*   `show_first_batch_augmentation_default: bool`
    *   **Description:** Default for whether to plot the first augmented training/validation batch. Passed to `SkorchModelAdapter`.
    *   **Default:** `False`
*   `force_flat_for_fixed_cv: bool`
    *   **Description:** If True, treats FIXED datasets as FLAT for 'full' dataset CV methods. Passed to `ImageDatasetHandler`.
    *   **Default:** `False`
*   `optimizer: Union[str, Type[torch.optim.Optimizer]]`
    *   **Description:** Default optimizer to use. Can be a string (e.g., "adamw", "sgd") or a `torch.optim.Optimizer` class type.
    *   **Default:** `torch.optim.AdamW`
*   `lr: float`
    *   **Description:** Default initial learning rate.
    *   **Default:** `0.001`
*   `max_epochs: int`
    *   **Description:** Default maximum number of training epochs.
    *   **Default:** `20`
*   `batch_size: int`
    *   **Description:** Default batch size.
    *   **Default:** `32`
*   `patience: int`
    *   **Description:** Default patience for `EarlyStopping` callback.
    *   **Default:** `10`
*   `module__dropout_rate: Optional[float]`
    *   **Description:** Default dropout rate if the chosen model module (e.g., `SimpleCNN`, `PretrainedViT` head) accepts it in `__init__`.
    *   **Default:** `None`
*   `**kwargs`: Additional keyword arguments.
    *   **Description:** These are passed to `ClassificationPipeline.__init__` and then to `SkorchModelAdapter`. This is how you set most optimizer-specific parameters (e.g., `optimizer__weight_decay=0.01`) or specific callback parameters (e.g., `callbacks__default_lr_scheduler__factor=0.5`) as defaults for the pipeline instance.

## II. `ClassificationPipeline` Method Parameters

These are parameters specific to individual methods of the `ClassificationPipeline`. When calling methods via `PipelineExecutor`, these are provided in the dictionary for that method step.

### A. Common to Most Methods
*   `results_detail_level_override: Optional[int]`
    *   **Description:** Overrides the pipeline's default `results_detail_level` for this specific method call.
    *   **Default:** `None` (uses pipeline default).
*   `plot_level_override: Optional[int]`
    *   **Description:** Overrides the pipeline's default `plot_level` for this specific method call.
    *   **Default:** `None` (uses pipeline default).

### B. `single_train`
*   `params: Optional[Dict[str, Any]]`
    *   **Description:** A dictionary of parameters to override the pipeline's defaults for this specific training run. Can include Skorch params (`lr`, `batch_size`, `max_epochs`), optimizer type (`optimizer: torch.optim.SGD`), optimizer params (`optimizer__momentum`), module params (`module__<param>`), or callback configurations (e.g., `callbacks__default_lr_scheduler: LRScheduler(...)` or individual `callbacks__default_lr_scheduler__policy: 'StepLR'`, `callbacks__default_lr_scheduler__step_size: 10`).
    *   **Default:** `None`.
*   `val_split_ratio: Optional[float]`
    *   **Description:** Overrides the default validation split ratio for this run.
    *   **Default:** `None` (uses `ImageDatasetHandler` default).
*   `save_model: bool`
    *   **Description:** If True, saves the trained model's state_dict.
    *   **Default:** `True`.

### C. `single_eval`
*   (Uses `results_detail_level_override` and `plot_level_override` as common params).

### D. `predict_images`
*   `image_sources: List[Union[str, Path, Image.Image, bytes]]`
    *   **Description:** List of image sources (local paths, URLs, PIL Images, or bytes).
    *   **Required.**
*   `original_identifiers: Optional[List[str]]`
    *   **Description:** Optional list of original names or IDs for each image source, used in the output.
    *   **Default:** `None`.
*   `persist_prediction_artifacts: bool`
    *   **Description:** If True, saves prediction JSON and plots (if `plot_level > 0`) and LIME images (if MinIO repo and LIME enabled). If False, no artifacts for *this prediction run* are saved persistently.
    *   **Default:** `True`.
*   `generate_lime_explanations: bool`
    *   **Description:** If True, generates LIME explanations.
    *   **Default:** `False`.
*   `lime_num_features: int`
    *   **Description:** Number of top superpixels LIME should identify.
    *   **Default:** `5`.
*   `lime_num_samples: int`
    *   **Description:** Number of perturbed samples LIME generates per image.
    *   **Default:** `1000`.
*   `prediction_plot_max_cols: int`
    *   **Description:** Maximum columns in the `plot_predictions` grid.
    *   **Default:** `4`.

### E. `non_nested_grid_search` / `nested_grid_search`
*   `param_grid: Union[Dict[str, list], List[Dict[str, list]]]`
    *   **Description:** The parameter grid for `GridSearchCV` or distribution dictionary/list for `RandomizedSearchCV`. Optimizers and scheduler policies can be strings (e.g., "adamw", "StepLR") and will be resolved. Scheduler-specific parameters should use the `callbacks__<scheduler_name>__<param_name>` format (e.g., `callbacks__default_lr_scheduler__step_size`). If this is a list of dicts for `GridSearchCV`, each dict represents one full configuration to test, and values should be lists of one item (e.g. `'lr': [0.001]`, `'callbacks__default_lr_scheduler': [LRScheduler(...)]`).
    *   **Required.**
*   `cv: int` (for `non_nested_grid_search`) / `outer_cv: int`, `inner_cv: int` (for `nested_grid_search`)
    *   **Description:** Number of cross-validation folds.
    *   **Default:** `5` (outer), `3` (inner).
*   `internal_val_split_ratio: Optional[float]`
    *   **Description:** Ratio for Skorch's internal validation split within each CV fold's training data.
    *   **Default:** `None` (uses pipeline's `val_split_ratio`).
*   `n_iter: Optional[int]`
    *   **Description:** Number of iterations for `RandomizedSearchCV`. Required if `method='random'`.
    *   **Default:** `None`.
*   `method: str`
    *   **Description:** Search method, 'grid' or 'random'.
    *   **Default:** `'grid'`.
*   `scoring: str`
    *   **Description:** Scikit-learn scorer string or callable.
    *   **Default:** `'accuracy'`.
*   `save_best_model: bool` (for `non_nested_grid_search` only)
    *   **Description:** If True, saves the best model found and refit by the search.
    *   **Default:** `True`.

### F. `cv_model_evaluation`
*   `params: Optional[Dict[str, Any]]`
    *   **Description:** Fixed hyperparameters for this evaluation run. Overrides pipeline defaults. Optimizer and scheduler policy can be strings. To set specific scheduler parameters, provide a fully configured `LRScheduler` object under the `callbacks__default_lr_scheduler` key.
    *   **Default:** `None`.
*   `cv: int`
    *   **Description:** Number of CV folds.
    *   **Default:** `5`.
*   `evaluate_on: str`
    *   **Description:** Which data to use: 'full' (FLAT or forced-FLAT FIXED) or 'test' (uses test split).
    *   **Default:** `'full'`.
*   `internal_val_split_ratio: Optional[float]`
    *   **Description:** Skorch's internal validation split within each CV fold.
    *   **Default:** `None`.
*   `confidence_level: float`
    *   **Description:** For calculating confidence intervals of metrics.
    *   **Default:** `0.95`.


## III. Skorch Hyperparameters (Prefixes: `optimizer__`, `module__`, `callbacks__<name>__`)

These are set either in `PipelineExecutor`'s `**kwargs`, in the `params` dict for fixed runs, or in `param_grid` for search.

### A. Optimizer Parameters (`optimizer__<param_name>`)
   Passed directly to the chosen `torch.optim.Optimizer` constructor.
*   `optimizer__weight_decay: float` (e.g., for AdamW, SGD)
*   `optimizer__momentum: float` (e.g., for SGD)
*   `optimizer__nesterov: bool` (e.g., for SGD)
*   `optimizer__betas: Tuple[float, float]` (e.g., for Adam, AdamW)
*   `optimizer__eps: float` (e.g., for Adam, AdamW, RMSprop)
*   `optimizer__alpha: float` (e.g., for RMSprop)
*   *...and any other valid parameter for the chosen optimizer.*

### B. Module Parameters (`module__<param_name>`)
   Passed directly to the chosen PyTorch module's `__init__` (e.g., `PretrainedViT`, `SimpleCNN`).
*   `module__dropout_rate: float` (For `SimpleCNN`, `DiffusionClassifier` head)
*   **For `PretrainedViT`:**
    *   `module__vit_model_variant: str` (e.g., 'vit_b_16', 'vit_l_16')
    *   `module__pretrained: bool`
    *   `module__unfreeze_strategy: str` ('none', 'encoder_tail', 'full_encoder')
    *   `module__num_transformer_blocks_to_unfreeze: int` (for 'encoder_tail')
    *   `module__unfreeze_cls_token: bool`
    *   `module__unfreeze_pos_embedding: bool`
    *   `module__unfreeze_patch_embedding: bool`
    *   `module__unfreeze_encoder_layernorm: bool`
    *   `module__custom_head_hidden_dims: Optional[List[int]]`
    *   `module__head_dropout_rate: float`
*   **For `ViTFromScratch` (if used):**
    *   `module__img_size: int`
    *   `module__patch_size: int`
    *   `module__embed_dim: int`
    *   `module__depth: int`
    *   `module__num_heads: int`
    *   `module__mlp_ratio: float`
    *   `module__attention_dropout: float`
    *   `module__projection_dropout: float`
    *   `module__mlp_dropout: float`
    *   `module__head_hidden_dims: Optional[List[int]]`
    *   `module__head_dropout_rate: float`

### C. Callback Parameters (`callbacks__<callback_name_in_list>__<attribute_name>`)
   The default callbacks are named: `default_early_stopping`, `default_lr_scheduler`, `default_train_acc_scorer`, `file_log_table_cb`.

*   **Early Stopping (`callbacks__default_early_stopping__<param>`):**
    *   `callbacks__default_early_stopping__monitor: str` (e.g., 'valid_loss')
    *   `callbacks__default_early_stopping__patience: int`
    *   `callbacks__default_early_stopping__lower_is_better: bool`
    *   `callbacks__default_early_stopping__threshold: float`
    *   `callbacks__default_early_stopping__threshold_mode: str` ('rel' or 'abs')
*   **LR Scheduler (`callbacks__default_lr_scheduler__<param>`):**
    *   `callbacks__default_lr_scheduler__policy: str` (e.g., 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR')
    *   **For `LRScheduler` itself (monitor only for certain policies):**
        *   `callbacks__default_lr_scheduler__monitor: str` (e.g., 'valid_loss', for ReduceLROnPlateau)
        *   `callbacks__default_lr_scheduler__event_name: str` (e.g., 'on_epoch_end')
        *   `callbacks__default_lr_scheduler__step_every: str` ('epoch' or 'batch')
    *   **Parameters for the *underlying PyTorch scheduler* (passed by Skorch's LRScheduler):**
        *   **For `ReduceLROnPlateau`:**
            *   `callbacks__default_lr_scheduler__mode: str` ('min' or 'max')
            *   `callbacks__default_lr_scheduler__factor: float`
            *   `callbacks__default_lr_scheduler__patience: int` (different from EarlyStopping patience)
            *   `callbacks__default_lr_scheduler__min_lr: float`
            *   `callbacks__default_lr_scheduler__threshold: float`
            *   `callbacks__default_lr_scheduler__cooldown: int`
        *   **For `StepLR`:**
            *   `callbacks__default_lr_scheduler__step_size: int`
            *   `callbacks__default_lr_scheduler__gamma: float`
        *   **For `MultiStepLR`:**
            *   `callbacks__default_lr_scheduler__milestones: List[int]`
            *   `callbacks__default_lr_scheduler__gamma: float`
        *   **For `CosineAnnealingLR`:**
            *   `callbacks__default_lr_scheduler__T_max: int` (Often number of epochs or total steps)
            *   `callbacks__default_lr_scheduler__eta_min: float`
        *   **For `ExponentialLR`:**
            *   `callbacks__default_lr_scheduler__gamma: float`
        *   *(Note: `verbose` for PyTorch schedulers can also be set this way)*

### D. General Skorch `NeuralNetClassifier` Parameters
   These can be set directly in `kwargs` or `params` dicts if not already direct arguments to `ClassificationPipeline`.
*   `criterion: Type[nn.Module]` (e.g., `nn.CrossEntropyLoss`)
*   `iterator_train__shuffle: bool`
*   `iterator_train__num_workers: int`, `iterator_valid__num_workers: int`
*   `iterator_train__pin_memory: bool`, `iterator_valid__pin_memory: bool`
*   `train_split: Optional[Callable]` (e.g., `skorch.dataset.ValidSplit(cv=0.15, stratified=True)`)
    * Note: `ClassificationPipeline` manages `train_split` internally for `single_train` and `non_nested_grid_search` based on `val_split_ratio` or `internal_val_split_ratio`. For `cv_model_evaluation` and `nested_grid_search`, the outer CV splitter is scikit-learn's, but the *inner* Skorch fits still use a `ValidSplit` for their own validation.
*   `warm_start: bool`
*   `verbose: int` (Controls skorch's training log verbosity, 0 for silent, 1 for progress bar, 2 for one line per epoch - your `FileLogTable` overrides some of this).

This list should cover the vast majority of parameters you'd want to configure or tune in your setup.