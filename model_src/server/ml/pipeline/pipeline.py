import contextlib
import io  # For capturing print output to a string
import json
import re
from datetime import datetime
from numbers import Number
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Callable, Any, Type, Optional, Union

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, make_scorer,
    roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, train_test_split, PredefinedSplit
)
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from torch.optim import AdamW, Adam, SGD

from .param_grid_utils import expand_hyperparameter_grid, parse_fixed_hyperparameters, DEFAULT_LR_SCHEDULER_NAME
from ..architectures import ModelType
from ..config import RANDOM_SEED, DEVICE, DEFAULT_IMG_SIZE, AugmentationStrategy
from ..dataset_utils import ImageDatasetHandler, DatasetStructure, PathImageDataset
from ..logger_utils import logger
from ..plotter import _save_figure_or_show, ResultsPlotter
from ..skorch_utils import SkorchModelAdapter
from ..skorch_utils import get_default_callbacks
from ...persistence import MinIORepository, LocalFileSystemRepository
from ...persistence.artifact_repo import ArtifactRepository

try:
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    LimeImageExplainer = None
    mark_boundaries = None

try:
    from skimage.segmentation import mark_boundaries
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    mark_boundaries = None


class ClassificationPipeline:
    """
    Manages image classification: data loading (paths), model selection,
    training, tuning, evaluation. Uses SkorchModelAdapter with dynamic transforms.
    """

    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: ModelType = ModelType.CNN,
                 model_load_path: Optional[Union[str, Path]] = None,
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 artifact_repository: Optional[ArtifactRepository] = None,
                 experiment_base_key_prefix: Optional[str] = "experiments",
                 results_detail_level: int = 1,
                 plot_level: int = 0,
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 augmentation_strategy: Union[str, AugmentationStrategy, Callable, None] = AugmentationStrategy.DEFAULT_STANDARD,
                 show_first_batch_augmentation_default: bool = False,
                 use_offline_augmented_data: bool = False,
                 force_flat_for_fixed_cv: bool = False,
                 optimizer: Union[str, Type[torch.optim.Optimizer]] = AdamW,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 module__dropout_rate: Optional[float] = None,

                 # --- LR Scheduler Default Configuration ---
                 lr_scheduler_policy_default: str = 'ReduceLROnPlateau',

                 # --- Catch-all for other skorch/optimizer/module params ---
                 **kwargs  # <<< Will capture optimizer__weight_decay, optimizer__momentum, etc.
                 ):
        """
            Initializes the Classification Pipeline.

            Args:
                dataset_path: Path to the root of the image dataset.
                model_type: Type of model to use for classification.
                    Options are 'cnn', 'simple_vit', 'flexible_vit', or 'diffusion'.
                model_load_path: Optional path to pre-trained model weights (.pt file) to load.
                img_size: Target image size for transformations (height, width).
                results_detail_level: Controls the verbosity of saved JSON results.
                    This is the default level for the pipeline, which can be overridden
                    per method call via the `results_detail_level_override` parameter in
                    individual pipeline methods.
                    Levels:

                    - 0: No JSON results file is saved for any method. Only summary CSV is updated.

                    - 1 (Basic Summary): Saves key metrics (overall accuracy, macro averages), best
                      parameters from tuning, summary of CV scores (e.g., mean/std test scores
                      from GridSearchCV), and paths to saved models. Excludes detailed lists like
                      full epoch histories, per-class metric arrays, y_true/pred/score arrays,
                      full GridSearchCV cv_results, and per-batch training data.

                    - 2 (Detailed Epoch-Level): Includes everything from Level 1, plus:
                        - Full epoch-by-epoch training histories (without per-batch data).
                        - Detailed per-class metrics.
                        - Raw `y_true`, `y_pred`, `y_score` arrays from evaluations.
                        - Data points for ROC and Precision-Recall curves.
                        - Full `cv_results` from GridSearchCV.
                        - `full_params_used` and `method_params_used`.

                    - 3 (Full Detail including Batch Data): Includes everything from Level 2, and
                      also preserves per-batch training/validation data (e.g., loss, batch size)
                      if present within the skorch History objects' epoch entries.
                plot_level: Default level for plotting results after methods run.
                    This can be overridden per method call via the `plot_level` parameter
                    in individual pipeline methods.
                    Levels:
                    - 0: No plotting is performed automatically after method execution.
                    - 1: Plots are generated and saved to files in a subdirectory next
                         to the results JSON.
                    - 2: Plots are generated, saved to files, AND displayed interactively
                         (using `plt.show()`). Requires a graphical backend.
                val_split_ratio: Default ratio for splitting train+validation data into
                                 training and validation sets for methods like `single_train` or
                                 as the internal validation split within skorch/CV folds if not
                                 otherwise specified.
                test_split_ratio_if_flat: Ratio for splitting a FLAT dataset into
                                          train+validation and test sets. Ignored for FIXED datasets.
                force_flat_for_fixed_cv: If True, treats a FIXED dataset structure (train/test splits)
                                         as a single pool of data for CV methods that operate on the
                                         'full' dataset (e.g., nested_grid_search, cv_model_evaluation
                                         with evaluate_on='full'). Use with caution.
                lr: Default learning rate for the optimizer.
                max_epochs: Default maximum number of training epochs.
                batch_size: Default batch size for training and evaluation.
                patience: Default patience for EarlyStopping callback.
                module__dropout_rate: Optional default dropout rate for model modules
                                      (if the model architecture supports it via __init__).
        """
        self.dataset_path = Path(dataset_path).resolve()
        # Ensure model_type is an instance of ModelType Enum
        if isinstance(model_type, str):
            try:
                self.model_type = ModelType(model_type)  # Convert string to Enum member
            except ValueError:
                raise ValueError(
                    f"Invalid model_type string: '{model_type}'. "
                    f"Valid types are: {[mt.value for mt in ModelType]}"
                )
        elif isinstance(model_type, ModelType):
            self.model_type = model_type
        else:
            raise TypeError(f"model_type must be a string or ModelType enum member, got {type(model_type)}")
        self.force_flat_for_fixed_cv = force_flat_for_fixed_cv
        self.results_detail_level = results_detail_level
        self.plot_level = plot_level
        self.augmentation_strategy = augmentation_strategy
        self.show_first_batch_augmentation_default = show_first_batch_augmentation_default
        self.artifact_repo : Optional[ArtifactRepository] = artifact_repository
        self.experiment_run_key_prefix: Optional[str] = experiment_base_key_prefix

        # --- LOGGING CAN NOW HAPPEN RELIABLY ---
        logger.info(f"Initializing Classification Pipeline:") # This will now use the configured logger
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Model Type: {self.model_type.value}")

        if self.artifact_repo and self.experiment_run_key_prefix:
            logger.info(f"  Artifact base key prefix for this run: {self.experiment_run_key_prefix} (using {type(self.artifact_repo).__name__})")
        else:
            logger.info("  Artifact repository not configured or base prefix missing. File outputs might be limited.")

        logger.info(f"  Default Augmentation Strategy: {str(augmentation_strategy)}") # Log received strategy
        logger.info(f"  Default Show First Batch Aug: {show_first_batch_augmentation_default}")

        self.dataset_handler = ImageDatasetHandler(
            root_path=self.dataset_path, img_size=img_size,
            val_split_ratio=val_split_ratio, test_split_ratio_if_flat=test_split_ratio_if_flat,
            augmentation_strategy=self.augmentation_strategy, use_offline_augmented_data=use_offline_augmented_data,
            force_flat_for_fixed_cv=self.force_flat_for_fixed_cv
        )

        if self.artifact_repo and self.experiment_run_key_prefix:
            logger.info(
                f"  Artifact base key prefix for this run: {self.experiment_run_key_prefix} (using {type(self.artifact_repo).__name__})")
        elif self.artifact_repo and not self.experiment_run_key_prefix:  # Repo exists, but no prefix provided by executor
            logger.warning(
                "  Artifact repository configured, but no experiment_base_key_prefix provided from executor. Specific run outputs might not be grouped correctly.")
            # You might decide to create a fallback prefix here based on timestamp if this case is valid
            # For now, this means artifact keys will be simpler (e.g. just run_id/filename)
        else:  # No repo
            logger.info("  Artifact repository not configured. File outputs will be disabled.")

        self.optimizer_type_config = optimizer # Store the configured optimizer (string or Type)
        self.lr_config = lr # Store configured lr

        # Resolve optimizer string to type if needed
        actual_optimizer_type: Type[torch.optim.Optimizer]
        if isinstance(optimizer, str):
            opt_lower = optimizer.lower()
            if opt_lower == "adamw":
                actual_optimizer_type = AdamW
            elif opt_lower == "adam":
                actual_optimizer_type = Adam
            elif opt_lower == "sgd":
                actual_optimizer_type = SGD
            # Add more optimizers here
            else:
                raise ValueError(f"Unsupported optimizer string: '{optimizer}'. Choose from 'adamw', 'adam', 'sgd'.")
        elif issubclass(optimizer, torch.optim.Optimizer):  # Check if it's a torch.optim.Optimizer subclass
            actual_optimizer_type = optimizer
        else:
            raise TypeError(f"Optimizer must be a string or a torch.optim.Optimizer type, got {type(optimizer)}")

        logger.info(f"  Using Optimizer: {actual_optimizer_type.__name__}")

        model_class = self._get_model_class(self.model_type)

        self.patience_default = patience
        self.lr_scheduler_policy_default = lr_scheduler_policy_default
        default_callbacks = get_default_callbacks(
            early_stopping_patience=self.patience_default,  # patience is from __init__ arg
            lr_scheduler_policy=self.lr_scheduler_policy_default,
            patience=2,  # Default patience for LR scheduler
        )

        module_params = {}
        if module__dropout_rate is not None: module_params['module__dropout_rate'] = module__dropout_rate
        module_params["module__num_classes"] = self.dataset_handler.num_classes

        self.model_adapter_config = {
            'module': model_class,
            'criterion': nn.CrossEntropyLoss,
            'optimizer': actual_optimizer_type,
            'lr': lr, 'max_epochs': max_epochs, 'batch_size': batch_size, 'device': DEVICE,
            'callbacks': default_callbacks,
            'patience_cfg': patience, 'monitor_cfg': 'valid_loss',
            'lr_policy_cfg': 'ReduceLROnPlateau', 'lr_patience_cfg': 5,
            'train_transform': self.dataset_handler.get_train_transform(),
            'valid_transform': self.dataset_handler.get_eval_transform(),
            'show_first_batch_augmentation': self.show_first_batch_augmentation_default,
            'use_offline_augmented_data': use_offline_augmented_data, # Pass to adapter
            'dataset_handler_ref': self.dataset_handler,            # Pass reference to adapter
            'classes': np.arange(self.dataset_handler.num_classes),
            'verbose': 0, # Default verbosity for adapter itself
            **module_params
        }

        # Add any other kwargs (like optimizer__weight_decay, optimizer__momentum)
        # These are passed directly to SkorchModelAdapter which passes them to torch.optim.Optimizer
        self.model_adapter_config.update(kwargs) # TODO - refactor this to accept only known kwargs

        init_config_for_adapter = self.model_adapter_config.copy()
        init_config_for_adapter.pop('patience_cfg', None); init_config_for_adapter.pop('monitor_cfg', None)
        init_config_for_adapter.pop('lr_policy_cfg', None); init_config_for_adapter.pop('lr_patience_cfg', None)

        # Important: Pass the base config to the adapter, not the processed one
        # Skorch handles parameter setting via set_params
        self.model_adapter = SkorchModelAdapter(**init_config_for_adapter)
        logger.info(f"  Model Adapter: Initialized with {model_class.__name__}")

        if model_load_path:
            self.load_model(model_load_path)
        # logger.info(f"Pipeline initialized successfully.") # Logged by executor

    def _get_s3_object_key(self, run_id: str, filename: str, sub_folder: Optional[str] = None) -> Optional[str]:
        """
        Constructs a full S3 object key.
        Args:
            run_id: The specific ID for the method run (e.g., "single_train_0").
            filename: The actual name of the file (e.g., "results.json", "model.pt").
            sub_folder: Optional sub-folder within the run_id directory (e.g., "single_train_plots").
        Returns:
            Full S3 object key as a string with forward slashes, or None if base prefix is not set.
        """
        if not self.experiment_run_key_prefix:
            return None

        if sub_folder:
            key_path = PurePath(self.experiment_run_key_prefix) / run_id / sub_folder / filename
        else:
            key_path = PurePath(self.experiment_run_key_prefix) / run_id / filename

        return key_path.as_posix()  # Ensure forward slashes for S3

    @staticmethod
    def _get_model_class(model_type_enum: ModelType) -> Type[nn.Module]:
        model_class = model_type_enum.get_model_class()
        if model_class is None:
            # This should ideally not happen if model_type_enum is validated
            raise ValueError(f"Unsupported model type: '{model_type_enum.value}'.")
        return model_class

    # In ClassificationPipeline class:
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_score: Optional[np.ndarray] = None,
                         detailed: bool = False) -> Dict[str, Any]:
        if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
        if y_score is not None and not isinstance(y_score, np.ndarray): y_score = np.array(y_score)

        metrics: Dict[str, Any] = {}  # Start empty
        class_metrics: Dict[str, Dict[str, float]] = {}  # Store per-class metrics here
        macro_metrics: Dict[str, float] = {}  # Store macro averages here
        detailed_data: Dict[str, Any] = {}  # Store detailed data here

        all_class_names = self.dataset_handler.classes
        num_classes_total = self.dataset_handler.num_classes
        if not all_class_names:
            logger.warning("Cannot compute metrics: class names not available.")
            return {'error': 'Class names missing'}

        # --- Overall Accuracy ---
        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)

        # --- Per-Class Metrics ---
        present_class_labels = np.unique(np.concatenate((y_true, y_pred)))  # Consider labels in both true and pred
        all_precisions, all_recalls, all_specificities, all_f1s = [], [], [], []
        all_roc_aucs, all_pr_aucs = [], []
        # For detailed results
        all_roc_curves = {}  # Store {class_name: {'fpr': list, 'tpr': list, 'thresholds': list}}
        all_pr_curves = {}  # Store {class_name: {'precision': list, 'recall': list, 'thresholds': list}}

        can_compute_auc = y_score is not None and len(y_score.shape) == 2 and y_score.shape[
            1] == num_classes_total and len(y_score) == len(y_true)
        if y_score is not None and not can_compute_auc:
            logger.warning(f"y_score shape incompatible. Cannot compute AUCs.")

        for i, class_name in enumerate(all_class_names):
            class_label = self.dataset_handler.class_to_idx.get(class_name, i)
            # Check if class actually present in y_true for some metrics
            is_present = class_label in np.unique(y_true)
            # Check if class present in y_true OR y_pred for basic metrics
            is_present_or_predicted = class_label in present_class_labels

            if not is_present_or_predicted:
                # Class completely absent, record NaNs for basic metrics
                class_metrics[class_name] = {'precision': np.nan, 'recall': np.nan, 'specificity': np.nan,
                                             'f1': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan}
                all_precisions.append(np.nan)
                all_recalls.append(np.nan)
                all_specificities.append(np.nan)
                all_f1s.append(np.nan)
                all_roc_aucs.append(np.nan)
                all_pr_aucs.append(np.nan)
                if detailed:  # Add empty curve data if detailed
                    all_roc_curves[class_name] = {'fpr': [], 'tpr': [], 'thresholds': []}
                    all_pr_curves[class_name] = {'precision': [], 'recall': [], 'thresholds': []}
                continue

            true_is_class = (y_true == class_label)
            pred_is_class = (y_pred == class_label)

            precision = precision_score(true_is_class, pred_is_class, zero_division=0)
            recall = recall_score(true_is_class, pred_is_class, zero_division=0)  # Sensitivity
            f1 = f1_score(true_is_class, pred_is_class, zero_division=0)
            # Specificity = TN / (TN + FP) = Recall of negative class
            specificity = recall_score(~true_is_class, ~pred_is_class, zero_division=0)

            roc_auc, pr_auc = np.nan, np.nan
            roc_curve_data = {'fpr': [], 'tpr': [], 'thresholds': []}
            pr_curve_data = {'precision': [], 'recall': [], 'thresholds': []}

            if can_compute_auc and is_present:  # Need true class present for meaningful AUC/curves
                score_for_class = y_score[:, class_label]
                if len(np.unique(true_is_class)) > 1:  # AUC/curves require both +ve/-ve samples
                    try:
                        roc_auc = roc_auc_score(true_is_class, score_for_class)
                    except ValueError:
                        pass  # Ignore if only one class present after all
                    except Exception as e:
                        logger.warning(f"ROC AUC Error (Class {class_name}): {e}")

                    try:
                        prec, rec, pr_thresh = precision_recall_curve(true_is_class, score_for_class)
                        order = np.argsort(rec)  # Sort by recall for AUC calc
                        pr_auc = auc(rec[order], prec[order])
                        if detailed:
                            pr_curve_data['precision'] = prec.tolist()
                            pr_curve_data['recall'] = rec.tolist()
                            # Thresholds might be one less
                            pr_curve_data['thresholds'] = pr_thresh.tolist() if pr_thresh is not None else []
                    except ValueError:
                        pass
                    except Exception as e:
                        logger.warning(f"PR AUC Error (Class {class_name}): {e}")

                    if detailed:
                        try:
                            fpr, tpr, roc_thresh = roc_curve(true_is_class, score_for_class)
                            roc_curve_data['fpr'] = fpr.tolist()
                            roc_curve_data['tpr'] = tpr.tolist()
                            roc_curve_data['thresholds'] = roc_thresh.tolist()
                        except ValueError:
                            pass
                        except Exception as e:
                            logger.warning(f"ROC Curve Error (Class {class_name}): {e}")

            # Store per-class results
            class_metrics[class_name] = {
                'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1,
                'roc_auc': roc_auc, 'pr_auc': pr_auc
            }
            # Append for macro calculation
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_specificities.append(specificity)
            all_f1s.append(f1)
            all_roc_aucs.append(roc_auc)
            all_pr_aucs.append(pr_auc)
            # Store detailed curve data if requested
            if detailed:
                all_roc_curves[class_name] = roc_curve_data
                all_pr_curves[class_name] = pr_curve_data

        # --- Macro Averages ---
        macro_metrics['precision'] = float(np.nanmean(all_precisions))
        macro_metrics['recall'] = float(np.nanmean(all_recalls))
        macro_metrics['specificity'] = float(np.nanmean(all_specificities))
        macro_metrics['f1'] = float(np.nanmean(all_f1s))
        macro_metrics['roc_auc'] = float(np.nanmean(all_roc_aucs)) if can_compute_auc else np.nan
        macro_metrics['pr_auc'] = float(np.nanmean(all_pr_aucs)) if can_compute_auc else np.nan

        metrics['per_class'] = class_metrics
        metrics['macro_avg'] = macro_metrics

        # --- Add Detailed Data if Requested ---
        if detailed:
            detailed_data['y_true'] = y_true.tolist()  # Convert to list for JSON
            detailed_data['y_pred'] = y_pred.tolist()
            if y_score is not None:
                detailed_data['y_score'] = y_score.tolist()
            detailed_data['roc_curve_points'] = all_roc_curves
            detailed_data['pr_curve_points'] = all_pr_curves
            metrics['detailed_data'] = detailed_data

        logger.debug(
            f"Computed Metrics: Acc={metrics['overall_accuracy']:.4f}, Macro F1={metrics['macro_avg']['f1']:.4f}")
        return metrics

    def _save_results(self,
                      results_data: Dict[str, Any],
                      method_name: str,
                      run_id: str,
                      method_params: Optional[Dict[str, Any]] = None,
                      results_detail_level: Optional[int] = None
                      ) -> Optional[Path]:
        if not self.artifact_repo or not self.experiment_run_key_prefix:
            logger.info(f"Artifact saving disabled for {run_id} (no repository/base_prefix). Results in memory only.")
            return None

        saved_artifact_identifier: Optional[str] = None

        current_detail_level = self.results_detail_level
        if results_detail_level is not None:
            current_detail_level = results_detail_level
            logger.debug(f"Results detail level overridden for this run to: {current_detail_level}")
        else:
            logger.debug(f"Using pipeline results detail level for this run: {current_detail_level}")

        # --- LEVEL 0: NO JSON SAVING, ONLY SUMMARY CSV ---
        if current_detail_level == 0:
            logger.info(f"Results detail level 0: Skipping JSON artifact saving for {run_id}.")
        else:
            logger.debug(f"Saving results for {run_id} (detail level: {current_detail_level})")
            results_to_save: Dict[str, Any] = {}

            # --- Level 1: Basic Information (Always included if saving JSON) ---
            # These are generally small and essential summary items.
            level_1_keys = [
                'method', 'run_id', 'params',  # Core identifiers and method params
                'overall_accuracy', 'macro_avg',  # Key metrics
                'best_params', 'best_score',  # From tuning methods
                'best_epoch', 'best_valid_metric_value', 'valid_metric_name',  # From single_train
                'train_loss_at_best', 'train_acc_at_best', 'valid_acc_at_best',  # From single_train best epoch
                'best_refit_model_epoch_info',  # From non_nested search refit
                'mean_test_accuracy', 'std_test_accuracy',  # From nested search
                'mean_test_f1_macro', 'std_test_f1_macro',  # From nested search
                'aggregated_metrics',  # From cv_model_evaluation
                'outer_cv_scores',  # Summary scores from nested CV
                'cv_fold_scores',  # Summary scores from cv_model_evaluation
                'evaluated_on', 'n_folds_requested', 'n_folds_processed', 'confidence_level',
                # Context for cv_model_eval
                'saved_model_path',  # Path to saved model if applicable
                'message', 'error'  # For status/error reporting
            ]
            for key in level_1_keys:
                if key in results_data:
                    results_to_save[key] = results_data[key]

            # Special handling for cv_results_summary from GridSearchCV for Level 1
            if 'cv_results' in results_data and isinstance(results_data['cv_results'], dict):
                summary_cv = {}
                for k, v_list in results_data['cv_results'].items():
                    if isinstance(v_list, (list, np.ndarray)) and len(v_list) > 0:
                        if k.startswith(('mean_test_', 'std_test_', 'mean_train_', 'std_train_', 'rank_test_')) or \
                                k in ['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']:
                            try:
                                if all(isinstance(item, (Number, np.number)) for item in v_list):
                                    summary_cv[k] = np.round(np.array(v_list), 5).tolist()
                            except (ValueError, TypeError):
                                pass
                    elif isinstance(v_list, (Number, np.number)):
                        summary_cv[k] = round(v_list, 5)
                if summary_cv:
                    results_to_save['cv_results_summary'] = summary_cv

            # --- Level 2: Detailed Metrics, Full Params, Epoch Histories (without batch data) ---
            if current_detail_level >= 2:
                level_2_keys_additive = [
                    'per_class',  # Detailed per-class metrics from _compute_metrics
                    'detailed_data',  # y_true, y_pred, y_score, roc/pr curve points from _compute_metrics
                    'full_params_used',  # Complete parameters used for a run
                    'method_params_used',  # Parameters passed to the method
                    'params_used_for_folds',  # For cv_model_evaluation
                    'cv_results',  # Full GridSearchCV output
                    'fold_detailed_results',  # Detailed metrics per fold from cv_model_evaluation
                    'outer_fold_best_params_found',  # Best params per outer fold in nested CV
                    # Histories (will be cleaned of batch data if level < 3)
                    'training_history',
                    'best_refit_model_history',
                    'outer_fold_best_model_histories',
                    'fold_training_histories',
                    'predictions'
                ]
                for key in level_2_keys_additive:
                    if key in results_data and key not in results_to_save:  # Add if not already there
                        results_to_save[key] = results_data[key]

            # --- Clean Batch Data from Histories if Level < 3 ---
            # This needs to be done *after* potentially adding history keys in Level 2
            if current_detail_level < 3:
                history_keys_to_clean_batches_from = [
                    'training_history', 'best_refit_model_history',
                    'outer_fold_best_model_histories', 'fold_training_histories'
                ]
                for hist_key in history_keys_to_clean_batches_from:
                    if hist_key in results_to_save and isinstance(results_to_save[hist_key], list):
                        cleaned_history_list = []
                        # Handle cases where history might be a list of lists (e.g. outer_fold_best_model_histories)
                        source_list = results_to_save[hist_key]
                        is_list_of_histories = all(
                            isinstance(item, list) or item is None for item in source_list) and any(
                            isinstance(item, list) for item in source_list)

                        if is_list_of_histories:
                            for single_history in source_list:
                                if isinstance(single_history, list):
                                    cleaned_single_history = []
                                    for epoch_data in single_history:
                                        if isinstance(epoch_data, dict):
                                            epoch_copy = {k: v for k, v in epoch_data.items() if
                                                          k != 'batches'}; cleaned_single_history.append(epoch_copy)
                                        else:
                                            cleaned_single_history.append(epoch_data)
                                    cleaned_history_list.append(cleaned_single_history)
                                else:
                                    cleaned_history_list.append(single_history)
                        else:
                            for epoch_data in source_list:
                                if isinstance(epoch_data, dict):
                                    epoch_copy = {k: v for k, v in epoch_data.items() if
                                                  k != 'batches'}; cleaned_history_list.append(epoch_copy)
                                else:
                                    cleaned_history_list.append(epoch_data)
                        results_to_save[hist_key] = cleaned_history_list

            # Level 3 includes everything already collected, including batch data in histories (as it wasn't stripped).

            artifact_key = self._get_s3_object_key(run_id, f"{method_name}_results.json")

            if self.artifact_repo.save_json(results_to_save, artifact_key):
                saved_artifact_identifier = artifact_key  # For MinIO, this is the object key
                logger.info(f"Results JSON saved via repository to: {saved_artifact_identifier}")
            else:
                logger.error(f"Failed to save results JSON via repository for {run_id} to key {artifact_key}.")

        return saved_artifact_identifier


    # --- Pipeline Methods ---

    def non_nested_grid_search(self,
                               param_grid: Union[Dict[str, list], List[Dict[str, list]]],
                               cv: int = 5,
                               internal_val_split_ratio: Optional[float] = None,
                               n_iter: Optional[int] = None,  # For RandomizedSearch
                               method: str = 'grid',  # 'grid' or 'random'
                               scoring: str = 'accuracy',  # Sklearn scorer string or callable
                               save_best_model: bool = True,
                               results_detail_level: Optional[int] = None,
                               plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs non-nested hyperparameter search (Grid/RandomizedSearchCV)
        using the train+validation data. Refits the best model on the train+val
        data and updates the pipeline_v1's main adapter. Does NOT evaluate on the
        test set itself. Works by passing paths directly.
        """
        method_lower = method.lower()
        # --- Generate unique run_id for this execution ---
        run_id = f"non_nested_{method_lower}_{datetime.now().strftime('%H%M%S')}"
        search_type = "GridSearchCV" if method_lower == 'grid' else "RandomizedSearchCV"
        logger.info(f"Performing non-nested {search_type} with {cv}-fold CV.")
        logger.info(f"Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None: raise ValueError("n_iter required for random search.")
        if method_lower not in ['grid', 'random']: raise ValueError(f"Unsupported search method: {method}.")

        # --- Expand the grid (handles optimizers and LRSchedulers) ---
        if isinstance(param_grid, dict):  # Single grid dictionary
            expanded_param_grid_for_search = expand_hyperparameter_grid(param_grid)
        elif isinstance(param_grid, list):  # List of grid dictionaries (for separate scenarios)
            expanded_param_grid_for_search = [expand_hyperparameter_grid(pg_dict) for pg_dict in param_grid]
        else:
            raise TypeError("param_grid must be a dictionary or list of dictionaries.")

        logger.info(
            f"Expanded Parameter Grid/Dist (for GridSearchCV):\n{json.dumps(expanded_param_grid_for_search, indent=2, default=str)}")

        # --- Get Data (Paths/Labels) ---
        # Only need trainval data for fitting the search
        X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels()
        if not X_trainval: raise RuntimeError("Train+validation data is empty.")
        logger.info(f"Using {len(X_trainval)} samples for Train+Validation in GridSearchCV.")

        # --- Determine & Validate Internal Validation Split ---
        default_internal_val_fallback = 0.15
        val_frac_to_use = internal_val_split_ratio if internal_val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. Using default fallback: {default_internal_val_fallback:.3f}")
             val_frac_to_use = default_internal_val_fallback
        logger.info(f"Skorch internal validation split configured: {val_frac_to_use * 100:.1f}% of each CV fold's training data.")
        train_split_config = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED)
        # --- End Determine & Validate ---

        # --- Setup Skorch Estimator for Search ---
        adapter_config = self.model_adapter_config.copy()
        adapter_config['train_split'] = train_split_config # Always set a valid split
        adapter_config['verbose'] = 0 # Show epoch table

        # Remove config keys not needed by SkorchModelAdapter init
        adapter_config.pop('patience_cfg', None) # TODO remove these 4 params entirely from all methods
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        estimator = SkorchModelAdapter(**adapter_config)

        # --- Setup Search ---
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        SearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        search_kwargs: Dict[str, Any] = {
            'estimator': estimator, 'cv': cv_splitter, 'scoring': scoring,
            'n_jobs': 1, 'verbose': 3, 'refit': True,  # Keep refit=True
            'return_train_score': True, 'error_score': 'raise'
        }

        if method_lower == 'grid':
            search_kwargs['param_grid'] = expanded_param_grid_for_search
        else:
            search_kwargs['param_distributions'] = expanded_param_grid_for_search
            search_kwargs['n_iter'] = n_iter
            search_kwargs['random_state'] = RANDOM_SEED
        search = SearchClass(**search_kwargs)

        logger.info(f"Fitting {SearchClass.__name__} on train+validation data (search verbose={search.verbose})...")

        # --- Describe number of combinations to search (adjusted for list of dicts) ---
        total_combinations = 0
        if method_lower == 'grid':
            if isinstance(param_grid, dict):
                total_combinations = int(np.prod([len(v) for v in param_grid.values() if isinstance(v, list)]))
            elif isinstance(param_grid, list):
                for pg_dict in param_grid:
                    if isinstance(pg_dict, dict):
                        total_combinations += int(np.prod([len(v) for v in pg_dict.values() if isinstance(v, list)]))
            logger.info(f"Total combinations to search (GridSearchCV): {total_combinations}")
        else:  # RandomizedSearchCV
            # If param_grid is a list of dicts for RandomizedSearchCV, n_iter applies per dict (subspace)
            # in newer scikit-learn versions. If it's a single dict, n_iter is the total.
            # The actual number of fits can be complex to pre-calculate precisely for RandomizedSearchCV with list of dicts
            # as it depends on scikit-learn version behavior.
            # The `n_candidates_` attribute of the fitted search object will tell the exact number.
            logger.info(f"Target number of iterations (RandomizedSearchCV): {n_iter}")

        # --- Capture stdout for scikit-learn's CV progress ---
        # Ensure contextlib is imported: import contextlib
        # Ensure io is imported: import io
        cv_progress_log = ""
        if search.verbose > 0: # Only capture if it's going to print
            string_io_buffer = io.StringIO()
            with contextlib.redirect_stdout(string_io_buffer): # Requires 'import contextlib'
                try:
                    search.fit(X_trainval, y=np.array(y_trainval))
                except Exception as e:
                    # Log intermediate output even if fit fails
                    cv_progress_log = string_io_buffer.getvalue()
                    logger.error(f"Error during {SearchClass.__name__}.fit: {e}", exc_info=True)
                    raise # Re-raise
            cv_progress_log = string_io_buffer.getvalue()
            string_io_buffer.close()
        else: # If search.verbose is 0, fit normally without capture
            search.fit(X_trainval, y=np.array(y_trainval))
        # --- End capture ---

        logger.info(f"Search completed.")

        # Log captured CV progress (if any)
        if cv_progress_log.strip():
            logger.info("--- GridSearchCV Internal CV Progress (Captured) ---")
            for line in cv_progress_log.strip().splitlines():
                logger.info(f"[SKL_CV] {line}")
            logger.info("--- End GridSearchCV Internal CV Progress ---")

        # --- Collect Results (Search Results Only) ---
        results = {
            'method': f"non_nested_{method_lower}_search",
            'run_id': run_id,
            'params': {'cv': cv, 'n_iter': n_iter if method_lower == 'random' else 'N/A', 'method': method_lower,
                       'scoring': scoring, 'internal_val_split_ratio': val_frac_to_use}, # Added internal_val_split_ratio
            'best_params': search.best_params_,
            'best_score': search.best_score_,  # This is the CV score on trainval
            'cv_results': search.cv_results_, # Contains scores, fit_times, etc. for each fold/param set
            'test_set_evaluation': {'message': 'Test set evaluation not performed in this method.'},
            'accuracy': np.nan,  # Indicate no test accuracy from this step
            'macro_avg': {}  # Indicate no test metrics from this step
        }

        # --- Store full config used by the winning estimator ---
        if hasattr(search, 'best_estimator_'):
             # Store best_params found by the search itself as the primary config
             results['full_params_used'] = search.best_params_.copy()
             # Augment with fixed params from original config that were not tuned
             for k,v in self.model_adapter_config.items():
                  # Add fixed params if they are not part of the tuned HPs and not complex objects
                  if not k.startswith(('optimizer__', 'module__', 'lr', 'batch_size', 'callbacks', 'train_transform', 'valid_transform')) and \
                     k not in results['full_params_used'] and isinstance(v, (str, int, float, bool, type(None))):
                        results['full_params_used'][k] = v
        else:
             results['full_params_used'] = {}


        # --- Update the pipeline's main adapter ---
        best_estimator_refit = None # Initialize
        if hasattr(search, 'best_estimator_'):
            best_estimator_refit = search.best_estimator_
            logger.info("Updating main pipeline_v1 adapter with the best model found and refit by GridSearchCV.")
            self.model_adapter = best_estimator_refit
            if not self.model_adapter.initialized_:
                 logger.warning("Refit best estimator seems not initialized, attempting initialize.")
                 try: self.model_adapter.initialize()
                 except Exception as init_err: logger.error(f"Failed to initialize refit estimator: {init_err}", exc_info=True)

            # --- Extract and log training history of best refit model (if available) ---
            if hasattr(best_estimator_refit, 'history_') and best_estimator_refit.history_:
                results['best_refit_model_history'] = best_estimator_refit.history_.to_list()
                # Log info about best epoch of refit model
                try:
                    refit_history = best_estimator_refit.history_
                    valid_loss_key_refit = 'valid_loss'  # Or whatever your early stopping monitors
                    if refit_history and valid_loss_key_refit in refit_history[0]: # Check if validation was run for refit
                        scores_refit = [epoch.get(valid_loss_key_refit, np.inf) for epoch in refit_history]
                        best_idx_refit = np.argmin(scores_refit)
                        best_epoch_hist_refit = refit_history[int(best_idx_refit)]
                        results['best_refit_model_epoch_info'] = {
                            'best_epoch': best_epoch_hist_refit.get('epoch'),
                            'best_valid_metric_value': float(best_epoch_hist_refit.get(valid_loss_key_refit, np.nan)),
                            'train_loss_at_best': float(best_epoch_hist_refit.get('train_loss', np.nan)),
                            'train_acc_at_best': float(best_epoch_hist_refit.get('train_acc', np.nan)),
                            'valid_acc_at_best': float(best_epoch_hist_refit.get('valid_acc', np.nan)),
                        }
                        logger.info(
                            f"Best refit model (on full train-val) converged at epoch: {results['best_refit_model_epoch_info']['best_epoch']}")
                    else:
                        logger.info(
                            "Best refit model was trained without a validation split (e.g. internal_val_split_ratio was 0 or invalid).")
                        if refit_history:
                            results['best_refit_model_epoch_info'] = {
                                'last_epoch': refit_history[-1].get('epoch'),
                                'last_train_loss': float(refit_history[-1].get('train_loss', np.nan)),
                                'last_train_acc': float(refit_history[-1].get('train_acc', np.nan)),
                            }

                except Exception as e:
                    logger.warning(f"Could not extract best epoch info from refit model history: {e}")
        else:
            logger.warning("GridSearchCV did not produce a 'best_estimator_'. Pipeline adapter not updated.")

        # --- Save Model (if requested and available) ---
        model_path_identifier = None
        arch_config_path_identifier = None  # For the new arch_config.json

        if save_best_model and best_estimator_refit is not None:
            if self.artifact_repo and self.experiment_run_key_prefix:
                try:
                    # --- NEW FILENAME LOGIC ---
                    model_type_short = self.model_type.value
                    run_type_short = "gridcv"
                    run_id_timestamp_part = run_id.split('_')[-1]  # From "non_nested_grid_TIMESTAMP"

                    best_cv_score = results.get('best_score', 0.0)
                    # Format to 2 decimal places, replace dot, remove leading zero
                    score_str = f"cvsc{best_cv_score:.2f}".replace('.', 'p').replace("0p", "p")

                    # Simplified param string (optional, can make filename long)
                    # best_params_short_list = []
                    # for k, v_param in sorted(results.get('best_params', {}).items())[:2]: # Max 2 params in name
                    #     k_short = k.split('__')[-1][:5] # Shorten key
                    #     v_str = str(v_param)[:5] # Shorten value
                    #     best_params_short_list.append(f"{k_short}{v_str}")
                    # params_filename_part = "_".join(best_params_short_list)
                    # params_filename_part = re.sub(r'[^\w_.-]', '', params_filename_part)
                    # model_filename_base = f"{model_type_short}_{run_type_short}_{params_filename_part}_{score_str}_{run_id_timestamp_part}"
                    # For simplicity, let's omit complex params from filename for grid search best model:
                    model_filename_base = f"{model_type_short}_{run_type_short}_{score_str}_{run_id_timestamp_part}"
                    # --- END NEW FILENAME LOGIC ---

                    model_pt_filename = f"{model_filename_base}.pt"
                    model_config_filename = f"{model_filename_base}_arch_config.json"

                    model_pt_object_key = self._get_s3_object_key(run_id, model_pt_filename)
                    model_config_object_key = self._get_s3_object_key(run_id, model_config_filename)

                    # 2. Save Model State Dictionary
                    model_state_dict = best_estimator_refit.module_.state_dict()
                    model_path_identifier = self.artifact_repo.save_model_state_dict(
                        model_state_dict, model_pt_object_key
                    )
                    if model_path_identifier:
                        logger.info(f"Best refit model state_dict saved via repository to: {model_path_identifier}")
                        results['saved_model_path'] = model_path_identifier  # Update results dict
                    else:
                        logger.error(
                            f"Failed to save best refit model state_dict for {run_id} to key {model_pt_object_key}.")
                        results['saved_model_path'] = None

                    # 3. Save Architectural Configuration
                    if model_path_identifier:  # Proceed only if .pt was saved
                        # The 'best_estimator_refit' is a SkorchModelAdapter instance.
                        # Its internal nn.Module is best_estimator_refit.module_
                        # The parameters used to *create* this specific module instance came from search.best_params_
                        # and the fixed parts of the adapter_config used to initialize the 'estimator' for GridSearchCV.

                        arch_config_to_save = {
                            'model_type': self.model_type.value,
                            'num_classes': self.dataset_handler.num_classes
                            # Add all 'module__' parameters from best_estimator_refit.get_params()
                            # These were the ones that resulted in the best model.
                        }

                        # Get effective parameters of the best refit estimator
                        # These include the 'module__xyz' parameters that defined its architecture
                        best_estimator_params = best_estimator_refit.get_params(deep=False)

                        for key, value in best_estimator_params.items():
                            if key.startswith('module__'):
                                # Special handling for types that are not JSON serializable by default
                                if isinstance(value, type):
                                    arch_config_to_save[
                                        key] = f"<class '{value.__module__}.{value.__name__}'>"  # Store as string
                                elif callable(value) and not isinstance(value, (nn.Module, torch.optim.Optimizer)):
                                    # For other callables like transform functions from dataset_handler,
                                    # it might be better to store a placeholder or a descriptive name.
                                    # For now, let's skip non-module/non-optimizer callables from module__
                                    # or convert them to string if simple.
                                    # However, 'module__' params should primarily be for nn.Module's __init__ args.
                                    # Transform functions are usually direct skorch params like 'train_transform'.
                                    pass
                                else:
                                    arch_config_to_save[key] = value
                            # Consider also saving key top-level skorch params if they define architecture,
                            # e.g., 'optimizer' if it was tuned and is a type rather than string.
                            # For now, focusing on module__ params.
                            # For HybridViT, for example, module__cnn_model_name, module__vit_model_variant etc.
                            # are critical.

                        arch_config_path_identifier = self.artifact_repo.save_json(
                            arch_config_to_save, model_config_object_key
                        )
                        if arch_config_path_identifier:
                            logger.info(
                                f"Architectural config for best model saved to: {arch_config_path_identifier}")
                            results['saved_model_arch_config_path'] = arch_config_path_identifier
                        else:
                            logger.error(
                                f"Failed to save architectural config for {run_id} to {model_config_object_key}.")
                            results['saved_model_arch_config_path'] = None

                except Exception as e:
                    logger.error(f"Failed to save best refit model or its config via repository: {e}",
                                 exc_info=True)
                    results['saved_model_path'] = None
                    results['saved_model_arch_config_path'] = None
            else:
                logger.warning(
                    f"Model saving skipped for {run_id}: no artifact repository or base key prefix configured.")
        elif save_best_model and best_estimator_refit is None:
            logger.warning(f"save_best_model=True for {run_id} but no best estimator was found/refit.")
            results['saved_model_path'] = None  # Ensure these keys exist even if saving fails
            results['saved_model_arch_config_path'] = None

        # If not saving, ensure keys are present but None
        if not save_best_model:
            results['saved_model_path'] = None
            results['saved_model_arch_config_path'] = None
        # --- End Save Model ---

        # --- Save Results JSON ---
        summary_params = results.get('params', {}).copy() # Get the method's specific params
        summary_params.update({f"best_{k}": v for k, v in results.get('best_params', {}).items() if isinstance(v, (str, int, float, bool))})
        json_artifact_key_or_path = self._save_results(
                            results_data=results,
                            method_name=f"non_nested_{method_lower}_search",
                            run_id=run_id,
                            method_params=summary_params,
                            results_detail_level=results_detail_level
                           )

        # --- Determine effective plot level ---
        current_plot_level = self.plot_level  # Start with pipeline default
        if plot_level is not None:
            current_plot_level = plot_level  # Use override if provided
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        # --- Plot results (conditionally) ---
        if current_plot_level > 0:
            plot_save_location_base: Optional[str] = None
            if self.artifact_repo and self.experiment_run_key_prefix:
                plot_save_location_base = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())

            if current_plot_level == 1 and not plot_save_location_base:
                logger.warning(f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no repository/base_key configured for saving.")
            else:
                logger.info(f"Plotting non_nested_cv results for {run_id} (plot level {current_plot_level}).")
                show_plots_flag = (current_plot_level == 2)
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_non_nested_cv_results(
                        results_input=results,
                        plot_save_dir_base=plot_save_location_base,
                        repository_for_plots=self.artifact_repo if plot_save_location_base else None,
                        show_plots=show_plots_flag
                    )
                except ImportError:
                     logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Plotting failed for {run_id}: {plot_err}", exc_info=True)

        logger.info(f"Non-nested {method_lower} search finished. Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"Best parameters found: {search.best_params_}")
        logger.info(f"Pipeline adapter has been updated with the best model refit on train+validation data.")

        return results

    def nested_grid_search(self,
                           param_grid: Union[Dict[str, list], List[Dict[str, list]]],
                           outer_cv: int = 5,
                           inner_cv: int = 3,
                           internal_val_split_ratio: Optional[float] = None,
                           n_iter: Optional[int] = None, # For RandomizedSearch
                           method: str = 'grid', # 'grid' or 'random'
                           scoring: str = 'accuracy', # Sklearn scorer string or callable
                           results_detail_level: Optional[int] = None,
                           plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs nested cross-validation for unbiased performance estimation.
        Uses the full dataset (respecting force_flat_for_fixed_cv).
        Passes paths to sklearn cross_validate. Skorch adapter handles transforms.
        """
        method_lower = method.lower()
        run_id = f"nested_{method_lower}_{datetime.now().strftime('%H%M%S')}"  # Generate ID
        search_type = "GridSearchCV" if method_lower == 'grid' else "RandomizedSearchCV"
        logger.info(f"Performing nested {search_type} search.")
        logger.info(f"  Outer CV folds: {outer_cv}, Inner CV folds: {inner_cv}")
        logger.info(f"  Parameter Grid/Dist for inner search:\n{json.dumps(param_grid, indent=2)}")
        logger.info(f"  Scoring Metric: {scoring}")

        # --- Expand the param_grid for the INNER search ---
        if isinstance(param_grid, dict):
            expanded_param_grid_for_inner_search = expand_hyperparameter_grid(param_grid)
        elif isinstance(param_grid, list):
            expanded_param_grid_for_inner_search = [expand_hyperparameter_grid(pg_dict) for pg_dict in param_grid]
        else:
            raise TypeError("param_grid for inner search must be a dictionary or list of dictionaries.")

        logger.debug(
            f"Expanded Parameter Grid/Dist for INNER search (for GridSearchCV):\n{json.dumps(expanded_param_grid_for_inner_search, indent=2, default=str)}")

        # --- Check Compatibility ---
        if self.dataset_handler.structure == DatasetStructure.FIXED and not self.force_flat_for_fixed_cv:
             # Provide a specific path for FIXED datasets without the flag
             raise ValueError(f"nested_grid_search requires a FLAT dataset structure "
                              f"or a FIXED structure with force_flat_for_fixed_cv=True.")

        # --- Standard Nested CV (FLAT or FIXED with force_flat_for_fixed_cv=True) ---
        logger.info("Proceeding with standard nested CV using the full dataset.")
        try:
            X_full, y_full = self.dataset_handler.get_full_paths_labels_for_cv()
            if not X_full: raise RuntimeError("Full dataset for CV is empty.")
            logger.info(f"Using {len(X_full)} samples for outer cross-validation.")
            y_full_np = np.array(y_full) # Needed for stratification
        except Exception as e:
            logger.error(f"Failed to get full dataset paths/labels for nested CV: {e}", exc_info=True)
            raise

        # --- Determine & Validate Internal Validation Split ---
        default_internal_val_fallback = 0.15
        val_frac_to_use = internal_val_split_ratio if internal_val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. "
                            f"Using default fallback: {default_internal_val_fallback:.3f} for inner loop fits.")
             val_frac_to_use = default_internal_val_fallback
        # --- End Determine & Validate ---

        logger.info(f"Inner loop Skorch validation split configured: {val_frac_to_use * 100:.1f}% of inner CV fold's training data.")
        train_split_config = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED)

        # --- Setup Inner Search Object ---
        adapter_config = self.model_adapter_config.copy()
        adapter_config['train_split'] = train_split_config # Always set a valid split
        adapter_config['verbose'] = 3

        # Remove config keys not needed by SkorchModelAdapter init
        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        base_estimator = SkorchModelAdapter(**adapter_config)

        inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_SEED)
        InnerSearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        inner_search_kwargs: Dict[str, Any] = {
            'estimator': base_estimator, 'cv': inner_cv_splitter, 'scoring': scoring,
            'n_jobs': 1, 'verbose': 3, 'refit': True, 'error_score': 'raise'
        }

        if method_lower == 'grid': inner_search_kwargs['param_grid'] = expanded_param_grid_for_inner_search
        else:
            inner_search_kwargs['param_distributions'] = expanded_param_grid_for_inner_search
            inner_search_kwargs['n_iter'] = n_iter
            inner_search_kwargs['random_state'] = RANDOM_SEED
        inner_search = InnerSearchClass(**inner_search_kwargs)

        # --- Setup Outer CV ---
        outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED + 1) # Different seed

        # Define multiple scorers for cross_validate
        scoring_dict = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            # Add others if needed (e.g., roc_auc requires predict_proba)
            # 'roc_auc_ovr': make_scorer(roc_auc_score, average='macro', multi_class='ovr', needs_proba=True)
        }

        # --- Run Nested CV using cross_validate ---
        logger.info(
            f"Running standard nested CV using cross_validate. Inner GridSearchCV verbose: {inner_search.verbose}")
        try:
            # --- Capture stdout ---
            cv_progress_log = ""
            if inner_search.verbose > 0:  # Capture if inner search will print
                string_io_buffer = io.StringIO()
                with contextlib.redirect_stdout(string_io_buffer):
                    try:
                        cv_results = cross_validate(
                            inner_search, X_full, y_full_np, cv=outer_cv_splitter, scoring=scoring_dict,
                            return_estimator=True, n_jobs=1, error_score='raise'
                            # cross_validate's own verbose controls joblib, not the inner_search prints directly for n_jobs=1
                        )
                    except Exception as e:
                        cv_progress_log = string_io_buffer.getvalue()
                        logger.error(f"Error during cross_validate: {e}", exc_info=True)
                        raise
                cv_progress_log = string_io_buffer.getvalue()
                string_io_buffer.close()
            else:
                cv_results = cross_validate(
                    inner_search, X_full, y_full_np, cv=outer_cv_splitter, scoring=scoring_dict,
                    return_estimator=True, n_jobs=1, error_score='raise'
                )
            # --- End capture ---

            logger.info("Nested cross-validation finished.")

            if cv_progress_log.strip():
                logger.info("--- Nested CV Inner Loop Progress (Captured) ---")
                for line in cv_progress_log.strip().splitlines():
                    logger.info(f"[NESTED_SKL_CV] {line}")
                logger.info("--- End Nested CV Inner Loop Progress ---")

            # --- Process and Save Results ---
            results: Dict[str, Any] = {
                'method': f"nested_{method_lower}_search",
                'run_id': run_id,
                'params': {
                    'outer_cv': outer_cv,
                    'inner_cv': inner_cv,
                    'n_iter': n_iter if method_lower=='random' else 'N/A',
                    'method': method_lower,
                    'scoring': scoring,
                    'forced_flat': self.force_flat_for_fixed_cv,
                    'internal_val_split_ratio': val_frac_to_use
                },
                'outer_cv_scores': {k: v.tolist() for k, v in cv_results.items() if k.startswith('test_')},
                'mean_test_accuracy': float(np.mean(cv_results['test_accuracy'])),
                'std_test_accuracy': float(np.std(cv_results['test_accuracy'])),
                'mean_test_f1_macro': float(np.mean(cv_results['test_f1_macro'])),
                'std_test_f1_macro': float(np.std(cv_results['test_f1_macro'])),
                # 'best_params_per_fold': "Estimators not returned" # Or return them if needed
            }

            # --- Extract histories and best params per fold ---
            if 'estimator' in cv_results:
                fold_histories_nested = []
                best_params_per_fold_nested = []
                inner_score_per_fold_nested = []
                for fold_idx, outer_fold_search_estimator in enumerate(cv_results['estimator']):
                    # History
                    if hasattr(outer_fold_search_estimator, 'best_estimator_') and \
                            hasattr(outer_fold_search_estimator.best_estimator_, 'history_') and \
                            outer_fold_search_estimator.best_estimator_.history_:
                        fold_histories_nested.append(outer_fold_search_estimator.best_estimator_.history_.to_list())
                    else:
                        fold_histories_nested.append(None)
                    # Best Params
                    if hasattr(outer_fold_search_estimator, 'best_params_'):
                        best_params_per_fold_nested.append(outer_fold_search_estimator.best_params_)
                    else:
                        best_params_per_fold_nested.append(None)
                    # Inner Best Score <<< NEW SECTION
                    if hasattr(outer_fold_search_estimator, 'best_score_'):
                        inner_score_per_fold_nested.append(outer_fold_search_estimator.best_score_)
                    else:
                        inner_score_per_fold_nested.append(np.nan)

                results['outer_fold_best_model_histories'] = fold_histories_nested
                results['outer_fold_best_params_found'] = best_params_per_fold_nested
                results['outer_fold_inner_cv_best_score'] = inner_score_per_fold_nested

            # For summary file
            results['accuracy'] = results['mean_test_accuracy']
            results['macro_avg'] = {'f1': results['mean_test_f1_macro']}
            method_name = results['method']

            # --- Save Results ---
            saved_json_path = self._save_results(results, f"nested_{method_lower}_search",
                                run_id=run_id,
                                method_params=results['params'],
                                results_detail_level=results_detail_level)

            # --- Determine effective plot level ---
            current_plot_level = self.plot_level  # Start with pipeline default
            if plot_level is not None:
                current_plot_level = plot_level  # Use override if provided
                logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

            # --- Plot results (conditionally) ---
            if current_plot_level > 0:
                # plot_save_dir_base will be the directory/S3_prefix for THIS SPECIFIC RUN's artifacts
                plot_save_dir_base_for_run: Optional[Union[str, Path]] = None

                if self.artifact_repo and self.experiment_run_key_prefix:
                    # If using a repository, construct the base key for this run's artifacts
                    plot_save_dir_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())
                else:
                    plot_save_dir_base_for_run = None
                    logger.warning(f"No artifact repo specified for this run.")

                # Condition to actually attempt saving plots to a file/repository:
                # Level 1 (save only) or Level 2 (save and show), AND a base location must exist.
                can_save_plots = (current_plot_level >= 1 and plot_save_dir_base_for_run is not None)
                should_show_plots_flag = (current_plot_level == 2)

                if not can_save_plots and current_plot_level == 1:
                    logger.warning(
                        f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no save location could be determined (e.g., no repository).")

                if can_save_plots or should_show_plots_flag:  # Proceed if saving OR showing
                    logger.info(f"Plotting {method_name} results for {run_id} (plot level {current_plot_level}).")
                    try:
                        from ..plotter import ResultsPlotter  # Ensure correct relative import
                        ResultsPlotter.plot_nested_cv_results(
                            results_input=results,
                            plot_save_dir_base=plot_save_dir_base_for_run,
                            repository_for_plots=self.artifact_repo if can_save_plots else None,
                            show_plots=should_show_plots_flag
                        )
                    except ImportError:
                        logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                    except Exception as plot_err:
                        logger.error(f"Plotting failed for {run_id}: {plot_err}", exc_info=True)

            logger.info(f"Nested CV Results (avg over {outer_cv} outer folds):")
            logger.info(f"  Mean Test Accuracy: {results['mean_test_accuracy']:.4f} +/- {results['std_test_accuracy']:.4f}")
            logger.info(f"  Mean Test Macro F1: {results['mean_test_f1_macro']:.4f} +/- {results['std_test_f1_macro']:.4f}")

            return results

        except Exception as e:
             logger.error(f"Standard nested CV failed: {e}", exc_info=True)
             # Return error information
             return {
                'method': f"nested_{method_lower}_search",
                'params': {'outer_cv': outer_cv, 'inner_cv': inner_cv, 'n_iter': n_iter if method_lower=='random' else 'N/A', 'method': method_lower, 'scoring': scoring},
                'error': str(e)
             }

    def cv_model_evaluation(self,
                            cv: int = 5,
                            evaluate_on: str = 'full',  # 'full' or 'test'
                            internal_val_split_ratio: Optional[float] = None,
                            params: Optional[Dict] = None,
                            confidence_level: float = 0.95,
                            results_detail_level: Optional[int] = None,
                            plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs K-Fold CV for evaluation using fixed hyperparameters.
        Can evaluate either on the 'full' dataset (trainval or combined)
        or only on the 'test' set. Uses Skorch adapter's internal
        ValidSplit for monitoring within each fold's training. Calculates CIs.

        Args:
            cv: Number of folds for cross-validation.
            evaluate_on: Which data split to use ('full' or 'test'). Default 'full'.
            internal_val_split_ratio: Fraction for internal validation split during fold training.
                                      Defaults to handler's val_split_ratio or 0.15 fallback.
            params: Dictionary of fixed hyperparameters to use for training each fold.
                    If None, uses defaults from pipeline_v1 config. Can be merged with
                    best_params from a previous step using executor logic.
            confidence_level: Confidence level for calculating CI (e.g., 0.95 for 95%).
        """
        logger.info(f"Performing {cv}-fold CV for evaluation with fixed parameters.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive).")

        # Inside cv_model_evaluation, after initial logger info:
        valid_eval_on = ['full', 'test']
        if evaluate_on not in valid_eval_on:
            raise ValueError(f"Invalid 'evaluate_on' value: '{evaluate_on}'. Must be one of {valid_eval_on}")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive).")

        run_id = f"cv_model_evaluation_{evaluate_on}_{datetime.now().strftime('%H%M%S')}"  # Include mode in ID

        # --- Get Data based on 'evaluate_on' ---
        if evaluate_on == 'full':
            if self.dataset_handler.structure == DatasetStructure.FIXED and not self.force_flat_for_fixed_cv:
                raise ValueError(
                    "cv_model_evaluation(evaluate_on='full') requires FLAT dataset or FIXED dataset with force_flat_for_fixed_cv=True.")
            try:
                X_selected, y_selected_list = self.dataset_handler.get_full_paths_labels_for_cv()
                logger.info(f"Using full dataset ({len(X_selected)} samples) for CV evaluation.")
            except Exception as e:
                logger.error(f"Failed to get full dataset paths/labels for CV evaluation: {e}", exc_info=True)
                raise
        elif evaluate_on == 'test':
            try:
                X_selected, y_selected_list = self.dataset_handler.get_test_paths_labels()
                logger.info(f"Using TEST dataset ({len(X_selected)} samples) for CV evaluation.")
                if not X_selected:
                    raise ValueError("Cannot perform CV evaluation on test set: Test set is empty.")
            except Exception as e:
                logger.error(f"Failed to get test set paths/labels for CV evaluation: {e}", exc_info=True)
                raise
        else:  # Should be caught by initial check
            pass
            # raise ValueError(f"Internal error: Unknown evaluate_on='{evaluate_on}'")

        y_selected_np = np.array(y_selected_list)
        if len(np.unique(y_selected_np)) < 2:
            logger.warning(
                f"Only one class present in the selected data ({evaluate_on} set). Stratification might behave unexpectedly.")
        # --- End Get Data ---

        # --- Hyperparameters for this evaluation ---
        eval_params = self.model_adapter_config.copy()
        if params:
            logger.info(f"Using provided parameters for CV evaluation: {params}")
            # Parse the provided params to resolve optimizer and create LRScheduler object
            parsed_params_for_cv = parse_fixed_hyperparameters(
                params,
                default_max_epochs_for_cosine=eval_params.get('max_epochs')
            )
            eval_params.update(parsed_params_for_cv)

        # Ensure critical module params
        eval_params['module'] = self._get_model_class(self.model_type)
        eval_params['module__num_classes'] = self.dataset_handler.num_classes
        eval_params['classes'] = np.arange(self.dataset_handler.num_classes) # Add if missing
        eval_params['train_transform'] = self.dataset_handler.get_train_transform() # Ensure these are not lost
        eval_params['valid_transform'] = self.dataset_handler.get_eval_transform()
        eval_params.setdefault('show_first_batch_augmentation', self.show_first_batch_augmentation_default)

        # --- Determine & Validate Internal Validation Split ---
        default_internal_val_fallback = 0.15
        val_frac_to_use = internal_val_split_ratio if internal_val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
            logger.warning(
                f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. Using default fallback: {default_internal_val_fallback:.3f} for fold fits.")
            val_frac_to_use = default_internal_val_fallback
        logger.info(
            f"Skorch internal validation split configured: {val_frac_to_use * 100:.1f}% of each CV fold's training data.")

        # --- Setup CV Strategy ---
        min_samples_per_class = cv if cv > 1 else 1
        unique_labels, counts = np.unique(y_selected_np, return_counts=True)
        if K := min(counts) < min_samples_per_class:
            logger.warning(f"The selected data ({evaluate_on} set) has only {K} instances "
                           f"of class '{unique_labels[np.argmin(counts)]}', "
                           f"which is less than the number of folds ({cv}). "
                           f"StratifiedKFold may fail or produce unreliable splits.")

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []  # Store summary dicts like {'accuracy': 0.X, 'f1_macro': 0.Y, ...}
        fold_histories = []
        fold_detailed_results = []

        # --- Determine if detailed metrics needed based on save/plot levels ---
        compute_detailed_metrics_flag = False
        effective_detail_level_for_compute = results_detail_level or self.results_detail_level

        current_plot_level = self.plot_level  # Start with pipeline default
        if plot_level is not None:
            current_plot_level = plot_level  # Use override if provided
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        # Compute if saving JSON (level > 0) AND plotting potentially detailed plots (plot_level > 0),
        # OR if saving detailed JSON (level >= 2)
        if (effective_detail_level_for_compute > 0 and current_plot_level > 0) or effective_detail_level_for_compute >= 2:
            compute_detailed_metrics_flag = True
            logger.debug(
                f"Will compute detailed metrics (detail_level={effective_detail_level_for_compute}, plot_level={current_plot_level})")

        # --- Manual Outer CV Loop ---
        for fold_idx, (outer_train_indices, outer_test_indices) in enumerate(
                cv_splitter.split(X_selected, y_selected_np)):  # <<< USE SELECTED DATA
            logger.info(f"--- Starting CV Evaluation Fold {fold_idx + 1}/{cv} ---")

            # --- Get Outer Fold Data from the SELECTED dataset ---
            X_outer_train = [X_selected[i] for i in outer_train_indices]
            y_outer_train = y_selected_np[outer_train_indices]
            X_fold_test = [X_selected[i] for i in outer_test_indices]  # Test set for this fold
            y_fold_test = y_selected_np[outer_test_indices]
            logger.debug(
                f"Outer split ({evaluate_on} set): {len(X_outer_train)} train / {len(X_fold_test)} test samples.")

            if not X_outer_train or not X_fold_test: # Check the fold's train and test parts
                logger.warning(f"Fold {fold_idx + 1} resulted in empty train ({len(X_outer_train)}) or test ({len(X_fold_test)}) set. Skipping.")
                # Append NaNs for all potential metrics if skipping
                fold_results.append({k: np.nan for k in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                                                         'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']})
                fold_detailed_results.append({'error': 'Skipped fold due to empty data.'})
                continue

            # Setup Estimator for this Fold
            fold_adapter_config = eval_params.copy()
            fold_adapter_config['train_split'] = ValidSplit(cv=val_frac_to_use, stratified=True,
                                                            random_state=RANDOM_SEED + fold_idx)
            fold_adapter_config['verbose'] = 0  # Show epoch table per fold

            n_outer_train = len(X_outer_train)
            n_inner_val = int(n_outer_train * val_frac_to_use)
            n_inner_train = n_outer_train - n_inner_val
            logger.debug(f"Fold {fold_idx + 1}: Internal split: ~{n_inner_train} train / ~{n_inner_val} valid.")

            fold_adapter_config.pop('patience_cfg', None)
            fold_adapter_config.pop('monitor_cfg', None)
            fold_adapter_config.pop('lr_policy_cfg', None)
            fold_adapter_config.pop('lr_patience_cfg', None)

            estimator_fold = SkorchModelAdapter(**fold_adapter_config)

            # Fit on Outer Train
            logger.info(f"Fitting model for fold {fold_idx + 1}...")
            try:
                estimator_fold.fit(X_outer_train, y=y_outer_train)
                if hasattr(estimator_fold, 'history_') and estimator_fold.history_:
                    fold_histories.append(estimator_fold.history)  # Store history if needed later
                else:
                    fold_histories.append([])  # Add empty list if no history
            except Exception as fit_err:
                logger.error(f"Fit failed for fold {fold_idx + 1}: {fit_err}", exc_info=True)
                fold_results.append({k: np.nan for k in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                                                         'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']})
                fold_detailed_results.append({'error': f'Fit failed: {fit_err}'})
                continue

            logger.info(f"Using best model from epoch {estimator_fold.history[-1]['epoch']}")

            # Evaluate on Outer Test Set
            logger.info(f"Evaluating model on outer test set for fold {fold_idx + 1}...")
            try:
                y_pred_fold_test = estimator_fold.predict(X_fold_test) # <<< Use X_fold_test
                y_score_fold_test = estimator_fold.predict_proba(X_fold_test) # <<< Use X_fold_test
                # Compute metrics using fold's test set labels
                fold_metrics = self._compute_metrics(y_fold_test, y_pred_fold_test, y_score_fold_test, # <<< Use y_fold_test
                                                  detailed=compute_detailed_metrics_flag)

                fold_summary = {
                    'accuracy': fold_metrics.get('overall_accuracy', np.nan),
                    'f1_macro': fold_metrics.get('macro_avg', {}).get('f1', np.nan),
                    'precision_macro': fold_metrics.get('macro_avg', {}).get('precision', np.nan),
                    'recall_macro': fold_metrics.get('macro_avg', {}).get('recall', np.nan),
                    'specificity_macro': fold_metrics.get('macro_avg', {}).get('specificity', np.nan),
                    'roc_auc_macro': fold_metrics.get('macro_avg', {}).get('roc_auc', np.nan),
                    'pr_auc_macro': fold_metrics.get('macro_avg', {}).get('pr_auc', np.nan),
                }
                fold_results.append(fold_summary)
                fold_detailed_results.append(fold_metrics)

                logger.info(
                    f"Fold {fold_idx + 1} Test Scores: Acc={fold_summary['accuracy']:.4f}, F1={fold_summary['f1_macro']:.4f}")
            except Exception as score_err:
                logger.error(f"Scoring failed for fold {fold_idx + 1}: {score_err}", exc_info=True)
                fold_results.append({k: np.nan for k in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                                                         'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']})
                fold_detailed_results.append({'error': f'Scoring failed: {score_err}'})

        # --- Aggregate Results ---
        if not fold_results:
            logger.error("CV evaluation failed: No results from any fold.")
            # Return minimal error dict
            return {'method': 'cv_model_evaluation', 'params': params or {},
                    'error': 'All folds failed or were skipped.'}

        df_results = pd.DataFrame(fold_results)
        results = {
            'method': 'cv_model_evaluation',
            'run_id': run_id,
            'params_used_for_folds': {k:v for k,v in eval_params.items() if not callable(v)},
            'evaluated_on': evaluate_on,
            'n_folds_requested': cv,
            'n_folds_processed': len(fold_results),
            'confidence_level': confidence_level,
            'cv_fold_scores': df_results.to_dict(orient='list'),
            'fold_detailed_results': fold_detailed_results,
            'fold_training_histories': [h.to_list() if h else [] for h in fold_histories],
        }

        # Calculate aggregated stats only if enough folds completed
        aggregated_metrics = {}
        K = results['n_folds_processed']
        if K > 1:
            dof = K - 1
            alpha = 1.0 - confidence_level
            try:
                t_crit = np.abs(stats.t.ppf(alpha / 2, dof))
            except ImportError:
                logger.warning(
                    "Scipy not found. Cannot calculate t-critical value for confidence intervals. Reporting SEM instead.")
                t_crit = None
            except Exception as e:
                logger.error(f"Error calculating t-critical value: {e}. Confidence intervals may be inaccurate.")
                t_crit = None

            metrics_to_aggregate = [k for k in df_results.columns if
                                    k != 'error']  # Aggregate all collected numeric metrics
            for metric_key in metrics_to_aggregate:
                scores = df_results[metric_key].dropna()
                count = len(scores)
                if count >= 2:
                    mean_score = np.mean(scores)
                    std_dev = np.std(scores, ddof=1)
                    sem = std_dev / np.sqrt(count)
                    h = (t_crit * sem) if t_crit is not None and not np.isnan(sem) else np.nan
                    aggregated_metrics[metric_key] = {
                        'mean': float(mean_score), 'std_dev': float(std_dev), 'sem': float(sem),
                        'margin_of_error': float(h) if not np.isnan(h) else None,  # Store as None if not calculable
                        'ci_lower': float(mean_score - h) if not np.isnan(h) else None,
                        'ci_upper': float(mean_score + h) if not np.isnan(h) else None
                    }
                elif count == 1:
                    aggregated_metrics[metric_key] = {'mean': float(scores.iloc[0]), 'std_dev': 0.0}
                else:
                    aggregated_metrics[metric_key] = {'mean': np.nan, 'std_dev': np.nan}
        elif K == 1:  # Only one fold processed
            logger.warning("Only 1 fold processed. Reporting mean scores, cannot calculate std dev or CI.")
            for metric_key in df_results.columns:
                if metric_key != 'error':
                    scores = df_results[metric_key].dropna()
                    aggregated_metrics[metric_key] = {'mean': float(scores.iloc[0]) if len(scores) > 0 else np.nan}
        else:  # K = 0
            logger.error("No folds successfully processed. Cannot aggregate metrics.")
            for metric_key in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'specificity_macro',
                               'roc_auc_macro', 'pr_auc_macro']:
                aggregated_metrics[metric_key] = {'mean': np.nan}

        results['aggregated_metrics'] = aggregated_metrics

        # --- Update top-level keys for summary CSV convenience ---
        results['accuracy'] = aggregated_metrics.get('accuracy', {}).get('mean', np.nan)
        results['macro_avg'] = {'f1': aggregated_metrics.get('f1_macro', {}).get('mean', np.nan)}
        # --- End Update ---

        # --- Save results ---
        summary_params = {k: v for k, v in eval_params.items() if isinstance(v, (str, int, float, bool))}
        summary_params['cv'] = cv
        summary_params['evaluated_on'] = evaluate_on  # <<< Add to summary params
        summary_params['internal_val_split_ratio'] = val_frac_to_use
        summary_params['confidence_level'] = confidence_level
        saved_json_path = self._save_results(results,
                                             "cv_model_evaluation",
                                             run_id=run_id,
                                             method_params=summary_params,
                                             results_detail_level=results_detail_level)

        # --- Plot results (conditionally) ---
        if current_plot_level > 0: # Proceed only if plotting is desired (level 1 or 2)
            # plot_save_dir_base will be the directory/S3_prefix for THIS SPECIFIC RUN's artifacts
            plot_save_dir_base_for_run: Optional[Union[str, Path]] = None

            if self.artifact_repo and self.experiment_run_key_prefix:
                # If using a repository, construct the base key for this run's artifacts
                plot_save_dir_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())
            else:
                plot_save_dir_base_for_run = None
                logger.warning(f"No artifact repo specified for this run.")

            # Condition to actually attempt saving plots to a file/repository:
            # Level 1 (save only) or Level 2 (save and show), AND a base location must exist.
            can_save_plots = (current_plot_level >= 1 and plot_save_dir_base_for_run is not None)
            should_show_plots_flag = (current_plot_level == 2)

            if not can_save_plots and current_plot_level == 1:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no save location could be determined (e.g., no repository).")

            if can_save_plots or should_show_plots_flag:  # Proceed if saving OR showing
                logger.info(f"Plotting cv for eval results for {run_id} (plot level {current_plot_level}).")
                try:
                    from ..plotter import ResultsPlotter # Corrected relative import
                    ResultsPlotter.plot_cv_model_evaluation_results(
                        results_input=results,
                        class_names=self.dataset_handler.classes,
                        plot_save_dir_base=plot_save_dir_base_for_run,
                        repository_for_plots=self.artifact_repo if can_save_plots else None,
                        show_plots=should_show_plots_flag
                    )
                except ImportError:
                    logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Plotting failed for {run_id} (cv for eval): {plot_err}", exc_info=True)

        # --- Updated Logging ---
        logger.info(f"CV Evaluation Summary (on {evaluate_on} data, {K} folds, {confidence_level * 100:.0f}% CI):")
        for metric_key, stats_dict in aggregated_metrics.items():
            mean_val = stats_dict.get('mean', np.nan)
            h_val = stats_dict.get('margin_of_error')  # Can be None or NaN
            if not np.isnan(mean_val):
                if h_val is not None and not np.isnan(h_val):
                    logger.info(f"  {metric_key.replace('_', ' ').title():<20}: {mean_val:.4f} +/- {h_val:.4f}")
                else:
                    logger.info(f"  {metric_key.replace('_', ' ').title():<20}: {mean_val:.4f} (CI not calculated)")
            else:
                logger.info(f"  {metric_key.replace('_', ' ').title():<20}: NaN")
        # --- End Updated Logging ---

        return results

    def single_train(self,
                     params: Optional[Dict[str, Any]] = None, # <<< General params override
                     # max_epochs: Optional[int] = None,
                     # lr: Optional[float] = None,
                     # batch_size: Optional[int] = None,
                     # Add other tunable params like weight_decay, dropout_rate here if needed
                     # optimizer__weight_decay: Optional[float] = None,
                     # module__dropout_rate: Optional[float] = None,
                     val_split_ratio: Optional[float] = None,  # Override handler's default split
                     save_model: bool = True,
                     results_detail_level: Optional[int] = None,
                     plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs a single training run using a train/validation split.
        Manually creates the split and uses Skorch with PredefinedSplit.
        Saves model and results in a unique subdirectory for this run.
        """
        logger.info("Starting single training run...")
        run_id = f"single_train_{datetime.now().strftime('%H%M%S')}"

        # --- Get Train+Validation Data ---
        X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels()
        if not X_trainval: raise RuntimeError("Train+validation data is empty.")
        y_trainval_np = np.array(y_trainval)

        # --- Determine Validation Split ---
        current_val_split_ratio = val_split_ratio if val_split_ratio is not None else self.dataset_handler.val_split_ratio
        train_split_config = None
        n_train, n_val = len(y_trainval_np), 0  # Default if no split

        if not 0.0 < current_val_split_ratio < 1.0:
            logger.warning(f"Validation split ratio ({current_val_split_ratio}) is invalid or zero. "
                           f"Training on full {len(X_trainval)} trainval samples without validation set.")
            X_fit, y_fit = X_trainval, y_trainval_np  # Use all data
        elif len(np.unique(y_trainval_np)) < 2:
            logger.warning(
                f"Only one class present in trainval data. Cannot stratify split. Training on full {len(X_trainval)} samples without validation.")
            X_fit, y_fit = X_trainval, y_trainval_np
        else:
            # Perform the split
            try:
                train_indices, val_indices = train_test_split(
                    np.arange(len(X_trainval)), test_size=current_val_split_ratio,
                    stratify=y_trainval_np, random_state=RANDOM_SEED)
            except ValueError as e:
                logger.warning(f"Stratified train/val split failed ({e}). Using non-stratified split.")
                train_indices, val_indices = train_test_split(
                    np.arange(len(X_trainval)), test_size=current_val_split_ratio,
                    random_state=RANDOM_SEED)

            # Use indices to get actual paths/labels for constructing X_fit, y_fit later
            X_train_paths_list = [X_trainval[i] for i in train_indices]
            y_train_labels_np = y_trainval_np[train_indices]
            X_val_paths_list = [X_trainval[i] for i in val_indices]
            y_val_labels_np = y_trainval_np[val_indices]
            n_train, n_val = len(y_train_labels_np), len(y_val_labels_np)

            # Combine for skorch fit (paths and labels separately)
            X_fit = X_train_paths_list + X_val_paths_list  # Combine lists of paths
            y_fit = np.concatenate((y_train_labels_np, y_val_labels_np))

            # Create PredefinedSplit using indices relative to the combined X_fit/y_fit
            test_fold = np.full(len(X_fit), -1, dtype=int)  # -1 indicates train
            test_fold[n_train:] = 0  # 0 indicates validation fold (indices from n_train onwards)
            ps = PredefinedSplit(test_fold=test_fold)
            # Skorch train_split requires a callable that yields train/test indices.
            # ValidSplit handles wrapping the PredefinedSplit correctly.
            train_split_config = ValidSplit(cv=ps,
                                            stratified=False)  # stratified=False because ps defines the split

        logger.info(f"Using split: {n_train} train / {n_val} validation samples.")

        # --- Configure Model Adapter ---
        adapter_config = self.model_adapter_config.copy()

        # Apply overrides from 'params' argument
        if params:
            logger.info(f"Applying custom parameters for this single_train run: {params}")
            # Parse the provided params to resolve optimizer strings and create LRScheduler object
            # Pass max_epochs from adapter_config_run as it might be needed for T_max default
            parsed_params = parse_fixed_hyperparameters(
                params,
                default_max_epochs_for_cosine=adapter_config.get('max_epochs')
            )
            adapter_config.update(parsed_params)
            # TODO: use params to override specific keys in adapter_config

        # Override params for this run
        # if max_epochs is not None: adapter_config['max_epochs'] = max_epochs
        # if lr is not None: adapter_config['lr'] = lr
        # if batch_size is not None: adapter_config['batch_size'] = batch_size
        # if optimizer__weight_decay is not None: adapter_config['optimizer__weight_decay'] = optimizer__weight_decay
        # if module__dropout_rate is not None: adapter_config['module__dropout_rate'] = module__dropout_rate
        # Set the train split strategy (None or PredefinedSplit via ValidSplit)
        adapter_config['train_split'] = train_split_config

        # --- Handle Callbacks based on validation ---
        # Start with the base callbacks list/None from the config
        final_callbacks = adapter_config.get('callbacks', []) # Get current callbacks
        if isinstance(final_callbacks, list): # Ensure it's a list
             # If parse_fixed_hyperparameters put a full LRScheduler object under 'callbacks__default_lr_scheduler'
             if 'callbacks__default_lr_scheduler' in adapter_config and isinstance(adapter_config['callbacks__default_lr_scheduler'], LRScheduler):
                 new_lr_scheduler_instance = adapter_config.pop('callbacks__default_lr_scheduler') # Get and remove temp key
                 # Find and replace or add the LRScheduler in the list
                 found_lr_scheduler = False
                 for i, (name, cb) in enumerate(final_callbacks):
                     if name == DEFAULT_LR_SCHEDULER_NAME:
                         final_callbacks[i] = (name, new_lr_scheduler_instance)
                         found_lr_scheduler = True
                         break
                 if not found_lr_scheduler: # Should not happen if get_default_callbacks includes it
                     final_callbacks.append((DEFAULT_LR_SCHEDULER_NAME, new_lr_scheduler_instance))
                 adapter_config['callbacks'] = final_callbacks

        if train_split_config is None:
            logger.warning(
                "No validation set. Callbacks monitoring validation metrics (EarlyStopping, LRScheduler) may be removed or ineffective.")
            # Filter out callbacks that depend on validation
            adapter_config['callbacks'] = [
                (name, cb) for name, cb in final_callbacks
                if not isinstance(cb, (EarlyStopping, LRScheduler))  # Keep others
            ]
        else:
            # Keep all callbacks from base config when validation exists
            adapter_config['callbacks'] = final_callbacks
        # --- End Callback Handling ---

        adapter_config['verbose'] = 0  # Show epoch table with train_acc

        # Pop config keys not needed by SkorchModelAdapter init directly
        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        # Instantiate the adapter for this training run
        adapter_for_train = SkorchModelAdapter(**adapter_config)

        # --- Train Model ---
        logger.info(f"Fitting model (run_id: {run_id})...")
        adapter_for_train.fit(X_fit, y=y_fit)  # Pass combined data (paths, labels)

        # --- Collect Results ---
        history = adapter_for_train.history
        # Start results dict, store effective config used for this run
        results: Dict[str, Any] = {'method': 'single_train',
                   'run_id': run_id,
                   'full_params_used': adapter_config.copy()}

        best_epoch_info = {}
        valid_loss_key = 'valid_loss'  # Default metric monitored
        # Check if validation ran by checking train_split and history content
        validation_was_run = train_split_config is not None and history and valid_loss_key in history[-1]

        if validation_was_run:
            try:
                scores = [epoch.get(valid_loss_key, np.inf) for epoch in history]
                best_idx = np.argmin(scores)  # Find index of min validation loss
                best_idx_int = int(best_idx)

                if best_idx_int < len(history):
                    best_epoch_hist = history[best_idx_int]
                    actual_best_epoch_num = best_epoch_hist.get('epoch')
                    best_epoch_info = {
                        'best_epoch': actual_best_epoch_num,
                        'best_valid_metric_value': float(best_epoch_hist.get(valid_loss_key, np.nan)),
                        'valid_metric_name': valid_loss_key,
                        'train_loss_at_best': float(best_epoch_hist.get('train_loss', np.nan)),
                        'train_acc_at_best': float(best_epoch_hist.get('train_acc', np.nan)),
                        'valid_acc_at_best': float(best_epoch_hist.get('valid_acc', np.nan)),
                    }
                    logger.info(
                        f"Training finished. Best validation performance found at Epoch {best_epoch_info['best_epoch']} "
                        f"({valid_loss_key}={best_epoch_info['best_valid_metric_value']:.4f})")
                else:
                    logger.error("Could not determine best epoch index from history scores.")
                    validation_was_run = False  # Fallback
            except Exception as e:
                logger.error(f"Error processing history for best epoch: {e}", exc_info=True)
                validation_was_run = False  # Fallback

        if not validation_was_run:  # No validation or error processing history
            last_epoch_hist = history[-1] if history else {}
            last_epoch_num = last_epoch_hist.get('epoch', len(history) if history else 0)
            if not history: logger.error("History empty after fit.")

            best_epoch_info = {
                'best_epoch': last_epoch_num,
                'best_valid_metric_value': np.nan, 'valid_metric_name': valid_loss_key,
                'train_loss_at_best': float(last_epoch_hist.get('train_loss', np.nan)),
                'train_acc_at_best': float(last_epoch_hist.get('train_acc', np.nan)),
                'valid_acc_at_best': np.nan,
            }
            if train_split_config is not None:  # Log warning only if split was intended but failed
                logger.warning(
                    f"Error finding best epoch based on validation. Reporting last epoch ({last_epoch_num}) stats.")

        results.update(best_epoch_info)
        # Include full history only if detailed results are requested
        results['training_history'] = history.to_list() if history else []

        # --- Save Model ---
        model_path_identifier = None  # Will store the S3 key or local path string
        if save_model:
            # --- NEW FILENAME LOGIC ---
            model_type_short = self.model_type.value  # e.g., "pvit", "hyvit"
            run_type_short = "sngl"
            # Use part of the existing run_id (which contains a timestamp)
            run_id_timestamp_part = run_id.split('_')[-1]  # Assumes run_id format like "single_train_TIMESTAMP"

            val_metric_val = results.get('best_valid_metric_value', np.nan)
            metric_name_short = "val_loss"  # Or adapt if you monitor other things like 'val_acc'
            if not np.isnan(val_metric_val):
                # Format to 2 decimal places, replace dot, remove leading zero if < 1
                metric_str = f"{metric_name_short}{val_metric_val:.2f}".replace('.', 'p').replace("0p", "p")
            else:
                metric_str = "no_val"

            epoch_num = results.get('best_epoch', 0)
            model_filename_base = f"{model_type_short}_{run_type_short}_ep{epoch_num}_{metric_str}_{run_id_timestamp_part}"
            # --- END NEW FILENAME LOGIC ---

            model_pt_filename = f"{model_filename_base}.pt"
            model_config_filename = f"{model_filename_base}_arch_config.json"

            model_pt_object_key = self._get_s3_object_key(run_id, model_pt_filename)  # run_id is still the folder name
            model_config_object_key = self._get_s3_object_key(run_id, model_config_filename)

            state_dict = adapter_for_train.module_.state_dict()
            model_path_identifier = self.artifact_repo.save_model_state_dict(state_dict, model_pt_object_key)

            # 2. Save architectural config
            if model_path_identifier:  # Only if model saving was successful
                # Get all relevant 'module__' parameters from the adapter's config
                # This adapter_config was used to initialize the SkorchModelAdapter for this run
                effective_adapter_config = adapter_for_train.get_params(
                    deep=False)  # Get effective params of this instance

                arch_config = {
                    'model_type': self.model_type.value,  # Store the enum value
                    'num_classes': self.dataset_handler.num_classes
                }
                for key, value in effective_adapter_config.items():
                    if key.startswith('module__'):
                        arch_config[key] = value
                    # Add other direct architectural params if SkorchModelAdapter has them (e.g. is_hybrid_input IF it were a direct param of module)
                    # For PretrainedViT, is_hybrid_input is a module__ param if you set it that way, or a constructor arg
                    # For your HybridViT, things like cnn_model_name are module__params.

                # Specifically ensure architectural params of PretrainedViT/Swin/Hybrid are captured
                # These should be retrievable from effective_adapter_config if they were set via module__
                # Example for PretrainedViT specific params (if they are not module__ in adapter_config
                # but rather direct constructor args to PretrainedViT that need to be known)
                # This part depends on how your HybridViT/PretrainedViT __init__ signatures are structured
                # and what Skorch passes as module__

                # The most reliable way is to fetch them from the actual module instance if possible,
                # but Skorch's get_params() on the adapter should give you the 'module__xyz' values.
                # Let's assume effective_adapter_config from skorch is sufficient here.

                arch_config_path_identifier = self.artifact_repo.save_json(arch_config, model_config_object_key)
                logger.info(f"Model architectural config saved to: {model_config_object_key}")
                results['saved_model_arch_config_path'] = arch_config_path_identifier
        results['saved_model_path'] = model_path_identifier  # Store key/path or None

        self.model_adapter = adapter_for_train
        logger.info(f"Main pipeline model adapter updated from single_train run: {run_id}")

        results['accuracy'] = results.get('valid_acc_at_best', np.nan)
        results['macro_avg'] = {}

        # --- Save Results JSON (this call is independent of plotting the data) ---
        simple_params = {k: v for k, v in adapter_config.items() if isinstance(v, (str, int, float, bool))}
        simple_params['val_split_ratio_used'] = current_val_split_ratio if train_split_config else 0.0

        json_artifact_key_or_path = self._save_results(  # _save_results now uses artifact_repo
            results_data=results,
            method_name="single_train",  # Used for part of the artifact key
            run_id=run_id,  # This is the specific method execution ID
            method_params=simple_params,
            results_detail_level=results_detail_level
        )

        # --- Determine effective plot level ---
        current_plot_level = self.plot_level  # Start with pipeline default
        if plot_level is not None: current_plot_level = plot_level

        # --- Plot results (conditionally) ---
        if current_plot_level > 0:
            # The base location for plots associated with this run.
            # For S3, this is a prefix. For local, it's a directory path.
            plot_save_location_base: Optional[str] = None
            if self.artifact_repo and self.experiment_run_key_prefix:
                # For S3, the plotter will append specific plot names to this base key
                plot_save_location_base = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())

            # If only saving (level 1) but no way to save (no repo/prefix), then skip file saving part of plotting
            if current_plot_level == 1 and not plot_save_location_base:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 but no repository/base_key configured for saving.")
                # If show_plots was also desired, it would be level 2.
            else:
                # Proceed if showing (level 2) OR if saving and save location exists (level 1)
                logger.info(f"Plotting single_train results for {run_id} (plot level {current_plot_level}).")
                show_plots_flag = (current_plot_level == 2)
                try:
                    from ..plotter import ResultsPlotter  # Ensure correct relative import
                    ResultsPlotter.plot_single_train_results(
                        results_input=results,
                        plot_save_dir_base=plot_save_location_base,  # Pass base key/path for plots
                        repository_for_plots=self.artifact_repo if plot_save_location_base else None,
                        # Pass repo if saving
                        show_plots=show_plots_flag
                    )
                except ImportError:
                    logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Plotting failed for {run_id}: {plot_err}", exc_info=True)
        return results

    def single_eval(self,
                    results_detail_level: Optional[int] = None,
                    plot_level: Optional[int] = None) -> Dict[str, Any]:
        """ Evaluates the current model adapter on the test set. """
        logger.info("Starting model evaluation on the test set...")
        run_id = f"single_eval_{datetime.now().strftime('%H%M%S')}" # Unique run ID for this execution

        if not self.model_adapter.initialized_:
             raise RuntimeError("Model adapter not initialized. Train or load first.")

        # --- Get Test Data ---
        X_test, y_test = self.dataset_handler.get_test_paths_labels()
        if not X_test:
             logger.warning("Test set is empty. Skipping evaluation.")
             return {'method': 'single_eval', 'message': 'Test set empty, evaluation skipped.'}
        y_test_np = np.array(y_test)

        # --- Make Predictions ---
        logger.info(f"Evaluating on {len(X_test)} test samples...")
        try:
             y_pred_test = self.model_adapter.predict(X_test)
             y_score_test = self.model_adapter.predict_proba(X_test)
        except Exception as e:
             logger.error(f"Prediction failed during single_eval: {e}", exc_info=True)
             raise RuntimeError("Failed to get predictions from model adapter.") from e

        # --- Compute Metrics ---
        # Determine detail level for computing metrics based on potential plotting needs
        compute_detailed_metrics_flag = False
        effective_detail_level_for_compute = results_detail_level or self.results_detail_level

        # --- Determine effective plot level ---
        current_plot_level = self.plot_level  # Start with pipeline default
        if plot_level is not None:
            current_plot_level = plot_level  # Use override if provided
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        # --- Compute Metrics (pass detailed flag) ---
        # If saving JSON (level > 0) AND plotting potentially detailed plots (plot_level > 0),
        # OR if saving detailed JSON (level >= 2), compute detailed metrics.
        if (effective_detail_level_for_compute > 0 and current_plot_level > 0) or effective_detail_level_for_compute >= 2:
            compute_detailed_metrics_flag = True

        metrics = self._compute_metrics(y_test_np, y_pred_test, y_score_test, detailed=compute_detailed_metrics_flag)
        results = {
            'method': 'single_eval',
            'params': {},
            'run_id': run_id,
            **metrics}
        method_name = results['method']  # Used for saving

        saved_json_path = self._save_results(results, "single_eval",
                           method_params=results['params'],
                           run_id=run_id,
                           results_detail_level=results_detail_level)

        # --- Plot results (conditionally) ---
        if current_plot_level > 0:
            # plot_save_dir_base will be the directory/S3_prefix for THIS SPECIFIC RUN's artifacts
            plot_save_dir_base_for_run: Optional[Union[str, Path]] = None

            if self.artifact_repo and self.experiment_run_key_prefix:
                # If using a repository, construct the base key for this run's artifacts
                plot_save_dir_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())
            else:
                plot_save_dir_base_for_run = None
                logger.warning(f"No artifact repo specified for this run.")

                # Condition to actually attempt saving plots to a file/repository:
                # Level 1 (save only) or Level 2 (save and show), AND a base location must exist.
            can_save_plots = (current_plot_level >= 1 and plot_save_dir_base_for_run is not None)
            should_show_plots_flag = (current_plot_level == 2)

            if not can_save_plots and current_plot_level == 1:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no save location could be determined (e.g., no repository).")

            if can_save_plots or should_show_plots_flag:  # Proceed if saving OR showing
                logger.info(f"Plotting {method_name} results for {run_id} (plot level {current_plot_level}).")
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_single_eval_results(
                        results_input=results,
                        class_names=self.dataset_handler.classes,
                        plot_artifact_base_key_or_path=plot_save_dir_base_for_run,
                        repository_for_plots=self.artifact_repo if can_save_plots else None,
                        show_plots=should_show_plots_flag
                    )
                except ImportError:
                     logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Plotting failed for {run_id}: {plot_err}", exc_info=True)

        return results

    def predict_images(self,
                       image_id_format_pairs: List[Tuple[Union[int, str], str]],
                       experiment_run_id_of_model: str,
                       username: str = "anonymous", # Default username for artifact storage
                       persist_prediction_artifacts: bool = True,
                       results_detail_level: Optional[int] = None,
                       # Still used for other things, just not LIME segments in JSON
                       plot_level: int = 0,
                       generate_lime_explanations: bool = False,
                       lime_num_features_to_show_plot: int = 5,
                       lime_num_samples_for_explainer: int = 1000,
                       prob_plot_top_k: int = -1
                       ) -> List[Dict[str, Any]]:

        predict_op_run_id = f"predict_op_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.info(f"Op {predict_op_run_id}: Starting prediction for {len(image_id_format_pairs)} images "
                    f"by user '{username}', using model from experiment '{experiment_run_id_of_model}'.")

        if not self.model_adapter.initialized_: logger.error(
            f"Op {predict_op_run_id}: Model adapter not initialized."); raise RuntimeError(
            "Model adapter not initialized.")
        if not image_id_format_pairs: logger.warning(f"Op {predict_op_run_id}: No image_id_format_pairs."); return []
        lime_explainer = None
        if generate_lime_explanations and LIME_AVAILABLE:
            lime_explainer = LimeImageExplainer(random_state=RANDOM_SEED)

            def lime_predict_fn(numpy_images_batch_lime):
                processed_images_lime = []
                for img_np_lime in numpy_images_batch_lime:
                    if img_np_lime.dtype == np.double or img_np_lime.dtype == np.float64 or img_np_lime.dtype == np.float32:
                        if img_np_lime.max() <= 1.0 and img_np_lime.min() >= 0.0:
                            img_np_lime = (img_np_lime * 255).astype(np.uint8)
                        else:
                            img_np_lime = np.clip(img_np_lime, 0, 255).astype(np.uint8)
                    elif img_np_lime.dtype != np.uint8:
                        img_np_lime = np.clip(img_np_lime, 0, 255).astype(np.uint8)
                    pil_img_lime = Image.fromarray(img_np_lime);
                    transformed_img_lime = self.dataset_handler.get_eval_transform()(pil_img_lime);
                    processed_images_lime.append(transformed_img_lime)
                if not processed_images_lime: return np.array([])
                batch_tensor_lime = torch.stack(processed_images_lime).to(self.model_adapter.device);
                self.model_adapter.module_.eval()
                with torch.no_grad():
                    logits_lime = self.model_adapter.module_(batch_tensor_lime); probs_lime = torch.softmax(logits_lime,
                                                                                                            dim=1)
                return probs_lime.cpu().numpy()
        elif generate_lime_explanations:
            logger.warning(f"Op {predict_op_run_id}: LIME requested but not available.")
        pil_images_for_processing: List[Tuple[Optional[Image.Image], Union[int, str]]] = []
        for image_id, img_format in image_id_format_pairs:
            pil_image: Optional[Image.Image] = None;
            img_filename = f"{image_id}.{img_format.lower().replace('.', '')}"
            image_key_or_path: str
            if self.artifact_repo and not isinstance(self.artifact_repo, LocalFileSystemRepository):
                image_key_or_path = str((PurePath("images") / username / img_filename).as_posix())
                try:
                    img_bytes = self.artifact_repo.download_file_to_memory(image_key_or_path)
                    if img_bytes:
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    else:
                        logger.warning(f"Image not found/empty S3: {image_key_or_path}")
                except Exception as e:
                    logger.error(f"Failed to load S3 image {image_key_or_path}: {e}")
            else:
                local_base = Path(self.artifact_repo.base_path if self.artifact_repo else ".");
                image_key_or_path = local_base / "images" / username / img_filename
                if image_key_or_path.is_file():
                    try:
                        pil_image = Image.open(image_key_or_path).convert('RGB')
                    except Exception as e:
                        logger.error(f"Failed to load local image {image_key_or_path}: {e}")
                else:
                    logger.warning(f"Image not found local: {image_key_or_path}")
            pil_images_for_processing.append((pil_image, image_id))
        valid_pil_images = [img for img, _ in pil_images_for_processing if img is not None]
        valid_image_ids = [img_id for img, img_id in pil_images_for_processing if img is not None]
        if not valid_pil_images: logger.error(f"Op {predict_op_run_id}: No valid images loaded."); return []
        logger.info(f"Op {predict_op_run_id}: Loaded {len(valid_pil_images)} images.")

        class InMemoryPILDataset(torch.utils.data.Dataset):
            def __init__(self, pil_images: List[Image.Image], identifiers: List[Union[int, str]], transform: Callable):
                self.pil_images = pil_images; self.identifiers = identifiers; self.transform = transform

            def __len__(self):
                return len(self.pil_images)

            def __getitem__(self, idx):
                img = self.pil_images[idx];
                identifier = self.identifiers[idx];
                label_tensor = torch.tensor(-1, dtype=torch.long)
                try:
                    transformed_img = self.transform(img); return transformed_img, label_tensor
                except Exception as e:
                    logger.warning(
                        f"Transform failed for image '{identifier}' (idx {idx}): {e}"); return None, label_tensor

        eval_transform = self.dataset_handler.get_eval_transform()
        prediction_dataset = InMemoryPILDataset(pil_images=valid_pil_images, identifiers=valid_image_ids,
                                                transform=eval_transform)
        dataloader = torch.utils.data.DataLoader(prediction_dataset,
                                                 batch_size=self.model_adapter_config.get('batch_size', 32),
                                                 shuffle=False, num_workers=0, collate_fn=PathImageDataset.collate_fn)
        all_probabilities_np: List[np.ndarray] = []
        self.model_adapter.module_.eval()
        with torch.no_grad():
            for batch_images, _ in dataloader:
                if batch_images is None or len(batch_images) == 0: continue
                batch_images = batch_images.to(self.model_adapter.device)
                logits = self.model_adapter.module_(batch_images)
                probabilities = torch.softmax(logits, dim=1)
                all_probabilities_np.extend(probabilities.cpu().numpy())
        if len(all_probabilities_np) != len(valid_image_ids):
            logger.error(
                f"Op {predict_op_run_id}: Mismatch: predictions ({len(all_probabilities_np)}) vs valid images ({len(valid_image_ids)}).")

        predictions_to_return_for_api = []
        # effective_results_detail_level is still used by _save_results for general verbosity control
        # but we will explicitly exclude LIME segments from the JSON saved by _save_results.
        effective_results_detail_level = self.results_detail_level if results_detail_level is None else results_detail_level

        for i, image_id in enumerate(valid_image_ids):
            if i >= len(all_probabilities_np): continue

            probs_np = all_probabilities_np[i]
            predicted_idx = int(np.argmax(probs_np))
            predicted_name = self.dataset_handler.classes[predicted_idx]
            confidence = float(probs_np[predicted_idx])
            top_k_val = min(prob_plot_top_k if prob_plot_top_k > 0 else self.dataset_handler.num_classes,
                            self.dataset_handler.num_classes)
            top_k_indices = np.argsort(probs_np)[-top_k_val:][::-1]
            top_k_preds_list = [(self.dataset_handler.classes[k_idx], float(probs_np[k_idx])) for k_idx in
                                top_k_indices]

            # This dictionary is what will be passed to the LIME plotter
            # It will contain segments if LIME runs successfully.
            lime_data_for_plotter = None

            # This dictionary is what will be written to the JSON file.
            # It will NOT contain segments.
            lime_data_for_json_file = None

            if generate_lime_explanations and lime_explainer:
                pil_image_for_lime = valid_pil_images[i]
                logger.debug(
                    f"Op {predict_op_run_id}: Generating LIME for image_id: {image_id} (Pred: {predicted_name})")
                try:
                    explanation = lime_explainer.explain_instance(
                        np.array(pil_image_for_lime), lime_predict_fn,
                        top_labels=1, hide_color=0, num_features=lime_num_features_to_show_plot,
                        num_samples=lime_num_samples_for_explainer, random_seed=RANDOM_SEED
                    )
                    lime_weights = explanation.local_exp.get(predicted_idx, [])

                    # Populate data for the plotter (always include segments if LIME ran)
                    lime_data_for_plotter = {
                        'explained_class_idx': predicted_idx,
                        'explained_class_name': predicted_name,
                        'feature_weights': lime_weights,
                        'segments_for_render': explanation.segments.tolist(),  # For plotter
                        'num_features_from_lime_run': lime_num_features_to_show_plot
                    }

                    # Populate data for the JSON file (exclude segments)
                    lime_data_for_json_file = {
                        'explained_class_idx': predicted_idx,
                        'explained_class_name': predicted_name,
                        'feature_weights': lime_weights,
                        'num_features_from_lime_run': lime_num_features_to_show_plot
                        # 'segments_for_render' is intentionally omitted here
                    }

                except Exception as lime_e:
                    logger.error(f"Op {predict_op_run_id}: LIME failed for {image_id}: {lime_e}", exc_info=False)
                    lime_data_for_json_file = {'error': str(lime_e)}  # Also set plotter data to this error
                    lime_data_for_plotter = lime_data_for_json_file

                    # Prepare the full content for the individual prediction JSON file
            single_prediction_json_content = {
                "image_id": image_id,
                "experiment_run_id_of_model": experiment_run_id_of_model,
                "image_user_source_path": f"images/{username}/{image_id}.{dict(image_id_format_pairs).get(image_id, 'unknown_fmt')}",
                "probabilities": probs_np.tolist(),
                "predicted_class_idx": predicted_idx,
                "predicted_class_name": predicted_name,
                "confidence": confidence,
                "top_k_predictions_for_plot": top_k_preds_list,
                "lime_explanation": lime_data_for_json_file  # Use the version without segments
            }

            predictions_to_return_for_api.append({
                "image_id": image_id, "experiment_id": experiment_run_id_of_model,
                "predicted_class": predicted_name, "confidence": confidence,
            })

            prediction_artifact_base_path = PurePath("predictions") / username / str(
                image_id) / experiment_run_id_of_model

            # Save individual prediction JSON
            if persist_prediction_artifacts and self.artifact_repo and effective_results_detail_level > 0:
                pred_json_key = str((prediction_artifact_base_path / "prediction_details.json").as_posix())
                # Pass the content that EXCLUDES segments
                self.artifact_repo.save_json(single_prediction_json_content, pred_json_key)
                logger.info(f"Op {predict_op_run_id}: Prediction JSON for image {image_id} saved to: {pred_json_key}")

            # Generate and Save LIME Plot (if enabled and data available)
            if generate_lime_explanations and lime_data_for_plotter and 'error' not in lime_data_for_plotter:
                if persist_prediction_artifacts and self.artifact_repo and plot_level > 0:
                    lime_plot_key = str((prediction_artifact_base_path / "plots" / "lime_explanation.png").as_posix())
                    ResultsPlotter.plot_lime_explanation_image(
                        original_pil_image=valid_pil_images[i],  # Pass the correct PIL image
                        lime_explanation_data=lime_data_for_plotter,  # Pass data WITH segments
                        lime_num_features_to_display=lime_num_features_to_show_plot,
                        output_path=lime_plot_key,
                        repository_for_plots=self.artifact_repo,
                        show_plots=(plot_level == 2),
                        image_identifier=str(image_id)
                    )
            elif generate_lime_explanations and (not lime_data_for_plotter or 'error' in lime_data_for_plotter):
                logger.warning(
                    f"Op {predict_op_run_id}: LIME plot generation skipped for image {image_id} due to missing LIME data or LIME error.")

            # Probability Distribution Plot (as before)
            if persist_prediction_artifacts and self.artifact_repo and plot_level > 0:
                prob_plot_key = str(
                    (prediction_artifact_base_path / "plots" / "probability_distribution.png").as_posix())
                ResultsPlotter.plot_single_prediction_probabilities(
                    probabilities=probs_np, class_names=self.dataset_handler.classes,
                    image_identifier=str(image_id), output_path=prob_plot_key,
                    repository_for_plots=self.artifact_repo, show_plots=(plot_level == 2),
                    top_n=prob_plot_top_k
                )

        logger.info(f"Op {predict_op_run_id}: Finished predictions for {len(valid_image_ids)} images.")
        return predictions_to_return_for_api

    def load_model(self, model_path_or_key: Union[str, Path]) -> None:
        """
        Loads a state_dict and its architectural config into the pipeline's model adapter.
        - model_path_or_key: Path to the .pt file.
          - For S3/MinIO: Relative path within the 'experiments/' prefix (e.g., "dataset/model/run_id/model.pt").
          - For Local: Can be absolute or relative.
        """
        run_id_for_log = f"load_model_op_{datetime.now().strftime('%H%M%S')}"
        logger.info(f"Operation {run_id_for_log}: Attempting to load model from: {model_path_or_key}")

        # Construct full paths/keys for model and its config
        # The experiment_run_id_of_model is part of model_path_or_key

        base_model_path_str = str(model_path_or_key)

        # For S3/MinIO, ensure the path is prefixed with 'experiments/' if not already
        # For local repo, this prefixing isn't strictly necessary if paths are absolute or correctly relative.
        full_model_pt_key_or_path = base_model_path_str
        if self.artifact_repo and not isinstance(self.artifact_repo, LocalFileSystemRepository):
            if not base_model_path_str.startswith("experiments/"):
                full_model_pt_key_or_path = str((PurePath("experiments") / base_model_path_str).as_posix())

        config_filename = f"{Path(base_model_path_str).stem}_arch_config.json"
        full_arch_config_key_or_path: str
        if self.artifact_repo and not isinstance(self.artifact_repo, LocalFileSystemRepository):
            full_arch_config_key_or_path = str((PurePath(full_model_pt_key_or_path).parent / config_filename).as_posix())
        else:
            full_arch_config_key_or_path = str(Path(full_model_pt_key_or_path).parent / config_filename)

        logger.debug(f"Op {run_id_for_log}: Effective model artifact path/key: {full_model_pt_key_or_path}")
        logger.debug(f"Op {run_id_for_log}: Effective arch_config path/key: {full_arch_config_key_or_path}")

        # --- 1. Load Architectural Config ---
        arch_config_dict: Optional[Dict[str, Any]] = None
        if self.artifact_repo:
            arch_config_dict = self.artifact_repo.load_json(full_arch_config_key_or_path)

        if arch_config_dict is None and Path(full_arch_config_key_or_path).is_file():  # Fallback
            logger.info(
                f"Op {run_id_for_log}: Arch config not found via repo or no repo, trying local: {full_arch_config_key_or_path}")
            try:
                with open(full_arch_config_key_or_path, 'r') as f:
                    arch_config_dict = json.load(f)
            except Exception as e:
                logger.error(f"Op {run_id_for_log}: Failed to load local arch_config: {e}")

        if arch_config_dict is None:
            logger.error(
                f"Op {run_id_for_log}: CRITICAL: Architecture config not found. Attempting load with current pipeline defaults.")
            # Fallback to current pipeline defaults if arch_config is missing
            if not self.model_adapter.initialized_:
                try:
                    self.model_adapter.initialize()
                except Exception as e:
                    raise RuntimeError(f"Op {run_id_for_log}: Default init failed (arch_config missing): {e}") from e
        else:
            logger.info(f"Op {run_id_for_log}: Loaded architecture config from: {full_arch_config_key_or_path}")
            # Re-initialize SkorchModelAdapter with loaded architecture
            loaded_model_type_str = arch_config_dict.pop('model_type')
            loaded_num_classes = arch_config_dict.pop('num_classes')
            try:
                loaded_model_type_enum = ModelType(loaded_model_type_str)
            except ValueError:
                raise RuntimeError(f"Op {run_id_for_log}: Invalid model_type '{loaded_model_type_str}' in arch config.")

            LoadedModelClass = self._get_model_class(loaded_model_type_enum)
            current_pipeline_defaults = self.model_adapter_config.copy()
            current_pipeline_defaults['module'] = LoadedModelClass
            current_pipeline_defaults['module__num_classes'] = loaded_num_classes

            for key_arch, val_arch in arch_config_dict.items():
                current_pipeline_defaults[key_arch] = val_arch

            # Clean up non-init keys & ensure necessary refs are passed
            for k_pop in ['patience_cfg', 'monitor_cfg', 'lr_policy_cfg', 'lr_patience_cfg']:
                current_pipeline_defaults.pop(k_pop, None)
            current_pipeline_defaults['dataset_handler_ref'] = self.dataset_handler
            current_pipeline_defaults['train_transform'] = self.dataset_handler.get_train_transform()
            current_pipeline_defaults['valid_transform'] = self.dataset_handler.get_eval_transform()
            # Ensure 'classes' is set for skorch compatibility during prediction
            current_pipeline_defaults['classes'] = np.arange(loaded_num_classes)

            logger.info(
                f"Op {run_id_for_log}: Re-initializing SkorchModelAdapter for {LoadedModelClass.__name__} with {loaded_num_classes} classes.")
            try:
                self.model_adapter = SkorchModelAdapter(**current_pipeline_defaults)
                self.model_adapter.initialize()
            except Exception as e_reinit:
                logger.error(f"Op {run_id_for_log}: Failed to re-init SkorchModelAdapter: {e_reinit}", exc_info=True)
                raise RuntimeError("Adapter re-initialization failed.")

        if not self.model_adapter.module_ or not isinstance(self.model_adapter.module_, nn.Module):
            raise RuntimeError("Op {run_id_for_log}: Adapter's nn.Module not found after initialization.")

        # --- 2. Load State Dict ---
        state_dict: Optional[Dict] = None
        map_location = self.model_adapter.device

        if self.artifact_repo:
            state_dict = self.artifact_repo.load_model_state_dict(str(full_model_pt_key_or_path),
                                                                  map_location=map_location)
            if state_dict: logger.info(
                f"Op {run_id_for_log}: Model state_dict loaded via repo from: {full_model_pt_key_or_path}")

        if state_dict is None:  # Fallback
            local_pt_path = Path(full_model_pt_key_or_path)  # This might be an absolute path if local repo
            if not self.artifact_repo and not local_pt_path.is_file():  # No repo and not a file, something is wrong with path
                local_pt_path = Path(model_path_or_key)  # Try original path if prefixing was wrong for local

            if local_pt_path.is_file():
                logger.info(f"Op {run_id_for_log}: Trying local load for state_dict: {local_pt_path}")
                try:
                    state_dict = torch.load(local_pt_path, map_location=map_location, weights_only=True)
                    logger.info(f"Op {run_id_for_log}: Model state_dict loaded from local: {local_pt_path}")
                except Exception as e_load_local:
                    logger.error(f"Op {run_id_for_log}: Failed to load local state_dict: {e_load_local}")
            elif not self.artifact_repo:
                logger.error(f"Op {run_id_for_log}: Model file not found locally: {local_pt_path} (no repo).")

        if state_dict:
            try:
                # Handle Skorch "module." prefix if present
                is_skorch_module_prefixed = all(k.startswith("module.") for k in state_dict.keys())
                if is_skorch_module_prefixed and hasattr(self.model_adapter, 'module_') and isinstance(
                        self.model_adapter.module_, nn.Module):
                    logger.debug("Op {run_id_for_log}: Stripping 'module.' prefix from state_dict keys.")
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

                self.model_adapter.module_.load_state_dict(state_dict)
                self.model_adapter.module_.eval()
                logger.info(f"Op {run_id_for_log}: Model state_dict applied successfully from: {model_path_or_key}")
            except Exception as e_apply:
                logger.error(f"Op {run_id_for_log}: Failed to apply state_dict: {e_apply}", exc_info=True)
                raise RuntimeError(f"Error applying state_dict from '{model_path_or_key}'.") from e_apply
        else:
            raise FileNotFoundError(
                f"Op {run_id_for_log}: Model state_dict could not be loaded from: {model_path_or_key}.")
