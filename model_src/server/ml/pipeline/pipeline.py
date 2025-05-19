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

from ..architectures import ModelType
from ..config import RANDOM_SEED, DEVICE, DEFAULT_IMG_SIZE
from ..dataset_utils import ImageDatasetHandler, DatasetStructure, PathImageDataset
from model_src.server.ml.logger_utils import logger
from ..skorch_utils import SkorchModelAdapter
from ..skorch_utils import get_default_callbacks
from ...persistence import MinIORepository
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


# --- Classification Pipeline ---
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
                 results_detail_level: int = 1, # Flag for detailed results
                 plot_level: int = 0,
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
                data_augmentation: Whether to apply data augmentation to the training set.
                force_flat_for_fixed_cv: If True, treats a FIXED dataset structure (train/test splits)
                                         as a single pool of data for CV methods that operate on the
                                         'full' dataset (e.g., nested_grid_search, cv_model_evaluation
                                         with evaluate_on='full'). Use with caution.
                lr: Default learning rate for the optimizer.
                max_epochs: Default maximum number of training epochs.
                batch_size: Default batch size for training and evaluation.
                patience: Default patience for EarlyStopping callback.
                optimizer__weight_decay: Default weight decay for the AdamW optimizer.
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

        # Logger is configured externally by the Executor
        logger.info(f"Initializing Classification Pipeline:")
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Model Type: {self.model_type.value}")
        logger.info(f"  Force Flat for Fixed CV: {self.force_flat_for_fixed_cv}")
        logger.info(f"  Default Results Detail Level: {self.results_detail_level}")
        logger.info(f"  Default Plot Level: {self.plot_level}")

        self.dataset_handler = ImageDatasetHandler(
            root_path=self.dataset_path, img_size=img_size,
            val_split_ratio=val_split_ratio, test_split_ratio_if_flat=test_split_ratio_if_flat,
            data_augmentation=data_augmentation, force_flat_for_fixed_cv=self.force_flat_for_fixed_cv
        )

        self.artifact_repo : Optional[ArtifactRepository] = artifact_repository
        self.experiment_run_key_prefix: Optional[str] = experiment_base_key_prefix

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

        model_class = self._get_model_class(self.model_type)

        intended_callbacks_setting = 'default'
        initial_callbacks_list = None
        if intended_callbacks_setting == 'default':
            initial_callbacks_list = get_default_callbacks(
                patience=patience, monitor='valid_loss',
                lr_policy='ReduceLROnPlateau', lr_patience=5
            )
        elif isinstance(intended_callbacks_setting, list):
            initial_callbacks_list = intended_callbacks_setting

        module_params = {}
        if module__dropout_rate is not None: module_params['module__dropout_rate'] = module__dropout_rate

        self.model_adapter_config = {
            'module': model_class, 'module__num_classes': self.dataset_handler.num_classes,
            'criterion': nn.CrossEntropyLoss, 'optimizer': torch.optim.AdamW, # TODO: Add support for other optimizers
            'lr': lr, 'max_epochs': max_epochs, 'batch_size': batch_size, 'device': DEVICE,
            'callbacks': initial_callbacks_list,
            'patience_cfg': patience, 'monitor_cfg': 'valid_loss',
            'lr_policy_cfg': 'ReduceLROnPlateau', 'lr_patience_cfg': 5,
            'train_transform': self.dataset_handler.get_train_transform(),
            'valid_transform': self.dataset_handler.get_eval_transform(),
            'classes': np.arange(self.dataset_handler.num_classes),
            'verbose': 0, # Default verbosity for adapter itself
            'optimizer__weight_decay': optimizer__weight_decay,
            **module_params
        }

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
        logger.info(f"Parameter Grid/Dist:\n{json.dumps(param_grid, indent=2)}") # Log the potentially complex grid
        logger.info(f"Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None: raise ValueError("n_iter required for random search.")
        if method_lower not in ['grid', 'random']: raise ValueError(f"Unsupported search method: {method}.")

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
        adapter_config.pop('patience_cfg', None)
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
            search_kwargs['param_grid'] = param_grid
        else:
            search_kwargs['param_distributions'] = param_grid
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
        if save_best_model and best_estimator_refit is not None:
            if self.artifact_repo and self.experiment_run_key_prefix:
                try:
                    score_str = f"cv_score{results.get('best_score', 0.0):.4f}".replace('.', 'p')
                    params_str_simple = "_".join(
                        [f"{k.split('__')[-1]}={v}" for k, v in sorted(results.get('best_params', {}).items())])
                    params_str_simple = re.sub(r'[<>:"/\\|?*]', '_', params_str_simple)[:50]
                    model_filename = f"{self.model_type.value}_best_{params_str_simple}_{score_str}.pt"
                    model_object_key = self._get_s3_object_key(run_id, model_filename)

                    model_state_dict = best_estimator_refit.module_.state_dict()
                    model_path_identifier = self.artifact_repo.save_model_state_dict(model_state_dict, model_object_key)

                    if model_path_identifier:
                        logger.info(f"Best refit model state_dict saved via repository to: {model_path_identifier}")
                    else:
                        logger.error(
                            f"Failed to save best refit model via repository for {run_id} to key {model_object_key}.")
                except Exception as e:
                    logger.error(f"Failed to save best refit model via repository: {e}", exc_info=True)
            else:
                logger.warning(
                    f"Model saving skipped for {run_id}: no artifact repository or base key prefix configured.")
        elif save_best_model and best_estimator_refit is None:
            logger.warning(f"save_best_model=True for {run_id} but no best estimator was found/refit.")
        results['saved_model_path'] = model_path_identifier
        # --- End Save Model ---

        # --- Save Results ---
        # The decision to save is now based on results_detail_level_override
        # If results_detail_level_override is explicitly 0, _save_results will skip JSON.
        # If None, it uses self.results_detail_level.
        # If self.results_detail_level is 0, it also skips JSON.
        # The summary CSV is always attempted by _save_results unless you add logic there to skip it for level 0.

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

        if method_lower == 'grid': inner_search_kwargs['param_grid'] = param_grid
        else:
            inner_search_kwargs['param_distributions'] = param_grid
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
            eval_params.update(params)
        else:
            logger.info(f"Using pipeline_v1 default parameters for CV evaluation.")
        module_dropout_rate = eval_params.pop('module__dropout_rate', None)
        eval_params.setdefault('module', self._get_model_class(self.model_type))
        eval_params.setdefault('module__num_classes', self.dataset_handler.num_classes)
        if module_dropout_rate is not None: eval_params['module__dropout_rate'] = module_dropout_rate

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
                     max_epochs: Optional[int] = None,
                     lr: Optional[float] = None,
                     batch_size: Optional[int] = None,
                     # Add other tunable params like weight_decay, dropout_rate here if needed
                     optimizer__weight_decay: Optional[float] = None,
                     module__dropout_rate: Optional[float] = None,
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
        # Override params for this run
        if max_epochs is not None: adapter_config['max_epochs'] = max_epochs
        if lr is not None: adapter_config['lr'] = lr
        if batch_size is not None: adapter_config['batch_size'] = batch_size
        if optimizer__weight_decay is not None: adapter_config['optimizer__weight_decay'] = optimizer__weight_decay
        if module__dropout_rate is not None: adapter_config['module__dropout_rate'] = module__dropout_rate
        # Set the train split strategy (None or PredefinedSplit via ValidSplit)
        adapter_config['train_split'] = train_split_config

        # --- Handle Callbacks based on validation ---
        # Start with the base callbacks list/None from the config
        current_callbacks = adapter_config.get('callbacks', [])
        if not isinstance(current_callbacks, list):  # Handle None case
            current_callbacks = []

        if train_split_config is None:
            logger.warning(
                "No validation set. Callbacks monitoring validation metrics (EarlyStopping, LRScheduler) may be removed or ineffective.")
            # Filter out callbacks that depend on validation
            adapter_config['callbacks'] = [
                (name, cb) for name, cb in current_callbacks
                if not isinstance(cb, (EarlyStopping, LRScheduler))  # Keep others
            ]
        else:
            # Keep all callbacks from base config when validation exists
            adapter_config['callbacks'] = current_callbacks
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
            if self.artifact_repo and self.experiment_run_key_prefix:
                try:
                    # Construct filename and then the full S3 object key or relative local path
                    val_metric_val = results.get('best_valid_metric_value', np.nan)
                    valid_loss_key = results.get('valid_metric_name', 'valid_loss')  # Get the actual key used
                    val_metric_str = f"val_{valid_loss_key.replace('_', '-')}{val_metric_val:.4f}" if not np.isnan(
                        val_metric_val) else "no_val"
                    model_filename = f"{self.model_type.value}_epoch{results.get('best_epoch', 0)}_{val_metric_str}.pt"

                    # Key for repository (S3 object key or relative path for local repo)
                    model_artifact_key = self._get_s3_object_key(run_id, model_filename)

                    model_state_dict = adapter_for_train.module_.state_dict()
                    model_path_identifier = self.artifact_repo.save_model_state_dict(model_state_dict,
                                                                                     model_artifact_key)

                    if model_path_identifier:
                        logger.info(f"Model state_dict saved via repository to: {model_path_identifier}")
                    else:
                        logger.error(
                            f"Failed to save model state_dict via repository for {run_id} to key {model_artifact_key}.")
                except Exception as e:
                    logger.error(f"Failed to save model via repository: {e}", exc_info=True)
            else:
                logger.warning(
                    f"Model saving skipped for {run_id}: no artifact repository or base key prefix configured.")
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
                        plot_artifact_base_key_or_path=plot_save_location_base,  # Pass base key/path for plots
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
                       image_sources: List[Union[str, Path, Image.Image, bytes]],
                       original_identifiers: Optional[List[str]] = None,
                       persist_prediction_artifacts: bool = True,
                       results_detail_level: Optional[int] = None,
                       plot_level: int = 0,
                       generate_lime_explanations: bool = False,
                       lime_num_features: int = 5,
                       lime_num_samples: int = 1000,
                       prediction_plot_max_cols: int = 4
                       ) -> List[Dict[str, Any]]:
        run_id = f"predict_images_{datetime.now().strftime('%Y%M%d_%H%M%S_%f')}"
        logger.info(f"Starting prediction ({run_id}) for {len(image_sources)} image sources...")
        if not persist_prediction_artifacts:
            logger.info(f"Prediction run {run_id}: Artifact persistence is DISABLED.")

        if not self.model_adapter.initialized_:
            raise RuntimeError("Model adapter not initialized. Train or load a model first.")
        if not image_sources:
            logger.warning("No image sources provided for prediction.")
            return []

        if original_identifiers and len(original_identifiers) != len(image_sources):
            logger.warning(
                "Mismatch: len(image_sources) != len(original_identifiers). Identifiers will be auto-generated.")
            original_identifiers = None  # Fallback to auto-generated

        if generate_lime_explanations and not LIME_AVAILABLE:
            logger.warning("LIME requested but library not available. Skipping LIME.")
            generate_lime_explanations = False
        elif generate_lime_explanations:
            logger.info(
                f"LIME explanations will be generated (num_features={lime_num_features}, num_samples={lime_num_samples}). This may take extra time per image.")

        pil_images_for_processing: List[Tuple[Optional[Image.Image], str]] = []  # (PIL_Image or None, identifier)

        for i, source in enumerate(image_sources):
            identifier = original_identifiers[i] if original_identifiers else f"source_{i}"
            pil_image: Optional[Image.Image] = None
            try:
                if isinstance(source, (str, Path)):
                    source_path = Path(source)
                    if str(source).startswith(('http://', 'https://')):
                        logger.debug(f"Downloading image from URL: {source}")
                        response = requests.get(str(source), timeout=10)
                        response.raise_for_status()
                        pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    elif source_path.exists() and source_path.is_file():
                        pil_image = Image.open(source_path).convert('RGB')
                    else:
                        logger.warning(f"Image path does not exist or is not a file: {source_path}")
                elif isinstance(source, Image.Image):
                    pil_image = source.convert('RGB') if source.mode != 'RGB' else source
                elif isinstance(source, bytes):
                    pil_image = Image.open(io.BytesIO(source)).convert('RGB')
                else:
                    logger.warning(f"Unsupported image source type: {type(source)} for identifier: {identifier}")
            except Exception as e:
                logger.error(
                    f"Failed to load image for identifier '{identifier}' from source '{str(source)[:100]}...': {e}")

            pil_images_for_processing.append((pil_image, identifier))

        valid_pil_images: List[Image.Image] = []
        valid_identifiers: List[str] = []

        for img, ident in pil_images_for_processing:
            if img is not None:
                valid_pil_images.append(img)
                valid_identifiers.append(ident)

        if not valid_pil_images:
            logger.error("No valid images could be loaded for prediction.")
            return []

        logger.info(f"Successfully loaded {len(valid_pil_images)} images for processing.")

        # --- Create a custom in-memory dataset for PIL images ---
        class InMemoryPILDataset(torch.utils.data.Dataset):
            def __init__(self, pil_images: List[Image.Image], identifiers: List[str], transform: Callable):
                self.pil_images = pil_images
                self.identifiers = identifiers  # Store identifiers to potentially pass them if needed
                self.transform = transform

            def __len__(self):
                return len(self.pil_images)

            def __getitem__(self, idx):
                img = self.pil_images[idx]
                identifier = self.identifiers[idx]  # Get corresponding identifier
                label_tensor = torch.tensor(-1, dtype=torch.long)  # Dummy label tensor
                try:
                    transformed_img = self.transform(img)
                    # Optionally, return identifier if collate_fn is adapted
                    return transformed_img, label_tensor  # , identifier
                except Exception as e:
                    logger.warning(f"Transform failed for image '{identifier}' (index {idx}): {e}")
                    return None, label_tensor  # , identifier

        eval_transform = self.dataset_handler.get_eval_transform()
        prediction_dataset = InMemoryPILDataset(
            pil_images=valid_pil_images,
            identifiers=valid_identifiers,  # Pass identifiers
            transform=eval_transform
        )

        dataloader = torch.utils.data.DataLoader(
            prediction_dataset,
            batch_size=self.model_adapter_config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,
            collate_fn=PathImageDataset.collate_fn  # This collate_fn filters Nones
        )

        all_probabilities_np: List[np.ndarray] = []
        # To map back results if collate_fn filters items, we need to know which items passed
        # This is complex if collate_fn is generic. Alternative: dataset returns (img, label, original_idx_or_id)
        # For now, assume collate_fn filters, and we map back based on order of non-None items.
        # This requires that the order of 'valid_identifiers' matches the order of items *before* collation filtering.

        # Let's create a list of identifiers that correspond to the images *after* potential transform failures
        # that would lead to `None` images being passed to collate_fn.
        # The `dataloader` will only yield batches from successfully transformed images.

        # Create a temporary list of indices for images that are successfully transformed and batched
        successful_indices_after_transform_and_collation = []
        temp_dataloader_for_indices = torch.utils.data.DataLoader(
            InMemoryPILDataset(pil_images=valid_pil_images, identifiers=valid_identifiers, transform=eval_transform),
            batch_size=self.model_adapter_config.get('batch_size', 32),
            shuffle=False, num_workers=0,
            collate_fn=lambda batch: [item[0] is not None for item in batch]  # just check if image is not None
        )

        temp_idx_counter = 0
        for batch_success_flags in temp_dataloader_for_indices:
            for success in batch_success_flags:
                if success:
                    successful_indices_after_transform_and_collation.append(temp_idx_counter)
                temp_idx_counter += 1
        # Now successful_indices_after_transform_and_collation contains indices relative to valid_pil_images

        self.model_adapter.module_.eval()
        with torch.no_grad():
            for batch_images, _ in dataloader:  # This dataloader uses the filtering collate_fn
                if batch_images is None or len(batch_images) == 0:
                    # This should ideally not happen if collate_fn returns empty tensors for empty batches
                    continue
                batch_images = batch_images.to(self.model_adapter.device)
                logits = self.model_adapter.module_(batch_images)
                probabilities = torch.softmax(logits, dim=1)
                all_probabilities_np.extend(probabilities.cpu().numpy())

        predictions_output = []  # This will be returned
        class_names = self.dataset_handler.classes
        num_classes_total = self.dataset_handler.num_classes

        lime_explainer = None
        if generate_lime_explanations and LIME_AVAILABLE:
            lime_explainer = LimeImageExplainer(random_state=RANDOM_SEED)

            # Define lime_predict_fn locally (needs self)
            def lime_predict_fn(numpy_images_batch_lime):
                # ... (lime_predict_fn implementation) ...
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

        # Build prediction output and generate LIME data if needed
        min_len = len(all_probabilities_np)  # Assume mapping logic from previous step is correct
        for i in range(min_len):
            original_valid_idx = successful_indices_after_transform_and_collation[i]
            current_identifier = valid_identifiers[original_valid_idx]
            pil_image_for_lime_and_plot = valid_pil_images[original_valid_idx]  # The actual PIL image
            probs_np = all_probabilities_np[i]
            predicted_idx = int(np.argmax(probs_np))
            predicted_name = class_names[predicted_idx] if class_names and 0 <= predicted_idx < len(
                class_names) else f"Class_{predicted_idx}"
            # ... (top-k prediction logic) ...
            top_k_val = min(3, num_classes_total);
            top_k_indices = np.argsort(probs_np)[-top_k_val:][::-1];
            top_k_preds_list = []
            for k_idx in top_k_indices: k_name = class_names[k_idx] if class_names and 0 <= k_idx < len(
                class_names) else f"Class_{k_idx}"; top_k_preds_list.append((k_name, float(probs_np[k_idx])))

            pred_item = {
                'identifier': current_identifier,
                'image_path': str(current_identifier) if isinstance(current_identifier, (str, Path)) and Path(
                    current_identifier).is_file() else "in-memory/url",
                'probabilities': probs_np.tolist(),
                'predicted_class_idx': predicted_idx,
                'predicted_class_name': predicted_name,
                'confidence': float(probs_np[predicted_idx]),
                'top_k_predictions': top_k_preds_list,
                'lime_explanation': None  # Initialize
            }

            if generate_lime_explanations and lime_explainer is not None:
                logger.debug(f"Generating LIME explanation for: {current_identifier} (Predicted: {predicted_name})")
                lime_data_for_output = {'error': 'LIME generation failed or skipped.'}
                try:
                    img_np_for_lime = np.array(pil_image_for_lime_and_plot)
                    explanation = lime_explainer.explain_instance(
                        image=img_np_for_lime, classifier_fn=lime_predict_fn, top_labels=1, hide_color=0,
                        num_features=lime_num_features, num_samples=lime_num_samples, random_seed=RANDOM_SEED
                    )
                    lime_weights = explanation.local_exp.get(predicted_idx, [])
                    lime_data_for_output = {
                        'explained_class_idx': predicted_idx,
                        'explained_class_name': predicted_name,
                        'feature_weights': lime_weights,
                        # We need segments for the server/plotter to render the LIME image.
                        # This will be part of the returned data, but NOT necessarily saved to JSON by default.
                        'segments_for_render': explanation.segments.tolist(),
                        'num_features_from_lime_run': lime_num_features
                    }

                    # --- Generate and Store LIME Image if MinIO Repo is used and persistence is on ---
                    if persist_prediction_artifacts and isinstance(self.artifact_repo, MinIORepository) and \
                            self.experiment_run_key_prefix and SKIMAGE_AVAILABLE and mark_boundaries:
                        logger.debug(f"Generating LIME image for S3 storage for {current_identifier}")
                        temp_lime_mask = np.zeros(explanation.segments.shape, dtype=bool)
                        positive_features_lime = sorted([(seg_id, w) for seg_id, w in lime_weights if w > 0],
                                                        key=lambda x: x[1], reverse=True)
                        feats_shown_count = 0
                        for seg_id, weight in positive_features_lime:
                            if feats_shown_count < lime_num_features:
                                temp_lime_mask[explanation.segments == seg_id] = True; feats_shown_count += 1
                            else:
                                break

                        if np.any(temp_lime_mask):
                            img_norm_lime = img_np_for_lime.astype(
                                float) / 255.0 if img_np_for_lime.max() > 1.0 else img_np_for_lime.astype(float)
                            lime_viz_np = mark_boundaries(img_norm_lime, temp_lime_mask, color=(1, 0, 0), mode='thick',
                                                          outline_color=(1, 0, 0))
                            lime_viz_pil = Image.fromarray((lime_viz_np * 255).astype(np.uint8))

                            lime_img_buffer = io.BytesIO()
                            lime_viz_pil.save(lime_img_buffer, format="PNG")
                            lime_img_buffer.seek(0)

                            # Construct S3 key for this LIME image.
                            # Example: <experiment_run_key_prefix>/<predict_run_id>/lime_explanations/<identifier_clean>.png
                            # The identifier might have slashes if it's a path, clean it.
                            safe_identifier_fname = re.sub(r'[\\/*?:"<>|]', "_",
                                                           str(pred_item['identifier']))  # Sanitize
                            if len(safe_identifier_fname) > 100: safe_identifier_fname = safe_identifier_fname[
                                                                                         -100:]  # Truncate

                            lime_img_s3_key = str((PurePath(
                                self.experiment_run_key_prefix) / run_id / "lime_visualizations" / f"{safe_identifier_fname}_lime.png").as_posix())

                            saved_lime_path = self.artifact_repo.save_image_object(
                                lime_img_buffer.getvalue(),
                                # self.artifact_repo.bucket_name, # Repo method should know its bucket
                                lime_img_s3_key,
                                content_type='image/png'
                            )
                            if saved_lime_path:
                                lime_data_for_output['s3_lime_image_key'] = saved_lime_path  # Store S3 path
                                logger.info(f"LIME visualization for {current_identifier} saved to: {saved_lime_path}")
                            else:
                                logger.error(f"Failed to save LIME visualization for {current_identifier} to S3.")
                        else:
                            logger.debug(
                                f"LIME: No positive features to highlight for {current_identifier}, not saving LIME image to S3.")


                except Exception as lime_e:
                    logger.error(f"LIME explanation processing failed for {current_identifier}: {lime_e}",
                                 exc_info=False)
                    lime_data_for_output = {'error': str(lime_e)}
                pred_item['lime_explanation'] = lime_data_for_output

            predictions_output.append(pred_item)

        logger.info(f"Successfully generated predictions for {len(predictions_output)} images.")

        # --- Optionally Save Prediction Results (JSON) ---
        if persist_prediction_artifacts:
            effective_save_detail_level = self.results_detail_level
            if results_detail_level is not None:
                effective_save_detail_level = results_detail_level

            if effective_save_detail_level > 0:
                logger.info(
                    f"Saving prediction results JSON ({run_id}) with detail level {effective_save_detail_level}...")

                # Create a copy of predictions_output for JSON, excluding 'segments_for_render'
                # if we want to keep the JSON smaller, as it's mainly for the plotter's immediate use.
                predictions_for_json = []
                for item in predictions_output:
                    item_copy = item.copy()
                    if 'lime_explanation' in item_copy and isinstance(item_copy['lime_explanation'], dict):
                        item_copy['lime_explanation'] = {k: v for k, v in item_copy['lime_explanation'].items() if
                                                         k != 'segments_for_render'}
                    predictions_for_json.append(item_copy)

                results_for_json = {
                    'method': 'predict_images', 'run_id': run_id,
                    'params': {'num_original_sources': len(image_sources), 'num_valid_loaded': len(valid_pil_images),
                               'num_images_predicted': len(predictions_output)},
                    'predictions': predictions_for_json  # Save the version without segments
                }
                self._save_results(  # This method uses self.artifact_repo internally
                    results_data=results_for_json, method_name="predict_images", run_id=run_id,
                    method_params=results_for_json['params'],
                    results_detail_level=effective_save_detail_level
                )
            else:
                logger.info(f"JSON results for {run_id} not saved (detail level 0).")
        else:
            logger.info(f"JSON results for {run_id} not saved (persist_prediction_artifacts is False).")

        # --- Plotting (Optional) ---
        current_plot_level = self.plot_level
        if plot_level is not None: current_plot_level = plot_level

        if current_plot_level > 0 and predictions_output:
            plot_save_location_base_for_run: Optional[str] = None
            # Only define a save location if persistence is enabled AND repo is configured
            if persist_prediction_artifacts and self.artifact_repo and self.experiment_run_key_prefix:
                plot_save_location_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())

            can_save_plots = (current_plot_level >= 1 and plot_save_location_base_for_run is not None)
            should_show_plots_flag = (current_plot_level == 2)

            if not can_save_plots and current_plot_level == 1:
                logger.warning(f"Plot saving for predictions {run_id} skipped: plot_level 1 but no save location.")

            if can_save_plots or should_show_plots_flag:
                logger.info(f"Plotting prediction results for {run_id} (plot level {current_plot_level}).")
                try:
                    from ..plotter import ResultsPlotter
                    identifier_to_pil_map = {ident: img for ident, img in zip(valid_identifiers, valid_pil_images)}
                    ResultsPlotter.plot_predictions(
                        predictions_output=predictions_output,  # Contains LIME data including segments_for_render
                        image_pil_map=identifier_to_pil_map,
                        plot_save_dir_base=plot_save_location_base_for_run,
                        repository_for_plots=self.artifact_repo if can_save_plots else None,
                        show_plots=should_show_plots_flag,
                        max_cols=prediction_plot_max_cols,
                        generate_lime_plots=generate_lime_explanations,  # This tells plotter to look for LIME data
                        lime_num_features_to_display=lime_num_features  # Use the one from predict_images
                    )
                except ImportError:
                    logger.error("Plotting skipped: ResultsPlotter/LIME class not found or libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Prediction plotting failed for {run_id}: {plot_err}", exc_info=True)
        elif current_plot_level > 0 and not predictions_output:
            logger.warning(f"Plotting skipped for predictions run {run_id}: No prediction outputs.")

        return predictions_output

    def load_model(self, model_path_or_key: Union[str, Path]) -> None:
        """
        Loads a state_dict into the pipeline's model adapter.
        The model_path_or_key can be a local file system path or an S3 object key
        if an artifact_repository is configured.
        """
        logger.info(f"Attempting to load model state_dict from: {model_path_or_key}")

        if not self.model_adapter.initialized_:
            logger.debug("Initializing skorch adapter before loading state_dict...")
            try:
                self.model_adapter.initialize()
            except Exception as e:
                raise RuntimeError("Could not initialize model adapter for loading.") from e

        if not hasattr(self.model_adapter, 'module_') or not isinstance(self.model_adapter.module_, nn.Module):
            raise RuntimeError("Adapter missing internal nn.Module ('module_'). Cannot load state_dict.")

        state_dict: Optional[Dict] = None
        map_location = self.model_adapter.device  # Determine map_location once

        # Try loading via artifact repository if available
        if self.artifact_repo:
            logger.debug(f"Attempting to load model via repository: {type(self.artifact_repo).__name__}")
            # Assume model_path_or_key is the key for the repository
            state_dict = self.artifact_repo.load_model_state_dict(str(model_path_or_key), map_location=map_location)
            if state_dict:
                logger.info(f"Model successfully loaded via repository from key/path: {model_path_or_key}")
            else:
                logger.warning(f"Failed to load model via repository from key/path: {model_path_or_key}. "
                               "Will attempt fallback to local filesystem if it's a path.")

        # Fallback or direct local file load if no repo or repo load failed (and it's a path)
        if state_dict is None:
            local_model_path = Path(model_path_or_key)
            if self.artifact_repo:  # If repo load failed, now try as local path.
                logger.info(
                    f"Repository load failed or not applicable for '{model_path_or_key}', trying as local path: {local_model_path}")

            if local_model_path.is_file():
                logger.debug(f"Attempting to load model from local filesystem path: {local_model_path}")
                try:
                    state_dict = torch.load(local_model_path, map_location=map_location, weights_only=True)
                    logger.info(f"Model successfully loaded from local filesystem: {local_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model from local filesystem path {local_model_path}: {e}",
                                 exc_info=True)
                    # No state_dict loaded, error will be raised below
            elif not self.artifact_repo:  # No repo and not a local file
                logger.error(f"Model file not found at local path: {local_model_path} (and no repository configured).")
                # No state_dict loaded, error will be raised below

        # Apply state_dict if successfully loaded
        if state_dict:
            try:
                self.model_adapter.module_.load_state_dict(state_dict)
                self.model_adapter.module_.eval()  # Set to eval mode
                logger.info(
                    f"Model state_dict loaded and applied successfully to the model adapter from: {model_path_or_key}")
            except Exception as e:
                logger.error(f"Failed to apply loaded state_dict to model: {e}", exc_info=True)
                if isinstance(e, RuntimeError) and "size mismatch" in str(e):
                    logger.error("Architecture mismatch likely. Ensure loaded weights match the current model config.")
                raise RuntimeError(f"Error applying state_dict from '{model_path_or_key}' to model.") from e
        else:
            # This means all attempts to load failed
            raise FileNotFoundError(f"Model could not be loaded from source: {model_path_or_key}. "
                                    "Check path/key and repository configuration.")
