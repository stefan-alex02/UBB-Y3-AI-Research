import contextlib
import io  # For capturing print output to a string
import json
import re
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Any, Type, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, make_scorer,
    roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, train_test_split, PredefinedSplit
)
from skorch.callbacks import Callback
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

from ..config import RANDOM_SEED, DEVICE, DEFAULT_IMG_SIZE, ModelType
from ..dataset_utils import ImageDatasetHandler, DatasetStructure, PathImageDataset
from ..logger_utils import logger
from ..plotter import ResultsPlotter, _create_plot_dir_from_base
from ..skorch_utils import SkorchModelAdapter
from ..skorch_utils import get_default_callbacks
from ..architectures import *


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
                 results_dir: Union[str, Path] = 'results',
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
                results_dir: Base directory where experiment results will be saved.
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

        self.results_dir = Path(results_dir) if results_dir else None
        if self.results_dir:
            base_results_dir = self.results_dir.resolve()
            dataset_name = self.dataset_path.name
            timestamp_init = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_dir = base_results_dir / dataset_name / self.model_type.value / f"{timestamp_init}_seed{RANDOM_SEED}"
            # Directory creation for experiment_dir itself will be handled by setup_logger or first _save_results
            # However, setup_logger needs it for log file path, so it's better if Executor creates it.
            # OR, if self.experiment_dir is None, logger is console only.
            # PipelineExecutor will handle logger setup using this.
            logger.info(f"  Base experiment results dir: {self.experiment_dir}") # Logged by executor
        else:
            self.experiment_dir = None  # Crucial for disabling file outputs
            logger.info("  Results Directory: None (file outputs disabled)")

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
            'criterion': nn.CrossEntropyLoss, 'optimizer': torch.optim.AdamW,
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

    @staticmethod
    def _get_model_class(model_type_enum: ModelType) -> Type[nn.Module]:
        model_mapping = {
            ModelType.CNN: SimpleCNN,
            ModelType.SIMPLE_VIT: SimpleViT,  # Kept SimpleViT
            ModelType.FLEXIBLE_VIT: FlexibleViT,  # Added FlexibleViT
            ModelType.DIFFUSION: DiffusionClassifier
        }
        model_class = model_mapping.get(model_type_enum) # Direct lookup using Enum member
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
        if not self.experiment_dir:
            logger.info(f"File saving disabled for {run_id} (no results directory). Returning results in memory only.")
            return None # Do not proceed with file saving

        method_params = method_params or {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        method_dir = self.experiment_dir / run_id
        method_dir.mkdir(parents=True, exist_ok=True)

        current_detail_level = self.results_detail_level
        if results_detail_level is not None:
            current_detail_level = results_detail_level
            logger.debug(f"Results detail level overridden for this run to: {current_detail_level}")
        else:
            logger.debug(f"Using pipeline results detail level for this run: {current_detail_level}")

        # --- LEVEL 0: NO JSON SAVING, ONLY SUMMARY CSV ---
        if current_detail_level == 0:
            logger.info(f"Results detail level 0: Skipping JSON results saving for {run_id}.")
        else:
            logger.debug(f"Saving results for {run_id} to: {method_dir} (detail level: {current_detail_level})")
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

                        if is_list_of_histories:  # e.g. outer_fold_best_model_histories
                            for single_history in source_list:
                                if isinstance(single_history, list):
                                    cleaned_single_history = []
                                    for epoch_data in single_history:
                                        if isinstance(epoch_data, dict):
                                            epoch_copy = {k: v for k, v in epoch_data.items() if k != 'batches'}
                                            cleaned_single_history.append(epoch_copy)
                                        else:
                                            cleaned_single_history.append(epoch_data)
                                    cleaned_history_list.append(cleaned_single_history)
                                else:
                                    cleaned_history_list.append(
                                        single_history)  # Append None or other non-list items as is
                        else:  # A single history list (e.g., training_history)
                            for epoch_data in source_list:
                                if isinstance(epoch_data, dict):
                                    epoch_copy = {k: v for k, v in epoch_data.items() if k != 'batches'}
                                    cleaned_history_list.append(epoch_copy)
                                else:
                                    cleaned_history_list.append(epoch_data)
                        results_to_save[hist_key] = cleaned_history_list

            # Level 3 includes everything already collected, including batch data in histories (as it wasn't stripped).

            logger.debug(
                f"Final keys to save in JSON for level {current_detail_level}: {list(results_to_save.keys())}")

            # --- Save results JSON ---
            json_filename = f"{method_name}_results.json"
            json_filepath_local = method_dir / json_filename
            try:
                # Your json_serializer function (ensure it's defined or imported)
                def json_serializer(obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj) if not np.isnan(obj) else None
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, Path):
                        return str(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, (slice, type, Callable)):
                        return None
                    elif isinstance(obj, (torch.optim.Optimizer, nn.Module, Callback)):
                        return str(type(obj).__name__)
                    elif isinstance(obj, ValidSplit):
                        return f"ValidSplit(cv={obj.cv}, stratified={obj.stratified})"
                    try:
                        return json.JSONEncoder.default(None, obj)
                    except TypeError:
                        return str(obj)

                with open(json_filepath_local, 'w', encoding='utf-8') as f:
                    json.dump(results_to_save, f, indent=4, default=json_serializer)
                logger.info(f"Results JSON saved to: {json_filepath_local}")
                json_filepath = json_filepath_local # Assign to outer scope variable on success
            except OSError as oe:
                logger.error(f"OS Error saving results JSON {json_filepath_local}: {oe}", exc_info=True)
            except Exception as e:
                logger.error(f"Failed to save results JSON {json_filepath_local}: {e}", exc_info=True)

        # --- Prepare and save summary CSV (always attempted if not returned early for level 0) ---
        csv_filepath = self.experiment_dir / f"summary_results_seed{RANDOM_SEED}.csv"
        try:
            agg_metrics = results_data.get('aggregated_metrics', {})
            macro_avg = results_data.get('macro_avg', {})
            overall_acc = np.nan
            macro_f1 = np.nan

            if agg_metrics:
                overall_acc = agg_metrics.get('accuracy', {}).get('mean', np.nan)
                macro_f1 = agg_metrics.get('f1_macro', {}).get('mean', np.nan)
            else:
                overall_acc = results_data.get('overall_accuracy', np.nan)
                macro_f1 = macro_avg.get('f1', np.nan) if isinstance(macro_avg, dict) else np.nan

            summary_params = {}
            allowed_types = (str, int, float, bool)
            key_cv_params = ['cv', 'outer_cv', 'inner_cv', 'n_iter', 'internal_val_split_ratio', 'confidence_level',
                             'evaluated_on']
            if method_params:
                for k, v in method_params.items():
                    if isinstance(v, allowed_types) or k in key_cv_params:
                        summary_params[k] = v
            if 'best_params' in results_data and isinstance(results_data['best_params'], dict):
                for k, v in results_data['best_params'].items():
                    if isinstance(v, allowed_types): summary_params[f'best_{k}'] = v

            summary = {
                'method_run_id': run_id, 'timestamp': timestamp,
                'accuracy': overall_acc, 'macro_f1': macro_f1,
                'accuracy_ci_margin': agg_metrics.get('accuracy', {}).get('margin_of_error', np.nan),
                'f1_macro_ci_margin': agg_metrics.get('f1_macro', {}).get('margin_of_error', np.nan),
                'precision_macro': agg_metrics.get('precision_macro', {}).get('mean', macro_avg.get('precision', np.nan) if isinstance(macro_avg, dict) else np.nan),
                'recall_macro': agg_metrics.get('recall_macro', {}).get('mean', macro_avg.get('recall', np.nan) if isinstance(macro_avg, dict) else np.nan),
                'specificity_macro': agg_metrics.get('specificity_macro', {}).get('mean', macro_avg.get('specificity', np.nan) if isinstance(macro_avg, dict) else np.nan),
                'roc_auc_macro': agg_metrics.get('roc_auc_macro', {}).get('mean', macro_avg.get('roc_auc', np.nan) if isinstance(macro_avg, dict) else np.nan),
                'pr_auc_macro': agg_metrics.get('pr_auc_macro', {}).get('mean', macro_avg.get('pr_auc', np.nan) if isinstance(macro_avg, dict) else np.nan),
                'best_cv_score': results_data.get('best_score', np.nan),
                'best_epoch': results_data.get('best_epoch', np.nan),
                **summary_params
            }
            summary = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in summary.items()}

            df_summary = pd.DataFrame([summary])
            file_exists = csv_filepath.exists()
            df_summary.to_csv(csv_filepath, mode='a', header=not file_exists, index=False, encoding='utf-8')
            logger.info(f"Summary results updated in: {csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to save summary results to CSV {csv_filepath}: {e}", exc_info=True)

        return json_filepath  # <<< RETURN PATH or None


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
        run_id = f"non_nested_{method_lower}_{datetime.now().strftime('%H%M%S_%f')}"
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
        model_path_str = None
        if save_best_model and best_estimator_refit is not None and self.experiment_dir:
            try:
                # Results for this method run will be in experiment_dir / run_id /
                method_run_dir = self.experiment_dir / run_id
                method_run_dir.mkdir(parents=True, exist_ok=True)

                # Construct filename using best score or params
                score_str = f"cv_score{results.get('best_score', 0.0):.4f}".replace('.', 'p')
                params_str_simple = "_".join([f"{k.split('__')[-1]}={v}" for k, v in sorted(results.get('best_params', {}).items())])
                params_str_simple = re.sub(r'[<>:"/\\|?*]', '_', params_str_simple)[:50] # Sanitize and shorten

                model_filename = f"{self.model_type.value}_best_{params_str_simple}_{score_str}.pt"
                model_path = method_run_dir / model_filename

                torch.save(best_estimator_refit.module_.state_dict(), model_path)
                model_path_str = str(model_path)
                logger.info(f"Best refit model state_dict saved to: {model_path_str}")
                results['saved_model_path'] = model_path_str
            except Exception as e:
                 logger.error(f"Failed to save best refit model: {e}", exc_info=True)
                 results['saved_model_path'] = None
        elif save_best_model: # save_best_model is True but no estimator
             logger.warning("save_best_model=True but no best estimator was found/refit.")
             results['saved_model_path'] = None
        # --- End Save Model ---

        # --- Save Results ---
        # The decision to save is now based on results_detail_level_override
        # If results_detail_level_override is explicitly 0, _save_results will skip JSON.
        # If None, it uses self.results_detail_level.
        # If self.results_detail_level is 0, it also skips JSON.
        # The summary CSV is always attempted by _save_results unless you add logic there to skip it for level 0.

        summary_params = results['params'].copy()
        summary_params.update({f"best_{k}": v for k, v in results.get('best_params', {}).items() if isinstance(v, (str, int, float, bool))})
        saved_json_path = self._save_results(results, f"non_nested_{method_lower}_search",
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
            plot_save_dir: Optional[Path] = None
            if self.experiment_dir:
                plot_save_dir = self.experiment_dir / run_id

            if current_plot_level == 1 and not plot_save_dir:
                logger.warning(f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no results_dir for pipeline.")
            else:
                logger.info(f"Plotting non_nested_cv results for {run_id} (plot level {current_plot_level}).")
                show_plots_flag = (current_plot_level == 2)
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_non_nested_cv_results(
                        results_data=results,
                        plot_save_dir_base=plot_save_dir,
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
        run_id = f"{method_lower}_{datetime.now().strftime('%H%M%S')}"  # Generate ID
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
                plot_save_dir: Optional[Path] = None
                if self.experiment_dir:
                    plot_save_dir = self.experiment_dir / run_id

                if current_plot_level == 1 and not plot_save_dir:
                    logger.warning(
                        f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no results_dir for pipeline.")
                else:
                    logger.info(f"Plotting nested_cv results for {run_id} (plot level {current_plot_level}).")
                    show_plots_flag = (current_plot_level == 2)
                    try:
                        from ..plotter import ResultsPlotter
                        ResultsPlotter.plot_nested_cv_results(
                            results_data=results,
                            plot_save_dir_base=plot_save_dir,
                            show_plots=show_plots_flag
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
            save_results: Whether to save JSON results and update summary CSV.
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

        run_id = f"cv_model_evaluation_{evaluate_on}_{datetime.now().strftime('%H%M%S_%f')}"  # Include mode in ID

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
        eval_params.setdefault('module', self._get_model_class(self.model_type.value))
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
            plot_save_dir_base: Optional[Path] = None # Base directory for the plotter to create its '_plots' subdir

            if self.experiment_dir: # If the pipeline has a results directory configured
                plot_save_dir_base = self.experiment_dir / run_id
            # If self.experiment_dir is None, plot_save_dir_base remains None.

            # Condition to actually plot:
            # - If level is 2 (show plots), we always attempt to plot (plotter handles saving if dir exists).
            # - If level is 1 (save only), we only plot if plot_save_dir_base is available.
            can_plot_to_file = (current_plot_level == 1 and plot_save_dir_base is not None)
            should_show_plot = (current_plot_level == 2)

            if can_plot_to_file or should_show_plot:
                logger.info(f"Plotting cv_model_evaluation results for {run_id} (plot level {current_plot_level}).")
                try:
                    from ..plotter import ResultsPlotter # Corrected relative import
                    ResultsPlotter.plot_cv_model_evaluation_results(
                        results_data=results, # Pass the in-memory results
                        class_names=self.dataset_handler.classes,
                        plot_save_dir_base=plot_save_dir_base, # Pass base dir for plots (can be None)
                        show_plots=should_show_plot # show_plots_flag was defined based on current_plot_level
                    )
                except ImportError:
                     logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Plotting failed for {run_id}: {plot_err}", exc_info=True)
            elif current_plot_level == 1 and not plot_save_dir_base: # Explicitly log why save-only plot is skipped
                logger.warning(f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no results_dir for pipeline, so no save location for plots.")

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
        # --- Generate unique run_id for this execution ---
        run_id = f"single_train_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

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
        model_path_str = None  # Initialize path string
        if save_model and self.experiment_dir:
            try:
                method_run_dir = self.experiment_dir / run_id
                method_run_dir.mkdir(parents=True, exist_ok=True)

                val_metric_val = results.get('best_valid_metric_value', np.nan)
                val_metric_str = f"val_{valid_loss_key.replace('_', '-')}{val_metric_val:.4f}" if not np.isnan(
                    val_metric_val) else "no_val"
                model_filename = f"{self.model_type.value}_epoch{results.get('best_epoch', 0)}_{val_metric_str}.pt"
                model_path = method_run_dir / model_filename

                torch.save(adapter_for_train.module_.state_dict(), model_path)
                model_path_str = str(model_path)
                logger.info(f"Model state_dict saved to: {model_path_str}")
                results['saved_model_path'] = model_path_str
            except Exception as e:
                logger.error(f"Failed to save model: {e}", exc_info=True)
                results['saved_model_path'] = None

        # --- Update main adapter and save results ---
        self.model_adapter = adapter_for_train  # Keep the trained adapter
        logger.info("Main pipeline_v1 model adapter updated with the model from single_train.")

        # Add summary metrics for saving logic
        results['accuracy'] = results.get('valid_acc_at_best', np.nan)  # Use valid acc if available for summary
        results['macro_avg'] = {}  # No macro avg from training phase

        # --- Save results ---
        # Prepare simple params for summary report
        simple_params = {k: v for k, v in adapter_config.items() if isinstance(v, (str, int, float, bool))}
        simple_params['val_split_ratio_used'] = current_val_split_ratio if train_split_config else 0.0
        # Pass run_id generated at the start
        saved_json_path = self._save_results(results, "single_train",
                           run_id=run_id,
                           method_params=simple_params,
                           results_detail_level=results_detail_level)

        # --- Determine effective plot level ---
        current_plot_level = self.plot_level  # Start with pipeline default
        if plot_level is not None:
            current_plot_level = plot_level  # Use override if provided
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        # --- Plot results (conditionally) ---
        if current_plot_level > 0:
            plot_save_dir: Optional[Path] = None
            if self.experiment_dir:  # We need a base save directory if plots are to be saved
                plot_save_dir = self.experiment_dir / run_id  # Plots go into the method's run_id folder
                # _create_plot_dir_from_base will add "_plots"

            if current_plot_level == 1 and not plot_save_dir:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no results_dir for pipeline.")
            else:
                logger.info(f"Plotting single_train results for {run_id} (plot level {current_plot_level}).")
                show_plots_flag = (current_plot_level == 2)
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_single_train_results(
                        results_data=results,
                        plot_save_dir_base=plot_save_dir,
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
        run_id = f"single_eval_{datetime.now().strftime('%H%M%S_%f')}" # Generate run_id here

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
        results = {'method': 'single_eval', 'params': {}, **metrics}

        run_id_for_save = f"single_eval_{datetime.now().strftime('%H%M%S')}"
        saved_json_path = self._save_results(results, "single_eval",
                           method_params=results['params'],
                           run_id=run_id_for_save,
                           results_detail_level=results_detail_level)

        # --- Plot results (conditionally) ---
        if current_plot_level > 0:
            plot_save_dir: Optional[Path] = None
            if self.experiment_dir: # If saving is possible
                plot_save_dir = self.experiment_dir / run_id

            if current_plot_level == 1 and not plot_save_dir:
                logger.warning(f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no results_dir for pipeline.")
            else:
                logger.info(f"Plotting single_eval results for {run_id} (plot level {current_plot_level}).")
                show_plots_flag = (current_plot_level == 2)
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_single_eval_results(
                        results_data=results,
                        class_names=self.dataset_handler.classes,
                        plot_save_dir_base=plot_save_dir,
                        show_plots=show_plots_flag
                    )
                except ImportError:
                     logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Plotting failed for {run_id}: {plot_err}", exc_info=True)

        return results

    def predict_images(self,
                       image_paths: List[Union[str, Path]],
                       results_detail_level: Optional[int] = None,  # For saving results
                       plot_level: int = 0,
                       # Optional: Add max_cols for plotting here if you want to control it via method params
                       # prediction_plot_max_cols: Optional[int] = None
                       ) -> List[Dict[str, Any]]:
        """
        Performs prediction on a list of new images.

        Args:
            image_paths: A list of file paths to the images to be predicted.
            results_detail_level_override: Controls saving of prediction results to a JSON file.
                If > 0, results are saved. Levels 1-3 behave like other methods regarding
                what's included if more data were to be added to the 'results_for_json' dict.
                Currently, it saves the 'predictions_output' list.
                If None, uses pipeline's default self.results_detail_level.
            plot_level: Level for plotting prediction results (0: no plot, 1: save, 2: save & show).
            # prediction_plot_max_cols_override: Optional override for max columns in prediction plot.

        Returns:
            A list of dictionaries, where each dictionary contains prediction details for an image.
        """
        run_id = f"predict_images_{datetime.now().strftime('%Y%M%d_%H%M%S_%f')}"
        logger.info(f"Starting prediction ({run_id}) for {len(image_paths)} images...")

        if not self.model_adapter.initialized_:
            raise RuntimeError("Model adapter not initialized. Train or load a model first.")
        if not image_paths:
            logger.warning("No image paths provided for prediction.")
            return []

        # Ensure image_paths are Path objects and filter out non-existent ones upfront
        valid_image_paths = []
        for p_str in image_paths:
            p = Path(p_str)
            if p.exists() and p.is_file():
                valid_image_paths.append(p)
            else:
                logger.warning(f"Image path does not exist or is not a file, skipping: {p}")

        if not valid_image_paths:
            logger.error("No valid image paths remaining after checking existence.")
            return []

        logger.info(f"Found {len(valid_image_paths)} valid image paths for prediction.")

        eval_transform = self.dataset_handler.get_eval_transform()
        # Use only valid_image_paths for the dataset
        prediction_dataset = PathImageDataset(paths=valid_image_paths, labels=None, transform=eval_transform)

        dataloader = torch.utils.data.DataLoader(
            prediction_dataset,
            batch_size=self.model_adapter_config.get('batch_size', 32),
            shuffle=False,
            num_workers=0,  # Consistent with other parts of your code
            collate_fn=PathImageDataset.collate_fn
        )

        all_probabilities_np: List[np.ndarray] = []  # Store as list of numpy arrays
        processed_image_indices_in_batch: List[List[int]] = []  # To track original indices if collate_fn filters

        self.model_adapter.module_.eval()
        with torch.no_grad():
            current_original_idx = 0
            for batch_images, _ in dataloader:  # Labels will be dummy from PathImageDataset if None was passed
                num_expected_in_batch = min(dataloader.batch_size, len(valid_image_paths) - current_original_idx)

                if batch_images is None or len(batch_images) == 0:
                    logger.warning(
                        f"A batch (expecting {num_expected_in_batch} images starting from original index {current_original_idx}) "
                        f"resulted in no processable images after collation. These images will be skipped.")
                    # Advance current_original_idx by the number of images this batch *should* have processed
                    current_original_idx += num_expected_in_batch
                    continue  # Skip to next batch

                batch_images = batch_images.to(self.model_adapter.device)
                logits = self.model_adapter.module_(batch_images)
                probabilities = torch.softmax(logits, dim=1)

                # Store probabilities
                batch_probs_np = probabilities.cpu().numpy()
                all_probabilities_np.extend(batch_probs_np)

                # Track which original images were successfully processed in this batch
                # This assumes PathImageDataset.__getitem__ returns (None,None) for bad images
                # and collate_fn filters them, so len(batch_images) is num successfully processed.
                # This part is tricky if PathImageDataset doesn't perfectly track original indices through errors.
                # For now, assume successful collation means they correspond to the next set of valid_image_paths.
                # A more robust way might be for PathImageDataset to return the original index.
                indices_this_batch = list(range(current_original_idx, current_original_idx + len(batch_images)))
                processed_image_indices_in_batch.append(indices_this_batch)
                current_original_idx += num_expected_in_batch  # Advance by expected, collate handles errors within

        # Process results
        predictions_output = []
        class_names = self.dataset_handler.classes
        num_classes = self.dataset_handler.num_classes

        # Flatten the list of successfully processed indices
        flat_processed_original_indices = [idx for sublist in processed_image_indices_in_batch for idx in sublist]

        if len(all_probabilities_np) != len(flat_processed_original_indices):
            logger.error(
                f"Mismatch between number of probabilities ({len(all_probabilities_np)}) and tracked processed indices ({len(flat_processed_original_indices)}). This indicates an issue in tracking processed images.")
            # Fallback: iterate up to the minimum length to avoid index errors
            min_len = min(len(all_probabilities_np), len(flat_processed_original_indices))
        else:
            min_len = len(all_probabilities_np)

        for i in range(min_len):
            original_path_idx = flat_processed_original_indices[i]
            if original_path_idx >= len(valid_image_paths):  # Should not happen with correct tracking
                logger.error(
                    f"Tracked original index {original_path_idx} is out of bounds for valid_image_paths (len {len(valid_image_paths)}).")
                continue

            current_image_path = valid_image_paths[original_path_idx]
            probs_np = all_probabilities_np[i]

            predicted_idx = np.argmax(probs_np)
            predicted_name = class_names[predicted_idx] if class_names and 0 <= predicted_idx < len(
                class_names) else f"Class_{predicted_idx}"
            top_k_val = min(3, num_classes)
            top_k_indices = np.argsort(probs_np)[-top_k_val:][::-1]
            top_k_preds_list = []
            for k_idx in top_k_indices:
                k_name = class_names[k_idx] if class_names and 0 <= k_idx < len(class_names) else f"Class_{k_idx}"
                top_k_preds_list.append((k_name, float(probs_np[k_idx])))

            predictions_output.append({
                'image_path': str(current_image_path),
                'probabilities': probs_np.tolist(),
                'predicted_class_idx': int(predicted_idx),
                'predicted_class_name': predicted_name,
                'confidence': float(probs_np[predicted_idx]),
                'top_k_predictions': top_k_preds_list
            })

        logger.info(
            f"Successfully generated predictions for {len(predictions_output)} out of {len(valid_image_paths)} valid input images.")

        # --- Optionally Save Prediction Results to JSON ---
        results_detail_level = results_detail_level or self.results_detail_level

        saved_json_path = None
        if results_detail_level > 0:
            logger.info(f"Saving prediction results ({run_id}) with detail level {results_detail_level}...")
            results_for_json = {
                'method': 'predict_images',
                'run_id': run_id,
                'params': {
                    'num_original_input_paths': len(image_paths),  # Original number requested
                    'num_valid_input_paths': len(valid_image_paths),  # Number found to exist
                    'num_images_processed': len(predictions_output)  # Number actually predicted
                },
                'predictions': predictions_output
            }
            saved_json_path = self._save_results(
                results_data=results_for_json,
                method_name="predict_images",
                run_id=run_id,
                method_params=results_for_json['params'],
                results_detail_level=results_detail_level
            )
        else:
            logger.info(f"Skipping saving of prediction results JSON for {run_id} (detail level is 0).")

        # --- Plotting (Optional) ---
        current_plot_level = self.plot_level
        if plot_level is not None:
            current_plot_level = plot_level
            logger.debug(f"Plot level for this prediction run: {current_plot_level}")

        # Define max_cols for plot, potentially from an override
        # For now, using a fixed value or a default from plotter itself.
        # If you add 'prediction_plot_max_cols' to method signature:
        # plot_max_cols = self.prediction_plot_max_cols # default from pipeline init
        # if prediction_plot_max_cols is not None:
        #    plot_max_cols = prediction_plot_max_cols
        # else:
        plot_max_cols = 4  # Default from plotter will be used if not passed.

        if current_plot_level > 0 and predictions_output:
            plot_save_location: Optional[Path] = None  # Specific directory for these prediction plots
            if self.experiment_dir:  # If we have a base directory, save plots there
                pred_plot_dir_parent = self.experiment_dir / "predictions_plots"
                plot_save_location = pred_plot_dir_parent / run_id
                # The plotter's _create_plot_dir_from_base will handle mkdir if plot_save_location is passed
                # OR if plot_predictions directly takes plot_save_location as the final dir

            show_plots_flag = (current_plot_level == 2)

            if current_plot_level == 1 and not plot_save_location:
                logger.warning(
                    "Plot saving for predictions skipped: plot_level is 1 (save only) but no results_dir was specified for the pipeline.")
            else:
                logger.info(
                    f"Plotting prediction results (level {current_plot_level}). Save location base: {plot_save_location if plot_save_location else 'None (showing only)'}")
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_predictions(
                        predictions_output=predictions_output,  # Pass raw data
                        plot_save_dir=plot_save_location,  # Pass the specific directory for saving (can be None)
                        show_plots=show_plots_flag,
                        max_cols=4  # Or make this configurable
                    )
                except ImportError:
                    logger.error("Plotting skipped: ResultsPlotter class not found or plotting libraries missing.")
                except Exception as plot_err:
                    logger.error(f"Prediction plotting failed for {run_id}: {plot_err}", exc_info=True)
        elif current_plot_level > 0 and not predictions_output:
            logger.warning(f"Plotting skipped for predictions run {run_id}: No prediction outputs were generated.")

        return predictions_output

    def load_model(self, model_path: Union[str, Path]) -> None:
        """ Loads a state_dict into the pipeline_v1's model adapter. """
        # (Keep logic from code_v6, it should work with the initialized adapter)
        model_path = Path(model_path)
        logger.info(f"Loading model state_dict from: {model_path}")
        if not model_path.is_file(): raise FileNotFoundError(f"Model file not found at {model_path}")

        if not self.model_adapter.initialized_:
            logger.debug("Initializing skorch adapter before loading state_dict...")
            try: self.model_adapter.initialize()
            except Exception as e: raise RuntimeError("Could not initialize model adapter for loading.") from e

        if not hasattr(self.model_adapter, 'module_') or not isinstance(self.model_adapter.module_, nn.Module):
             raise RuntimeError("Adapter missing internal nn.Module ('module_'). Cannot load state_dict.")

        try:
            map_location = self.model_adapter.device
            state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
            logger.debug(f"State_dict loaded successfully to device '{map_location}'.")
            self.model_adapter.module_.load_state_dict(state_dict)
            self.model_adapter.module_.eval() # Set to eval mode
            logger.info("Model state_dict loaded successfully into the model adapter.")
        except Exception as e:
            logger.error(f"Failed to load state_dict into model: {e}", exc_info=True)
            if isinstance(e, RuntimeError) and "size mismatch" in str(e):
                 logger.error("Architecture mismatch likely. Ensure loaded weights match the current model config.")
            raise RuntimeError("Error loading state_dict into the model adapter module.") from e
