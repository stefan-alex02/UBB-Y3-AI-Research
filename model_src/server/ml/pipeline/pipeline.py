import contextlib
import io
import json
from datetime import datetime
from numbers import Number
from pathlib import Path, PurePath
from typing import Dict, List, Tuple, Callable, Any, Type, Optional, Union

import numpy as np
import pandas as pd
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
from ..plotter import ResultsPlotter
from ..skorch_utils import SkorchModelAdapter
from ..skorch_utils import get_default_callbacks
from ...persistence import LocalFileSystemRepository
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
                 lr_scheduler_policy_default: str = 'ReduceLROnPlateau',
                 **kwargs
                 ):
        """
        Initializes the Classification Pipeline for image classification tasks.

        This constructor sets up the pipeline with configuration for dataset handling,
        model architecture, training parameters, and artifact persistence. It prepares
        the model adapter with appropriate transforms and initializes the dataset handler.

        Args:
            dataset_path: Path to the root of the image dataset directory.
            model_type: Type of model architecture to use (CNN, ViT, etc.).
                       Can be a ModelType enum value or string.
            model_load_path: Optional path to pre-trained model weights to load at initialization.
                            If provided, the model is loaded immediately.
            img_size: Target size (height, width) for image resizing in preprocessing.
            artifact_repository: Repository for saving/loading models, results, and plots.
                                If None, persistence features will be limited.
            experiment_base_key_prefix: Base prefix for organizing artifacts in the repository.
                                       Used to group related experiment runs.
            results_detail_level: Controls verbosity of saved JSON results (0-3):
                                 0: No results saved
                                 1: Basic metrics only
                                 2: Detailed results with histories
                                 3: Full detail including batch-level data
            plot_level: Controls visualization behavior (0-2):
                       0: No plots generated
                       1: Plots saved but not displayed
                       2: Plots saved and displayed
            val_split_ratio: Proportion of training data to use for validation.
            test_split_ratio_if_flat: Proportion to reserve for testing when using flat dataset structure.
            augmentation_strategy: Data augmentation approach to use during training.
                                  Can be enum value, string name, or custom transform function.
            show_first_batch_augmentation_default: Whether to visualize the first batch of
                                                  augmented training data.
            use_offline_augmented_data: Whether to use pre-augmented images instead of
                                       real-time augmentation.
            force_flat_for_fixed_cv: If True, treats a structured dataset as flat for
                                    cross-validation purposes.
            optimizer: Optimizer to use for training (AdamW, Adam, SGD).
                      Can be a string name or optimizer class.
            lr: Learning rate for the optimizer.
            max_epochs: Maximum number of training epochs.
            batch_size: Batch size for training and inference.
            patience: Number of epochs with no improvement before early stopping.
            module__dropout_rate: Dropout rate to use in model architecture (if supported).
            lr_scheduler_policy_default: Learning rate scheduler policy name.
            **kwargs: Additional arguments passed to the model adapter.

        Raises:
            TypeError: If model_type is not a string or ModelType enum
            ValueError: If model_type string doesn't match any supported architecture
            RuntimeError: If dataset loading fails

        Note:
            The pipeline uses a SkorchModelAdapter internally to integrate PyTorch models with
            scikit-learn compatible training workflows. The dataset structure is automatically
            detected (flat or train/val/test splits) and handled appropriately.
        """
        self.dataset_path = Path(dataset_path).resolve()
        if isinstance(model_type, str):
            try:
                self.model_type = ModelType(model_type)
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

        logger.info(f"Initializing Classification Pipeline:")
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Model Type: {self.model_type.value}")

        if self.artifact_repo and self.experiment_run_key_prefix:
            logger.info(f"  Artifact base key prefix for this run: {self.experiment_run_key_prefix} (using {type(self.artifact_repo).__name__})")
        else:
            logger.info("  Artifact repository not configured or base prefix missing. File outputs might be limited.")

        logger.info(f"  Default Augmentation Strategy: {str(augmentation_strategy)}")
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
        elif self.artifact_repo and not self.experiment_run_key_prefix:
            logger.warning(
                "  Artifact repository configured, but no experiment_base_key_prefix provided from executor. Specific run outputs might not be grouped correctly.")
        else:
            logger.info("  Artifact repository not configured. File outputs will be disabled.")

        self.optimizer_type_config = optimizer
        self.lr_config = lr

        actual_optimizer_type: Type[torch.optim.Optimizer]
        if isinstance(optimizer, str):
            opt_lower = optimizer.lower()
            if opt_lower == "adamw":
                actual_optimizer_type = AdamW
            elif opt_lower == "adam":
                actual_optimizer_type = Adam
            elif opt_lower == "sgd":
                actual_optimizer_type = SGD
            else:
                raise ValueError(f"Unsupported optimizer string: '{optimizer}'. Choose from 'adamw', 'adam', 'sgd'.")
        elif issubclass(optimizer, torch.optim.Optimizer):
            actual_optimizer_type = optimizer
        else:
            raise TypeError(f"Optimizer must be a string or a torch.optim.Optimizer type, got {type(optimizer)}")

        logger.info(f"  Using Optimizer: {actual_optimizer_type.__name__}")

        model_class = self._get_model_class(self.model_type)

        self.patience_default = patience
        self.lr_scheduler_policy_default = lr_scheduler_policy_default
        default_callbacks = get_default_callbacks(
            early_stopping_patience=self.patience_default,
            lr_scheduler_policy=self.lr_scheduler_policy_default,
            patience=2,
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
            'use_offline_augmented_data': use_offline_augmented_data,
            'dataset_handler_ref': self.dataset_handler,
            'classes': np.arange(self.dataset_handler.num_classes),
            'verbose': 0,
            **module_params
        }

        self.model_adapter_config.update(kwargs)

        init_config_for_adapter = self.model_adapter_config.copy()
        init_config_for_adapter.pop('patience_cfg', None); init_config_for_adapter.pop('monitor_cfg', None)
        init_config_for_adapter.pop('lr_policy_cfg', None); init_config_for_adapter.pop('lr_patience_cfg', None)

        self.model_adapter = SkorchModelAdapter(**init_config_for_adapter)
        logger.info(f"  Model Adapter: Initialized with {model_class.__name__}")

        if model_load_path:
            self.load_model(model_load_path)

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

        return key_path.as_posix()

    @staticmethod
    def _get_model_class(model_type_enum: ModelType) -> Type[nn.Module]:
        """
        Retrieves the model class corresponding to the specified ModelType enum.
        :param model_type_enum: An instance of ModelType enum representing the desired model type.
        :return: The corresponding PyTorch model class.
        """
        model_class = model_type_enum.get_model_class()
        if model_class is None:
            raise ValueError(f"Unsupported model type: '{model_type_enum.value}'.")
        return model_class

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_score: Optional[np.ndarray] = None,
                         detailed: bool = False) -> Dict[str, Any]:
        """
        Calculates classification performance metrics from prediction results.

        This internal method computes a comprehensive set of evaluation metrics based on the
        ground truth labels and model predictions. It handles both binary and multi-class
        classification scenarios and can optionally calculate confidence-based metrics when
        probability scores are provided.

        Args:
            y_true: Ground truth labels as a numpy array.
            y_pred: Predicted class labels as a numpy array.
            y_score: Optional probability scores for each class, with shape (n_samples, n_classes).
                    Required for computing AUC and other probability-based metrics.
            detailed: Whether to include detailed data like raw predictions and curve points
                    in the results. When True, includes arrays of predictions and ROC/PR curve
                    coordinates that can be used for visualization.

        Returns:
            Dict containing organized metrics at multiple levels:
                - overall_accuracy: Accuracy across all classes
                - per_class: Dict mapping class names to individual metrics:
                    - precision: Precision for this class (TP / (TP + FP))
                    - recall: Recall/sensitivity for this class (TP / (TP + FN))
                    - specificity: Specificity for this class (TN / (TN + FP))
                    - f1: F1 score for this class
                    - roc_auc: ROC AUC score (requires y_score)
                    - pr_auc: Precision-Recall AUC score (requires y_score)
                - macro_avg: Dict with macro-averaged metrics across all classes
                - detailed_data: Optional detailed arrays (when detailed=True):
                    - y_true: Original ground truth labels
                    - y_pred: Model predictions
                    - y_score: Probability scores for each class (if provided)
                    - roc_curve_points: ROC curve coordinates per class
                    - pr_curve_points: PR curve coordinates per class

        Note:
            This method intelligently handles classes that do not appear in the test data
            by setting their metrics to NaN, allowing macro-averaging to work correctly.
            The ROC and PR metrics are only computed when y_score contains valid probabilities
            and the class has both positive and negative examples.
        """
        if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
        if y_score is not None and not isinstance(y_score, np.ndarray): y_score = np.array(y_score)

        metrics: Dict[str, Any] = {}
        class_metrics: Dict[str, Dict[str, float]] = {}
        macro_metrics: Dict[str, float] = {}
        detailed_data: Dict[str, Any] = {}

        all_class_names = self.dataset_handler.classes
        num_classes_total = self.dataset_handler.num_classes
        if not all_class_names:
            logger.warning("Cannot compute metrics: class names not available.")
            return {'error': 'Class names missing'}

        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)

        # Per-class metrics
        present_class_labels = np.unique(np.concatenate((y_true, y_pred)))
        all_precisions, all_recalls, all_specificities, all_f1s = [], [], [], []
        all_roc_aucs, all_pr_aucs = [], []
        all_roc_curves = {}
        all_pr_curves = {}

        can_compute_auc = y_score is not None and len(y_score.shape) == 2 and y_score.shape[
            1] == num_classes_total and len(y_score) == len(y_true)
        if y_score is not None and not can_compute_auc:
            logger.warning(f"y_score shape incompatible. Cannot compute AUCs.")

        for i, class_name in enumerate(all_class_names):
            class_label = self.dataset_handler.class_to_idx.get(class_name, i)
            is_present = class_label in np.unique(y_true)
            is_present_or_predicted = class_label in present_class_labels

            if not is_present_or_predicted:
                class_metrics[class_name] = {'precision': np.nan, 'recall': np.nan, 'specificity': np.nan,
                                             'f1': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan}
                all_precisions.append(np.nan)
                all_recalls.append(np.nan)
                all_specificities.append(np.nan)
                all_f1s.append(np.nan)
                all_roc_aucs.append(np.nan)
                all_pr_aucs.append(np.nan)
                if detailed:
                    all_roc_curves[class_name] = {'fpr': [], 'tpr': [], 'thresholds': []}
                    all_pr_curves[class_name] = {'precision': [], 'recall': [], 'thresholds': []}
                continue

            true_is_class = (y_true == class_label)
            pred_is_class = (y_pred == class_label)

            precision = precision_score(true_is_class, pred_is_class, zero_division=0)
            recall = recall_score(true_is_class, pred_is_class, zero_division=0)
            f1 = f1_score(true_is_class, pred_is_class, zero_division=0)
            specificity = recall_score(~true_is_class, ~pred_is_class, zero_division=0)

            roc_auc, pr_auc = np.nan, np.nan
            roc_curve_data = {'fpr': [], 'tpr': [], 'thresholds': []}
            pr_curve_data = {'precision': [], 'recall': [], 'thresholds': []}

            if can_compute_auc and is_present:
                score_for_class = y_score[:, class_label]
                if len(np.unique(true_is_class)) > 1:
                    try:
                        roc_auc = roc_auc_score(true_is_class, score_for_class)
                    except ValueError:
                        pass
                    except Exception as e:
                        logger.warning(f"ROC AUC Error (Class {class_name}): {e}")

                    try:
                        prec, rec, pr_thresh = precision_recall_curve(true_is_class, score_for_class)
                        order = np.argsort(rec)
                        pr_auc = auc(rec[order], prec[order])
                        if detailed:
                            pr_curve_data['precision'] = prec.tolist()
                            pr_curve_data['recall'] = rec.tolist()
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

            class_metrics[class_name] = {
                'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1,
                'roc_auc': roc_auc, 'pr_auc': pr_auc
            }
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_specificities.append(specificity)
            all_f1s.append(f1)
            all_roc_aucs.append(roc_auc)
            all_pr_aucs.append(pr_auc)
            if detailed:
                all_roc_curves[class_name] = roc_curve_data
                all_pr_curves[class_name] = pr_curve_data

        # Macro averages
        macro_metrics['precision'] = float(np.nanmean(all_precisions))
        macro_metrics['recall'] = float(np.nanmean(all_recalls))
        macro_metrics['specificity'] = float(np.nanmean(all_specificities))
        macro_metrics['f1'] = float(np.nanmean(all_f1s))
        macro_metrics['roc_auc'] = float(np.nanmean(all_roc_aucs)) if can_compute_auc else np.nan
        macro_metrics['pr_auc'] = float(np.nanmean(all_pr_aucs)) if can_compute_auc else np.nan

        metrics['per_class'] = class_metrics
        metrics['macro_avg'] = macro_metrics

        # Detailed data
        if detailed:
            detailed_data['y_true'] = y_true.tolist()
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
        """
        Saves method execution results to the artifact repository based on detail level.

        This internal method handles persisting results from pipeline operations like training,
        evaluation, and hyperparameter tuning. It controls which fields are included based on
        the results detail level and ensures proper organization in the artifact repository.

        Args:
            results_data: Dictionary containing all results to potentially save.
            method_name: Name of the pipeline method that generated these results.
            run_id: Unique identifier for this execution run.
            method_params: Optional dictionary of parameters used for this method execution.
                          Used for result metadata and filtering.
            results_detail_level: Controls verbosity of saved JSON results:
                                - 0: No JSON results saved
                                - 1: Basic summary metrics only
                                - 2: Detailed results including histories
                                - 3: Full detail with batch-level data
                                If None, uses the pipeline's default level.

        Returns:
            Path or identifier to the saved artifact, or None if saving was disabled or failed.

        Note:
            This is an internal method used by public pipeline methods to standardize result
            persistence. The artifact path structure follows the convention:
            {experiment_base_key_prefix}/{run_id}/{method_name}_results.json
        """
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

        if current_detail_level == 0:
            logger.info(f"Results detail level 0: Skipping JSON artifact saving for {run_id}.")
        else:
            logger.debug(f"Saving results for {run_id} (detail level: {current_detail_level})")
            results_to_save: Dict[str, Any] = {}

            level_1_keys = [
                'method', 'run_id', 'params',
                'overall_accuracy', 'macro_avg',
                'best_params', 'best_score',
                'best_epoch', 'best_valid_metric_value', 'valid_metric_name',
                'train_loss_at_best', 'train_acc_at_best', 'valid_acc_at_best',
                'best_refit_model_epoch_info',
                'mean_test_accuracy', 'std_test_accuracy',
                'mean_test_f1_macro', 'std_test_f1_macro',
                'aggregated_metrics',
                'outer_cv_scores',
                'cv_fold_scores',
                'evaluated_on', 'n_folds_requested', 'n_folds_processed', 'confidence_level',
                'saved_model_path',
                'message', 'error'
            ]
            for key in level_1_keys:
                if key in results_data:
                    results_to_save[key] = results_data[key]

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

            if current_detail_level >= 2:
                level_2_keys_additive = [
                    'per_class',
                    'detailed_data',
                    'full_params_used',
                    'method_params_used',
                    'params_used_for_folds',
                    'cv_results',
                    'fold_detailed_results',
                    'outer_fold_best_params_found',
                    'training_history',
                    'best_refit_model_history',
                    'outer_fold_best_model_histories',
                    'fold_training_histories',
                    'predictions'
                ]
                for key in level_2_keys_additive:
                    if key in results_data and key not in results_to_save:
                        results_to_save[key] = results_data[key]

            if current_detail_level < 3:
                history_keys_to_clean_batches_from = [
                    'training_history', 'best_refit_model_history',
                    'outer_fold_best_model_histories', 'fold_training_histories'
                ]
                for hist_key in history_keys_to_clean_batches_from:
                    if hist_key in results_to_save and isinstance(results_to_save[hist_key], list):
                        cleaned_history_list = []
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

            artifact_key = self._get_s3_object_key(run_id, f"{method_name}_results.json")

            if self.artifact_repo.save_json(results_to_save, artifact_key):
                saved_artifact_identifier = artifact_key  # For MinIO, this is the object key
                logger.info(f"Results JSON saved via repository to: {saved_artifact_identifier}")
            else:
                logger.error(f"Failed to save results JSON via repository for {run_id} to key {artifact_key}.")

        return saved_artifact_identifier

    def non_nested_grid_search(self,
                               param_grid: Union[Dict[str, list], List[Dict[str, list]]],
                               cv: int = 5,
                               val_split_ratio: Optional[float] = None,
                               n_iter: Optional[int] = None,
                               method: str = 'grid',
                               scoring: str = 'accuracy',
                               save_best_model: bool = True,
                               results_detail_level: Optional[int] = None,
                               plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs non-nested hyperparameter search and model selection using the train+validation data.

        This method runs either GridSearchCV or RandomizedSearchCV (based on the 'method' parameter)
        to find optimal hyperparameters. After finding the best configuration, it refits the model
        on the entire train+validation dataset and updates the pipeline's main model adapter.

        Note: Unlike nested_grid_search, this method may give optimistically biased performance
        estimates since the same data is used for both model selection and evaluation.

        Args:
            param_grid: Hyperparameter search space defined as either:
                - A dictionary mapping parameter names to lists of values to try
                - A list of such dictionaries for searching different parameter combinations
            cv: Number of cross-validation folds for the search
            val_split_ratio: Optional ratio for internal validation split within each CV fold's
                             training data. If None, uses the pipeline's default val_split_ratio.
            n_iter: Number of parameter settings sampled when method='random'.
                    Required for RandomizedSearchCV, ignored for GridSearchCV.
            method: Search method to use:
                    - 'grid': Exhaustive grid search (GridSearchCV)
                    - 'random': Random sampling (RandomizedSearchCV)
            scoring: Metric used for evaluation (string name of scikit-learn scoring metric
                    or callable)
            save_best_model: If True, saves the best model found to the artifact repository
            results_detail_level: Controls verbosity of saved JSON results:
                    - 0: No JSON results saved
                    - 1: Basic summary (metrics, best params)
                    - 2: Detailed (includes epoch histories)
                    - 3: Full detail with batch data
                    If None, uses the pipeline's default level.
            plot_level: Controls result plotting:
                    - 0: No plotting
                    - 1: Generate and save plots
                    - 2: Generate, save, and display plots
                    If None, uses the pipeline's default level.

        Returns:
            Dict containing search results, including:
                - method: Name of the method ('non_nested_grid_search' or 'non_nested_random_search')
                - run_id: Unique identifier for this run
                - params: Parameters used for this method
                - best_params: Best hyperparameters found during search
                - best_score: Best cross-validation score achieved
                - cv_results: Full results from GridSearchCV/RandomizedSearchCV
                - best_refit_model_history: Training history of the best model refit on full data
                - best_refit_model_epoch_info: Details about the best epoch of the refit model
                - saved_model_path: Path to the saved model (if save_best_model=True)

        Raises:
            ValueError: If n_iter is not provided when method='random'
            ValueError: If an unsupported search method is specified
            RuntimeError: If train+validation data is empty

        Note:
            This method does NOT evaluate on the test set. For an unbiased estimation
            of model performance, use nested_grid_search or separately evaluate using
            single_eval after this method.
        """
        method_lower = method.lower()
        run_id = f"non_nested_{method_lower}_{datetime.now().strftime('%H%M%S')}"
        search_type = "GridSearchCV" if method_lower == 'grid' else "RandomizedSearchCV"
        logger.info(f"Performing non-nested {search_type} with {cv}-fold CV.")
        logger.info(f"Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None: raise ValueError("n_iter required for random search.")
        if method_lower not in ['grid', 'random']: raise ValueError(f"Unsupported search method: {method}.")

        if isinstance(param_grid, dict):
            expanded_param_grid_for_search = expand_hyperparameter_grid(param_grid)
        elif isinstance(param_grid, list):
            expanded_param_grid_for_search = [expand_hyperparameter_grid(pg_dict) for pg_dict in param_grid]
        else:
            raise TypeError("param_grid must be a dictionary or list of dictionaries.")

        logger.info(
            f"Expanded Parameter Grid/Dist (for GridSearchCV):\n{json.dumps(expanded_param_grid_for_search, indent=2, default=str)}")

        X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels()
        if not X_trainval: raise RuntimeError("Train+validation data is empty.")
        logger.info(f"Using {len(X_trainval)} samples for Train+Validation in GridSearchCV.")

        # Validate internal validation split
        default_internal_val_fallback = 0.15
        val_frac_to_use = val_split_ratio if val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. Using default fallback: {default_internal_val_fallback:.3f}")
             val_frac_to_use = default_internal_val_fallback
        logger.info(f"Skorch internal validation split configured: {val_frac_to_use * 100:.1f}% of each CV fold's training data.")
        train_split_config = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED)

        # Setup Skorch Estimator
        adapter_config = self.model_adapter_config.copy()
        adapter_config['train_split'] = train_split_config
        adapter_config['verbose'] = 0

        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        estimator = SkorchModelAdapter(**adapter_config)

        # Setup search
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        SearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        search_kwargs: Dict[str, Any] = {
            'estimator': estimator, 'cv': cv_splitter, 'scoring': scoring,
            'n_jobs': 1, 'verbose': 3, 'refit': True,
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

        # Number of combinations
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
            logger.info(f"Target number of iterations (RandomizedSearchCV): {n_iter}")

        # Capture scikit-learn CV progress
        cv_progress_log = ""
        if search.verbose > 0:
            string_io_buffer = io.StringIO()
            with contextlib.redirect_stdout(string_io_buffer):
                try:
                    search.fit(X_trainval, y=np.array(y_trainval))
                except Exception as e:
                    cv_progress_log = string_io_buffer.getvalue()
                    logger.error(f"Error during {SearchClass.__name__}.fit: {e}", exc_info=True)
                    raise
            cv_progress_log = string_io_buffer.getvalue()
            string_io_buffer.close()
        else:
            search.fit(X_trainval, y=np.array(y_trainval))

        logger.info(f"Search completed.")

        if cv_progress_log.strip():
            logger.info("--- GridSearchCV Internal CV Progress (Captured) ---")
            for line in cv_progress_log.strip().splitlines():
                logger.info(f"[SKL_CV] {line}")
            logger.info("--- End GridSearchCV Internal CV Progress ---")

        # Collect results
        results = {
            'method': f"non_nested_{method_lower}_search",
            'run_id': run_id,
            'params': {'cv': cv, 'n_iter': n_iter if method_lower == 'random' else 'N/A', 'method': method_lower,
                       'scoring': scoring, 'val_split_ratio': val_frac_to_use},
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'test_set_evaluation': {'message': 'Test set evaluation not performed in this method.'},
            'accuracy': np.nan,
            'macro_avg': {}
        }

        # Store config of best estimator
        if hasattr(search, 'best_estimator_'):
             results['full_params_used'] = search.best_params_.copy()
             for k,v in self.model_adapter_config.items():
                  if not k.startswith(('optimizer__', 'module__', 'lr', 'batch_size', 'callbacks', 'train_transform', 'valid_transform')) and \
                     k not in results['full_params_used'] and isinstance(v, (str, int, float, bool, type(None))):
                        results['full_params_used'][k] = v
        else:
             results['full_params_used'] = {}


        # Update adapter
        best_estimator_refit = None
        if hasattr(search, 'best_estimator_'):
            best_estimator_refit = search.best_estimator_
            logger.info("Updating main pipeline_v1 adapter with the best model found and refit by GridSearchCV.")
            self.model_adapter = best_estimator_refit
            if not self.model_adapter.initialized_:
                 logger.warning("Refit best estimator seems not initialized, attempting initialize.")
                 try: self.model_adapter.initialize()
                 except Exception as init_err: logger.error(f"Failed to initialize refit estimator: {init_err}", exc_info=True)

            if hasattr(best_estimator_refit, 'history_') and best_estimator_refit.history_:
                results['best_refit_model_history'] = best_estimator_refit.history_.to_list()
                try:
                    refit_history = best_estimator_refit.history_
                    valid_loss_key_refit = 'valid_loss'
                    if refit_history and valid_loss_key_refit in refit_history[0]:
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
                            "Best refit model was trained without a validation split (e.g. val_split_ratio was 0 or invalid).")
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

        # Save Model
        model_path_identifier = None
        arch_config_path_identifier = None

        if save_best_model and best_estimator_refit is not None:
            if self.artifact_repo and self.experiment_run_key_prefix:
                try:
                    model_type_short = self.model_type.value
                    run_type_short = "gridcv"
                    run_id_timestamp_part = run_id.split('_')[-1]

                    best_cv_score = results.get('best_score', 0.0)
                    score_str = f"cvsc{best_cv_score:.2f}".replace('.', 'p').replace("0p", "p")

                    model_filename_base = f"{model_type_short}_{run_type_short}_{score_str}_{run_id_timestamp_part}"

                    model_pt_filename = f"{model_filename_base}.pt"
                    model_config_filename = f"{model_filename_base}_arch_config.json"

                    model_pt_object_key = self._get_s3_object_key(run_id, model_pt_filename)
                    model_config_object_key = self._get_s3_object_key(run_id, model_config_filename)

                    # Save Model State Dictionary
                    model_state_dict = best_estimator_refit.module_.state_dict()
                    model_path_identifier = self.artifact_repo.save_model_state_dict(
                        model_state_dict, model_pt_object_key
                    )
                    if model_path_identifier:
                        logger.info(f"Best refit model state_dict saved via repository to: {model_path_identifier}")
                        results['saved_model_path'] = model_path_identifier
                    else:
                        logger.error(
                            f"Failed to save best refit model state_dict for {run_id} to key {model_pt_object_key}.")
                        results['saved_model_path'] = None

                    # Save Architectural Configuration
                    if model_path_identifier:

                        arch_config_to_save = {
                            'model_type': self.model_type.value,
                            'num_classes': self.dataset_handler.num_classes,
                            'img_size_h': self.dataset_handler.img_size[0],
                            'img_size_w': self.dataset_handler.img_size[1],
                        }

                        best_estimator_params = best_estimator_refit.get_params(deep=False)

                        for key, value in best_estimator_params.items():
                            if key.startswith('module__'):
                                if isinstance(value, type):
                                    arch_config_to_save[
                                        key] = f"<class '{value.__module__}.{value.__name__}'>"
                                else:
                                    arch_config_to_save[key] = value

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
            results['saved_model_path'] = None
            results['saved_model_arch_config_path'] = None

        if not save_best_model:
            results['saved_model_path'] = None
            results['saved_model_arch_config_path'] = None

        # Save Results JSON
        summary_params = results.get('params', {}).copy()
        summary_params.update({f"best_{k}": v for k, v in results.get('best_params', {}).items() if isinstance(v, (str, int, float, bool))})
        json_artifact_key_or_path = self._save_results(
                            results_data=results,
                            method_name=f"non_nested_{method_lower}_search",
                            run_id=run_id,
                            method_params=summary_params,
                            results_detail_level=results_detail_level
                           )

        # Plot level
        current_plot_level = self.plot_level
        if plot_level is not None:
            current_plot_level = plot_level
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        # Plot results
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
                           val_split_ratio: Optional[float] = None,
                           n_iter: Optional[int] = None,
                           method: str = 'grid',
                           scoring: str = 'accuracy',
                           results_detail_level: Optional[int] = None,
                           plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs nested cross-validation for unbiased performance estimation and hyperparameter tuning.

        Nested CV uses an outer loop for performance estimation and an inner loop for hyperparameter
        optimization. This provides a less biased estimate of model generalization performance
        compared to non-nested methods.

        Args:
            param_grid: Hyperparameter search space defined as either:
                - A dictionary mapping parameter names to lists of values to try
                - A list of such dictionaries for searching multiple parameter subspaces
            outer_cv: Number of folds for the outer cross-validation loop (performance estimation)
            inner_cv: Number of folds for the inner cross-validation loop (hyperparameter tuning)
            val_split_ratio: Optional ratio for internal validation split within each inner fold's
                             training data. If None, uses the pipeline's default val_split_ratio.
            n_iter: Number of parameter settings sampled when method='random'.
                    Required for RandomizedSearchCV, ignored for GridSearchCV.
            method: Search method to use:
                    - 'grid': Exhaustive grid search (GridSearchCV)
                    - 'random': Random sampling (RandomizedSearchCV)
            scoring: Metric used for evaluation. Can be:
                    - A string specifying a scikit-learn scoring name (e.g., 'accuracy', 'f1')
                    - A callable that takes (estimator, X, y) and returns a scalar
            results_detail_level: Controls verbosity of saved JSON results:
                    - 0: No JSON results saved
                    - 1: Basic summary (metrics, best params)
                    - 2: Detailed (includes epoch histories)
                    - 3: Full detail with batch data
                    If None, uses the pipeline's default level.
            plot_level: Controls result plotting:
                    - 0: No plotting
                    - 1: Generate and save plots
                    - 2: Generate, save, and display plots
                    If None, uses the pipeline's default level.

        Returns:
            Dict containing nested CV results, including:
                - method: Name of the method ('nested_grid_search' or 'nested_random_search')
                - run_id: Unique identifier for this run
                - params: Parameters used for this method
                - outer_cv_scores: Scores for each outer fold
                - mean_test_accuracy, std_test_accuracy: Mean and standard deviation of test accuracy
                - mean_test_f1_macro, std_test_f1_macro: Mean and standard deviation of macro F1 score
                - outer_fold_best_params_found: Best parameters for each outer fold
                - aggregated_metrics: Overall performance metrics with confidence intervals
                - best_params: Most common best parameters across outer folds

        Raises:
            ValueError: If using RandomizedSearchCV without specifying n_iter
            ValueError: If using a FIXED dataset structure without setting force_flat_for_fixed_cv
            RuntimeError: If no data is available

        Note:
            This method operates on the full dataset, respecting the force_flat_for_fixed_cv setting.
            For FIXED dataset structures, force_flat_for_fixed_cv must be True.
        """
        method_lower = method.lower()
        run_id = f"nested_{method_lower}_{datetime.now().strftime('%H%M%S')}"
        search_type = "GridSearchCV" if method_lower == 'grid' else "RandomizedSearchCV"
        logger.info(f"Performing nested {search_type} search.")
        logger.info(f"  Outer CV folds: {outer_cv}, Inner CV folds: {inner_cv}")
        logger.info(f"  Parameter Grid/Dist for inner search:\n{json.dumps(param_grid, indent=2)}")
        logger.info(f"  Scoring Metric: {scoring}")

        # Expand the param_grid for the INNER search
        if isinstance(param_grid, dict):
            expanded_param_grid_for_inner_search = expand_hyperparameter_grid(param_grid)
        elif isinstance(param_grid, list):
            expanded_param_grid_for_inner_search = [expand_hyperparameter_grid(pg_dict) for pg_dict in param_grid]
        else:
            raise TypeError("param_grid for inner search must be a dictionary or list of dictionaries.")

        logger.debug(
            f"Expanded Parameter Grid/Dist for INNER search (for GridSearchCV):\n{json.dumps(expanded_param_grid_for_inner_search, indent=2, default=str)}")

        # Check Compatibility
        if self.dataset_handler.structure == DatasetStructure.FIXED and not self.force_flat_for_fixed_cv:
             raise ValueError(f"nested_grid_search requires a FLAT dataset structure "
                              f"or a FIXED structure with force_flat_for_fixed_cv=True.")

        # Standard Nested CV
        logger.info("Proceeding with standard nested CV using the full dataset.")
        try:
            X_full, y_full = self.dataset_handler.get_full_paths_labels_for_cv()
            if not X_full: raise RuntimeError("Full dataset for CV is empty.")
            logger.info(f"Using {len(X_full)} samples for outer cross-validation.")
            y_full_np = np.array(y_full)
        except Exception as e:
            logger.error(f"Failed to get full dataset paths/labels for nested CV: {e}", exc_info=True)
            raise

        default_internal_val_fallback = 0.15
        val_frac_to_use = val_split_ratio if val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. "
                            f"Using default fallback: {default_internal_val_fallback:.3f} for inner loop fits.")
             val_frac_to_use = default_internal_val_fallback

        logger.info(f"Inner loop Skorch validation split configured: {val_frac_to_use * 100:.1f}% of inner CV fold's training data.")
        train_split_config = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED)

        # Setup Inner Search Object
        adapter_config = self.model_adapter_config.copy()
        adapter_config['train_split'] = train_split_config
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

        # Setup Outer CV
        outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED + 1) # Different seed

        scoring_dict = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        }

        logger.info(
            f"Running standard nested CV using cross_validate. Inner GridSearchCV verbose: {inner_search.verbose}")
        try:
            cv_progress_log = ""
            if inner_search.verbose > 0:
                string_io_buffer = io.StringIO()
                with contextlib.redirect_stdout(string_io_buffer):
                    try:
                        cv_results = cross_validate(
                            inner_search, X_full, y_full_np, cv=outer_cv_splitter, scoring=scoring_dict,
                            return_estimator=True, n_jobs=1, error_score='raise'
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

            logger.info("Nested cross-validation finished.")

            if cv_progress_log.strip():
                logger.info("--- Nested CV Inner Loop Progress (Captured) ---")
                for line in cv_progress_log.strip().splitlines():
                    logger.info(f"[NESTED_SKL_CV] {line}")
                logger.info("--- End Nested CV Inner Loop Progress ---")

            # Process and Save Results
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
                    'val_split_ratio': val_frac_to_use
                },
                'outer_cv_scores': {k: v.tolist() for k, v in cv_results.items() if k.startswith('test_')},
                'mean_test_accuracy': float(np.mean(cv_results['test_accuracy'])),
                'std_test_accuracy': float(np.std(cv_results['test_accuracy'])),
                'mean_test_f1_macro': float(np.mean(cv_results['test_f1_macro'])),
                'std_test_f1_macro': float(np.std(cv_results['test_f1_macro'])),
            }

            # Extract histories and best params
            if 'estimator' in cv_results:
                fold_histories_nested = []
                best_params_per_fold_nested = []
                inner_score_per_fold_nested = []
                for fold_idx, outer_fold_search_estimator in enumerate(cv_results['estimator']):
                    if hasattr(outer_fold_search_estimator, 'best_estimator_') and \
                            hasattr(outer_fold_search_estimator.best_estimator_, 'history_') and \
                            outer_fold_search_estimator.best_estimator_.history_:
                        fold_histories_nested.append(outer_fold_search_estimator.best_estimator_.history_.to_list())
                    else:
                        fold_histories_nested.append(None)
                    if hasattr(outer_fold_search_estimator, 'best_params_'):
                        best_params_per_fold_nested.append(outer_fold_search_estimator.best_params_)
                    else:
                        best_params_per_fold_nested.append(None)
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

            # Save Results
            saved_json_path = self._save_results(results, f"nested_{method_lower}_search",
                                run_id=run_id,
                                method_params=results['params'],
                                results_detail_level=results_detail_level)

            current_plot_level = self.plot_level
            if plot_level is not None:
                current_plot_level = plot_level
                logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

            # Plot results
            if current_plot_level > 0:
                plot_save_dir_base_for_run: Optional[Union[str, Path]] = None

                if self.artifact_repo and self.experiment_run_key_prefix:
                    plot_save_dir_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())
                else:
                    plot_save_dir_base_for_run = None
                    logger.warning(f"No artifact repo specified for this run.")

                can_save_plots = (current_plot_level >= 1 and plot_save_dir_base_for_run is not None)
                should_show_plots_flag = (current_plot_level == 2)

                if not can_save_plots and current_plot_level == 1:
                    logger.warning(
                        f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no save location could be determined (e.g., no repository).")

                if can_save_plots or should_show_plots_flag:
                    logger.info(f"Plotting {method_name} results for {run_id} (plot level {current_plot_level}).")
                    try:
                        from ..plotter import ResultsPlotter
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
             return {
                'method': f"nested_{method_lower}_search",
                'params': {'outer_cv': outer_cv, 'inner_cv': inner_cv, 'n_iter': n_iter if method_lower=='random' else 'N/A', 'method': method_lower, 'scoring': scoring},
                'error': str(e)
             }

    def cv_model_evaluation(self,
                            cv: int = 5,
                            evaluate_on: str = 'full',
                            val_split_ratio: Optional[float] = None,
                            params: Optional[Dict] = None,
                            confidence_level: float = 0.95,
                            results_detail_level: Optional[int] = None,
                            plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs K-Fold cross-validation evaluation using fixed hyperparameters.

        This method evaluates model performance by training on K-1 folds and testing on the remaining fold,
        repeating for each fold. Unlike hyperparameter search methods, this uses fixed parameters for all
        folds and provides statistical confidence intervals for performance metrics.

        Args:
            cv: Number of folds for cross-validation.
            evaluate_on: Which data to use for evaluation:
                        - 'full': Use the entire dataset (or train+val in FIXED structure)
                        - 'test': Use only the test set (only for FIXED structure)
            val_split_ratio: Fraction for internal validation split during fold training.
                            If None, uses the pipeline's default val_split_ratio.
            params: Dictionary of fixed hyperparameters to use for training each fold.
                    If None, uses the current pipeline configuration.
            confidence_level: Confidence level for calculating confidence intervals
                             (e.g., 0.95 for 95% CI).
            results_detail_level: Controls verbosity of saved JSON results:
                                - 0: No JSON results saved
                                - 1: Basic summary (metrics, CI)
                                - 2: Detailed (includes epoch histories)
                                - 3: Full detail with all data
                                If None, uses the pipeline's default level.
            plot_level: Controls result plotting:
                       - 0: No plotting
                       - 1: Generate and save plots
                       - 2: Generate, save, and display plots
                       If None, uses the pipeline's default level.

        Returns:
            Dict containing evaluation results, including:
                - method: Name of the method ('cv_model_evaluation')
                - run_id: Unique identifier for this run
                - params: Parameters used for this method
                - evaluated_on: Data split used for evaluation ('full' or 'test')
                - n_folds_requested: Number of folds requested
                - n_folds_processed: Number of folds successfully processed
                - cv_fold_scores: List of performance metrics for each fold
                - fold_histories: Training histories for each fold (if detail level ≥2)
                - aggregated_metrics: Dictionary of metrics with means, std. devs, and confidence intervals
                - accuracy: Overall mean accuracy (for summary reporting)
                - macro_avg: Overall mean macro F1 score (for summary reporting)

        Raises:
            ValueError: If evaluate_on is not 'full' or 'test'
            ValueError: If confidence_level is not between 0 and 1
            ValueError: If using 'full' with FIXED dataset structure without force_flat_for_fixed_cv=True
            ValueError: If insufficient samples per class for the requested number of folds

        Note:
            This method works with both FLAT and FIXED dataset structures, though using 'full' with
            a FIXED structure requires force_flat_for_fixed_cv=True to be set in the pipeline.
        """
        logger.info(f"Performing {cv}-fold CV for evaluation with fixed parameters.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive).")

        valid_eval_on = ['full', 'test']
        if evaluate_on not in valid_eval_on:
            raise ValueError(f"Invalid 'evaluate_on' value: '{evaluate_on}'. Must be one of {valid_eval_on}")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1 (exclusive).")

        run_id = f"cv_model_evaluation_{evaluate_on}_{datetime.now().strftime('%H%M%S')}"

        # Get Data based on 'evaluate_on'
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
        else:
            raise ValueError(f"Internal error: Unknown evaluate_on='{evaluate_on}'")

        y_selected_np = np.array(y_selected_list)
        if len(np.unique(y_selected_np)) < 2:
            logger.warning(
                f"Only one class present in the selected data ({evaluate_on} set). Stratification might behave unexpectedly.")

        # Hyperparams
        eval_params = self.model_adapter_config.copy()
        if params:
            logger.info(f"Using provided parameters for CV evaluation: {params}")
            parsed_params_for_cv = parse_fixed_hyperparameters(
                params,
                default_max_epochs_for_cosine=eval_params.get('max_epochs')
            )
            eval_params.update(parsed_params_for_cv)

        eval_params['module'] = self._get_model_class(self.model_type)
        eval_params['module__num_classes'] = self.dataset_handler.num_classes
        eval_params['classes'] = np.arange(self.dataset_handler.num_classes)
        eval_params['train_transform'] = self.dataset_handler.get_train_transform()
        eval_params['valid_transform'] = self.dataset_handler.get_eval_transform()
        eval_params.setdefault('show_first_batch_augmentation', self.show_first_batch_augmentation_default)

        # Validate internal validation split
        default_internal_val_fallback = 0.15
        val_frac_to_use = val_split_ratio if val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
            logger.warning(
                f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. Using default fallback: {default_internal_val_fallback:.3f} for fold fits.")
            val_frac_to_use = default_internal_val_fallback
        logger.info(
            f"Skorch internal validation split configured: {val_frac_to_use * 100:.1f}% of each CV fold's training data.")

        # Setup CV Strategy
        min_samples_per_class = cv if cv > 1 else 1
        unique_labels, counts = np.unique(y_selected_np, return_counts=True)
        if K := min(counts) < min_samples_per_class:
            logger.warning(f"The selected data ({evaluate_on} set) has only {K} instances "
                           f"of class '{unique_labels[np.argmin(counts)]}', "
                           f"which is less than the number of folds ({cv}). "
                           f"StratifiedKFold may fail or produce unreliable splits.")

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []
        fold_histories = []
        fold_detailed_results = []

        compute_detailed_metrics_flag = False
        effective_detail_level_for_compute = results_detail_level or self.results_detail_level

        current_plot_level = self.plot_level
        if plot_level is not None:
            current_plot_level = plot_level
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        if (effective_detail_level_for_compute > 0 and current_plot_level > 0) or effective_detail_level_for_compute >= 2:
            compute_detailed_metrics_flag = True
            logger.debug(
                f"Will compute detailed metrics (detail_level={effective_detail_level_for_compute}, plot_level={current_plot_level})")

        # Manual Outer CV Loop
        for fold_idx, (outer_train_indices, outer_test_indices) in enumerate(
                cv_splitter.split(X_selected, y_selected_np)):
            logger.info(f"--- Starting CV Evaluation Fold {fold_idx + 1}/{cv} ---")

            X_outer_train = [X_selected[i] for i in outer_train_indices]
            y_outer_train = y_selected_np[outer_train_indices]
            X_fold_test = [X_selected[i] for i in outer_test_indices]
            y_fold_test = y_selected_np[outer_test_indices]
            logger.debug(
                f"Outer split ({evaluate_on} set): {len(X_outer_train)} train / {len(X_fold_test)} test samples.")

            if not X_outer_train or not X_fold_test:
                logger.warning(f"Fold {fold_idx + 1} resulted in empty train ({len(X_outer_train)}) or test ({len(X_fold_test)}) set. Skipping.")
                fold_results.append({k: np.nan for k in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                                                         'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']})
                fold_detailed_results.append({'error': 'Skipped fold due to empty data.'})
                continue

            # Setup Estimator
            fold_adapter_config = eval_params.copy()
            fold_adapter_config['train_split'] = ValidSplit(cv=val_frac_to_use, stratified=True,
                                                            random_state=RANDOM_SEED + fold_idx)
            fold_adapter_config['verbose'] = 0

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
                    fold_histories.append(estimator_fold.history)
                else:
                    fold_histories.append([])
            except Exception as fit_err:
                logger.error(f"Fit failed for fold {fold_idx + 1}: {fit_err}", exc_info=True)
                fold_results.append({k: np.nan for k in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro',
                                                         'specificity_macro', 'roc_auc_macro', 'pr_auc_macro']})
                fold_detailed_results.append({'error': f'Fit failed: {fit_err}'})
                continue

            logger.info(f"Using best model from epoch {estimator_fold.history[-1]['epoch']}")

            # Evaluate
            logger.info(f"Evaluating model on outer test set for fold {fold_idx + 1}...")
            try:
                y_pred_fold_test = estimator_fold.predict(X_fold_test)
                y_score_fold_test = estimator_fold.predict_proba(X_fold_test)
                fold_metrics = self._compute_metrics(y_fold_test, y_pred_fold_test, y_score_fold_test,
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

        # Aggregate results
        if not fold_results:
            logger.error("CV evaluation failed: No results from any fold.")
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
                                    k != 'error']
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
                        'margin_of_error': float(h) if not np.isnan(h) else None,
                        'ci_lower': float(mean_score - h) if not np.isnan(h) else None,
                        'ci_upper': float(mean_score + h) if not np.isnan(h) else None
                    }
                elif count == 1:
                    aggregated_metrics[metric_key] = {'mean': float(scores.iloc[0]), 'std_dev': 0.0}
                else:
                    aggregated_metrics[metric_key] = {'mean': np.nan, 'std_dev': np.nan}
        elif K == 1:
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

        results['accuracy'] = aggregated_metrics.get('accuracy', {}).get('mean', np.nan)
        results['macro_avg'] = {'f1': aggregated_metrics.get('f1_macro', {}).get('mean', np.nan)}

        # Save results
        summary_params = {k: v for k, v in eval_params.items() if isinstance(v, (str, int, float, bool))}
        summary_params['cv'] = cv
        summary_params['evaluated_on'] = evaluate_on
        summary_params['val_split_ratio'] = val_frac_to_use
        summary_params['confidence_level'] = confidence_level
        saved_json_path = self._save_results(results,
                                             "cv_model_evaluation",
                                             run_id=run_id,
                                             method_params=summary_params,
                                             results_detail_level=results_detail_level)

        # Plot results
        if current_plot_level > 0:
            plot_save_dir_base_for_run: Optional[Union[str, Path]] = None

            if self.artifact_repo and self.experiment_run_key_prefix:
                plot_save_dir_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())
            else:
                plot_save_dir_base_for_run = None
                logger.warning(f"No artifact repo specified for this run.")

            can_save_plots = (current_plot_level >= 1 and plot_save_dir_base_for_run is not None)
            should_show_plots_flag = (current_plot_level == 2)

            if not can_save_plots and current_plot_level == 1:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no save location could be determined (e.g., no repository).")

            if can_save_plots or should_show_plots_flag:
                logger.info(f"Plotting cv for eval results for {run_id} (plot level {current_plot_level}).")
                try:
                    from ..plotter import ResultsPlotter
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

        # Logging summary
        logger.info(f"CV Evaluation Summary (on {evaluate_on} data, {K} folds, {confidence_level * 100:.0f}% CI):")
        for metric_key, stats_dict in aggregated_metrics.items():
            mean_val = stats_dict.get('mean', np.nan)
            h_val = stats_dict.get('margin_of_error')
            if not np.isnan(mean_val):
                if h_val is not None and not np.isnan(h_val):
                    logger.info(f"  {metric_key.replace('_', ' ').title():<20}: {mean_val:.4f} +/- {h_val:.4f}")
                else:
                    logger.info(f"  {metric_key.replace('_', ' ').title():<20}: {mean_val:.4f} (CI not calculated)")
            else:
                logger.info(f"  {metric_key.replace('_', ' ').title():<20}: NaN")

        return results

    def single_train(self,
                     params: Optional[Dict[str, Any]] = None,
                     val_split_ratio: Optional[float] = None,
                     save_model: bool = True,
                     results_detail_level: Optional[int] = None,
                     plot_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs a single training run using a train/validation split.

        This method trains the model on the training data and evaluates on a validation set.
        It handles the data splitting process and configures Skorch with the appropriate
        validation strategy. The trained model becomes the pipeline's main model adapter.

        Args:
            params: Optional dictionary of hyperparameters to use for training.
                    If None, uses the current pipeline configuration.
            val_split_ratio: Fraction of training data to use for validation.
                            If None, uses the pipeline's default val_split_ratio.
            save_model: Whether to save the trained model to the artifact repository.
                       Set to False to skip model saving.
            results_detail_level: Controls verbosity of saved JSON results:
                                - 0: No JSON results saved
                                - 1: Basic summary (metrics)
                                - 2: Detailed (includes epoch histories)
                                - 3: Full detail with batch data
                                If None, uses the pipeline's default level.
            plot_level: Controls result plotting:
                       - 0: No plotting
                       - 1: Generate and save plots
                       - 2: Generate, save, and display plots
                       If None, uses the pipeline's default level.

        Returns:
            Dict containing training results, including:
                - method: Name of the method ('single_train')
                - run_id: Unique identifier for this run
                - params: Parameters used for this method
                - history: Training history data
                - val_metrics: Validation metrics if validation set was used
                - saved_model_path: Path to the saved model (if save_model=True)
                - accuracy: Validation accuracy (for summary reporting)
                - macro_avg: Validation macro F1 score (for summary reporting)

        Raises:
            RuntimeError: If train+validation data is empty
            ValueError: If validation split ratio is invalid

        Note:
            When save_model=True, the model is saved to the artifact repository with
            its configuration and evaluation metrics. The model becomes the pipeline's
            active model for further use (e.g., prediction, evaluation).
        """
        logger.info("Starting single training run...")
        run_id = f"single_train_{datetime.now().strftime('%H%M%S')}"

        # Train+Validation data
        X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels()
        if not X_trainval: raise RuntimeError("Train+validation data is empty.")
        y_trainval_np = np.array(y_trainval)

        # Validation split
        current_val_split_ratio = val_split_ratio if val_split_ratio is not None else self.dataset_handler.val_split_ratio
        train_split_config = None
        n_train, n_val = len(y_trainval_np), 0

        if not 0.0 < current_val_split_ratio < 1.0:
            logger.warning(f"Validation split ratio ({current_val_split_ratio}) is invalid or zero. "
                           f"Training on full {len(X_trainval)} trainval samples without validation set.")
            X_fit, y_fit = X_trainval, y_trainval_np
        elif len(np.unique(y_trainval_np)) < 2:
            logger.warning(
                f"Only one class present in trainval data. Cannot stratify split. Training on full {len(X_trainval)} samples without validation.")
            X_fit, y_fit = X_trainval, y_trainval_np
        else:
            try:
                train_indices, val_indices = train_test_split(
                    np.arange(len(X_trainval)), test_size=current_val_split_ratio,
                    stratify=y_trainval_np, random_state=RANDOM_SEED)
            except ValueError as e:
                logger.warning(f"Stratified train/val split failed ({e}). Using non-stratified split.")
                train_indices, val_indices = train_test_split(
                    np.arange(len(X_trainval)), test_size=current_val_split_ratio,
                    random_state=RANDOM_SEED)

            X_train_paths_list = [X_trainval[i] for i in train_indices]
            y_train_labels_np = y_trainval_np[train_indices]
            X_val_paths_list = [X_trainval[i] for i in val_indices]
            y_val_labels_np = y_trainval_np[val_indices]
            n_train, n_val = len(y_train_labels_np), len(y_val_labels_np)

            X_fit = X_train_paths_list + X_val_paths_list
            y_fit = np.concatenate((y_train_labels_np, y_val_labels_np))

            test_fold = np.full(len(X_fit), -1, dtype=int)
            test_fold[n_train:] = 0
            ps = PredefinedSplit(test_fold=test_fold)
            train_split_config = ValidSplit(cv=ps, stratified=False) # ps defines the split

        logger.info(f"Using split: {n_train} train / {n_val} validation samples.")

        adapter_config = self.model_adapter_config.copy()

        if params:
            logger.info(f"Applying custom parameters for this single_train run: {params}")
            parsed_params = parse_fixed_hyperparameters(
                params,
                default_max_epochs_for_cosine=adapter_config.get('max_epochs')
            )
            adapter_config.update(parsed_params)

        adapter_config['train_split'] = train_split_config

        # Handle callbacks
        final_callbacks = adapter_config.get('callbacks', [])
        if isinstance(final_callbacks, list):
             if 'callbacks__default_lr_scheduler' in adapter_config and isinstance(adapter_config['callbacks__default_lr_scheduler'], LRScheduler):
                 new_lr_scheduler_instance = adapter_config.pop('callbacks__default_lr_scheduler')
                 found_lr_scheduler = False
                 for i, (name, cb) in enumerate(final_callbacks):
                     if name == DEFAULT_LR_SCHEDULER_NAME:
                         final_callbacks[i] = (name, new_lr_scheduler_instance)
                         found_lr_scheduler = True
                         break
                 if not found_lr_scheduler:
                     final_callbacks.append((DEFAULT_LR_SCHEDULER_NAME, new_lr_scheduler_instance))
                 adapter_config['callbacks'] = final_callbacks

        if train_split_config is None:
            logger.warning(
                "No validation set. Callbacks monitoring validation metrics (EarlyStopping, LRScheduler) may be removed or ineffective.")
            adapter_config['callbacks'] = [
                (name, cb) for name, cb in final_callbacks
                if not isinstance(cb, (EarlyStopping, LRScheduler))
            ]
        else:
            adapter_config['callbacks'] = final_callbacks

        adapter_config['verbose'] = 0

        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        # Instantiate the adapter
        adapter_for_train = SkorchModelAdapter(**adapter_config)

        # Train model
        logger.info(f"Fitting model (run_id: {run_id})...")
        adapter_for_train.fit(X_fit, y=y_fit)

        # Collect results
        history = adapter_for_train.history
        results: Dict[str, Any] = {'method': 'single_train',
                   'run_id': run_id,
                   'full_params_used': adapter_config.copy()}

        best_epoch_info = {}
        valid_loss_key = 'valid_loss'
        validation_was_run = train_split_config is not None and history and valid_loss_key in history[-1]

        if validation_was_run:
            try:
                scores = [epoch.get(valid_loss_key, np.inf) for epoch in history]
                best_idx = np.argmin(scores)
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
                    validation_was_run = False
            except Exception as e:
                logger.error(f"Error processing history for best epoch: {e}", exc_info=True)
                validation_was_run = False

        if not validation_was_run:
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
            if train_split_config is not None:
                logger.warning(
                    f"Error finding best epoch based on validation. Reporting last epoch ({last_epoch_num}) stats.")

        results.update(best_epoch_info)
        results['training_history'] = history.to_list() if history else []

        # Save model
        model_path_identifier = None
        if save_model:
            model_type_short = self.model_type.value
            run_type_short = "sngl"
            run_id_timestamp_part = run_id.split('_')[-1]

            val_metric_val = results.get('best_valid_metric_value', np.nan)
            metric_name_short = "val_loss"
            if not np.isnan(val_metric_val):
                metric_str = f"{metric_name_short}{val_metric_val:.2f}".replace('.', 'p').replace("0p", "p")
            else:
                metric_str = "no_val"

            epoch_num = results.get('best_epoch', 0)
            model_filename_base = f"{model_type_short}_{run_type_short}_ep{epoch_num}_{metric_str}_{run_id_timestamp_part}"

            model_pt_filename = f"{model_filename_base}.pt"
            model_config_filename = f"{model_filename_base}_arch_config.json"

            model_pt_object_key = self._get_s3_object_key(run_id, model_pt_filename)
            model_config_object_key = self._get_s3_object_key(run_id, model_config_filename)

            state_dict = adapter_for_train.module_.state_dict()
            model_path_identifier = self.artifact_repo.save_model_state_dict(state_dict, model_pt_object_key)

            # Save config
            if model_path_identifier:
                effective_adapter_config = adapter_for_train.get_params(deep=False)

                arch_config = {
                    'model_type': self.model_type.value,
                    'num_classes': self.dataset_handler.num_classes,
                    'img_size_h': self.dataset_handler.img_size[0],
                    'img_size_w': self.dataset_handler.img_size[1],
                }
                for key, value in effective_adapter_config.items():
                    if key.startswith('module__'):
                        arch_config[key] = value

                arch_config_path_identifier = self.artifact_repo.save_json(arch_config, model_config_object_key)
                logger.info(f"Model architectural config saved to: {model_config_object_key}")
                results['saved_model_arch_config_path'] = arch_config_path_identifier
        results['saved_model_path'] = model_path_identifier

        self.model_adapter = adapter_for_train
        logger.info(f"Main pipeline model adapter updated from single_train run: {run_id}")

        results['accuracy'] = results.get('valid_acc_at_best', np.nan)
        results['macro_avg'] = {}

        # Save results
        simple_params = {k: v for k, v in adapter_config.items() if isinstance(v, (str, int, float, bool))}
        simple_params['val_split_ratio_used'] = current_val_split_ratio if train_split_config else 0.0

        json_artifact_key_or_path = self._save_results(
            results_data=results,
            method_name="single_train",
            run_id=run_id,
            method_params=simple_params,
            results_detail_level=results_detail_level
        )

        current_plot_level = self.plot_level
        if plot_level is not None: current_plot_level = plot_level

        # Plot results
        if current_plot_level > 0:
            plot_save_location_base: Optional[str] = None
            if self.artifact_repo and self.experiment_run_key_prefix:
                plot_save_location_base = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())

            if current_plot_level == 1 and not plot_save_location_base:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 but no repository/base_key configured for saving.")
            else:
                logger.info(f"Plotting single_train results for {run_id} (plot level {current_plot_level}).")
                show_plots_flag = (current_plot_level == 2)
                try:
                    from ..plotter import ResultsPlotter
                    ResultsPlotter.plot_single_train_results(
                        results_input=results,
                        plot_save_dir_base=plot_save_location_base,
                        repository_for_plots=self.artifact_repo if plot_save_location_base else None,
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
        """
        Evaluates the current model adapter on the test set.

        This method performs inference using the currently loaded model on the test data
        and calculates performance metrics. It does not modify the model or perform any training.

        Args:
            results_detail_level: Controls verbosity of saved JSON results:
                                - 0: No JSON results saved
                                - 1: Basic summary (metrics)
                                - 2: Detailed (includes predictions)
                                - 3: Full detail with confidence scores
                                If None, uses the pipeline's default level.
            plot_level: Controls result plotting:
                       - 0: No plotting
                       - 1: Generate and save plots
                       - 2: Generate, save, and display plots
                       If None, uses the pipeline's default level.

        Returns:
            Dict containing evaluation results, including:
                - method: Name of the method ('single_eval')
                - run_id: Unique identifier for this run
                - metrics: Computed performance metrics
                - accuracy: Test accuracy (for summary reporting)
                - macro_avg: Test macro F1 score (for summary reporting)
                - confusion_matrix: Confusion matrix as a nested list
                - detailed_results: Detailed prediction results (if detail level ≥2)

        Raises:
            RuntimeError: If the model adapter is not initialized (needs training or loading first)

        Note:
            This method is typically used after training a model with single_train or
            non_nested_grid_search to evaluate its performance on unseen test data.
        """
        logger.info("Starting model evaluation on the test set...")
        run_id = f"single_eval_{datetime.now().strftime('%H%M%S')}"

        if not self.model_adapter.initialized_:
             raise RuntimeError("Model adapter not initialized. Train or load first.")

        # Test data
        X_test, y_test = self.dataset_handler.get_test_paths_labels()
        if not X_test:
             logger.warning("Test set is empty. Skipping evaluation.")
             return {'method': 'single_eval', 'message': 'Test set empty, evaluation skipped.'}
        y_test_np = np.array(y_test)

        # Make predictions
        logger.info(f"Evaluating on {len(X_test)} test samples...")
        try:
             y_pred_test = self.model_adapter.predict(X_test)
             y_score_test = self.model_adapter.predict_proba(X_test)
        except Exception as e:
             logger.error(f"Prediction failed during single_eval: {e}", exc_info=True)
             raise RuntimeError("Failed to get predictions from model adapter.") from e

        # Compute metrics
        compute_detailed_metrics_flag = False
        effective_detail_level_for_compute = results_detail_level or self.results_detail_level

        current_plot_level = self.plot_level
        if plot_level is not None:
            current_plot_level = plot_level
            logger.debug(f"Plot level overridden for this run to: {current_plot_level}")

        if (effective_detail_level_for_compute > 0 and current_plot_level > 0) or effective_detail_level_for_compute >= 2:
            compute_detailed_metrics_flag = True

        metrics = self._compute_metrics(y_test_np, y_pred_test, y_score_test, detailed=compute_detailed_metrics_flag)
        results = {
            'method': 'single_eval',
            'params': {},
            'run_id': run_id,
            **metrics}
        method_name = results['method']

        saved_json_path = self._save_results(results, "single_eval",
                           method_params=results['params'],
                           run_id=run_id,
                           results_detail_level=results_detail_level)

        # Plot results
        if current_plot_level > 0:
            plot_save_dir_base_for_run: Optional[Union[str, Path]] = None

            if self.artifact_repo and self.experiment_run_key_prefix:
                plot_save_dir_base_for_run = str((PurePath(self.experiment_run_key_prefix) / run_id).as_posix())
            else:
                plot_save_dir_base_for_run = None
                logger.warning(f"No artifact repo specified for this run.")
            can_save_plots = (current_plot_level >= 1 and plot_save_dir_base_for_run is not None)
            should_show_plots_flag = (current_plot_level == 2)

            if not can_save_plots and current_plot_level == 1:
                logger.warning(
                    f"Plot saving to file skipped for {run_id}: plot_level is 1 (save only) but no save location could be determined (e.g., no repository).")

            if can_save_plots or should_show_plots_flag:
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
                       image_tasks_for_pipeline: List[Any],
                       experiment_run_id_of_model: str,
                       username: str = "anonymous",
                       persist_prediction_artifacts: bool = True,
                       results_detail_level: Optional[int] = None,
                       plot_level: int = 0,
                       generate_lime_explanations: bool = False,
                       lime_num_features_to_show_plot: int = 5,
                       lime_num_samples_for_explainer: int = 1000,
                       prob_plot_top_k: int = -1
                       ) -> List[Dict[str, Any]]:
        """
        Performs inference on a list of images using the current model adapter.

        This method processes each image through the model, generates predictions with confidence
        scores, and optionally creates visualizations including LIME explanations. Results can
        be persisted to the artifact repository.

        Args:
            image_tasks_for_pipeline: List of image task objects to predict on. Each task should
                                     contain image data or paths that can be processed by the model.
            experiment_run_id_of_model: Identifier of the experiment run that produced the model.
                                       Used for organization and traceability.
            username: Name of the user requesting prediction, for tracking purposes.
            persist_prediction_artifacts: Whether to save prediction results and visualizations
                                         to the artifact repository.
            results_detail_level: Controls verbosity of saved JSON results:
                                - 0: No JSON results saved
                                - 1: Basic prediction results
                                - 2: Detailed with confidence scores
                                - 3: Full detail with intermediate activations
                                If None, uses the pipeline's default level.
            plot_level: Controls result plotting:
                       - 0: No plotting
                       - 1: Generate and save plots
                       - 2: Generate, save, and display plots
            generate_lime_explanations: Whether to generate LIME explanations for predictions.
                                       Requires LIME to be installed.
            lime_num_features_to_show_plot: Number of top features to show in LIME explanation plots.
            lime_num_samples_for_explainer: Number of samples to use for LIME explainer.
            prob_plot_top_k: Number of top classes to show in probability plots. If -1, shows all classes.

        Returns:
            List of dictionaries, one per image, containing:
                - image_id: Identifier for the image
                - predicted_class: Name of the predicted class
                - confidence: Confidence score for the prediction
                - all_probs: Probabilities for all classes
                - explanation_path: Path to LIME explanation (if generated)
                - artifact_paths: Paths to saved artifacts (if persisted)

        Raises:
            RuntimeError: If the model adapter is not initialized
            ValueError: If image_tasks_for_pipeline is empty
            ImportError: If generate_lime_explanations=True but LIME is not available

        Note:
            LIME explanations provide visual insight into which parts of the image most influenced
            the prediction. These are computationally expensive and increase prediction time
            significantly when enabled.
        """
        predict_op_run_id = f"predict_op_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.info(f"Op {predict_op_run_id}: Starting prediction for {len(image_tasks_for_pipeline)} images "
                    f"by user '{username}', using model from experiment '{experiment_run_id_of_model}'.")

        if not self.model_adapter.initialized_: logger.error(
            f"Op {predict_op_run_id}: Model adapter not initialized."); raise RuntimeError(
            "Model adapter not initialized.")
        if not image_tasks_for_pipeline: logger.warning(f"Op {predict_op_run_id}: No image_tasks_for_pipeline."); return []
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
                    pil_img_lime = Image.fromarray(img_np_lime)
                    transformed_img_lime = self.dataset_handler.get_eval_transform()(pil_img_lime)
                    processed_images_lime.append(transformed_img_lime)
                if not processed_images_lime: return np.array([])
                batch_tensor_lime = torch.stack(processed_images_lime).to(self.model_adapter.device)
                self.model_adapter.module_.eval()
                with torch.no_grad():
                    logits_lime = self.model_adapter.module_(batch_tensor_lime); probs_lime = torch.softmax(logits_lime,
                                                                                                            dim=1)
                return probs_lime.cpu().numpy()
        elif generate_lime_explanations:
            logger.warning(f"Op {predict_op_run_id}: LIME requested but not available.")
        pil_images_for_processing: List[Tuple[Optional[Image.Image], Any]] = []
        valid_tasks: List[Any] = []

        for task in image_tasks_for_pipeline:
            pil_image: Optional[Image.Image] = None
            img_filename = f"{task.image_id}.{task.image_format.lower().replace('.', '')}"
            image_key_or_path: str | Path
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
            if pil_image:
                pil_images_for_processing.append((pil_image, task))
                valid_tasks.append(task)
            else:
                logger.warning(
                    f"Could not load image for task: image_id={task.image_id}, prediction_id={task.prediction_id}")
        valid_pil_images = [img for img, _ in pil_images_for_processing if img is not None]
        valid_image_ids = [image_id for img, (image_id, _, _) in pil_images_for_processing if img is not None]
        if not valid_pil_images: logger.error(f"Op {predict_op_run_id}: No valid images loaded."); return []
        logger.info(f"Op {predict_op_run_id}: Loaded {len(valid_pil_images)} images.")

        class InMemoryPILDataset(torch.utils.data.Dataset):
            def __init__(self, pil_images: List[Image.Image], identifiers: List[Union[int, str]], transform: Callable):
                self.pil_images = pil_images; self.identifiers = identifiers; self.transform = transform

            def __len__(self):
                return len(self.pil_images)

            def __getitem__(self, idx):
                img = self.pil_images[idx]
                identifier = self.identifiers[idx]
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
        effective_results_detail_level = self.results_detail_level if results_detail_level is None else results_detail_level

        for i, task_object in enumerate(valid_tasks):
            if i >= len(all_probabilities_np): continue

            image_id = task_object.image_id
            prediction_id = task_object.prediction_id

            probs_np = all_probabilities_np[i]
            predicted_idx = int(np.argmax(probs_np))
            predicted_name = self.dataset_handler.classes[predicted_idx]
            confidence = float(probs_np[predicted_idx])
            top_k_val = min(prob_plot_top_k if prob_plot_top_k > 0 else self.dataset_handler.num_classes,
                            self.dataset_handler.num_classes)
            top_k_indices = np.argsort(probs_np)[-top_k_val:][::-1]
            top_k_preds_list = [(self.dataset_handler.classes[k_idx], float(probs_np[k_idx])) for k_idx in
                                top_k_indices]

            lime_data_for_plotter = None

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

                    lime_data_for_plotter = {
                        'explained_class_idx': predicted_idx,
                        'explained_class_name': predicted_name,
                        'feature_weights': lime_weights,
                        'segments_for_render': explanation.segments.tolist(),
                        'num_features_from_lime_run': lime_num_features_to_show_plot
                    }

                    lime_data_for_json_file = {
                        'explained_class_idx': predicted_idx,
                        'explained_class_name': predicted_name,
                        'feature_weights': lime_weights,
                        'num_features_from_lime_run': lime_num_features_to_show_plot
                    }

                except Exception as lime_e:
                    logger.error(f"Op {predict_op_run_id}: LIME failed for {image_id}: {lime_e}", exc_info=False)
                    lime_data_for_json_file = {'error': str(lime_e)}
                    lime_data_for_plotter = lime_data_for_json_file

            single_prediction_json_content = {
                "image_id": image_id,
                "prediction_id": prediction_id,
                "experiment_run_id_of_model": experiment_run_id_of_model,
                "image_user_source_path": f"images/{username}/{image_id}.{task_object.image_format.lower().replace('.', '')}",
                "probabilities": probs_np.tolist(),
                "predicted_class_idx": predicted_idx,
                "predicted_class_name": predicted_name,
                "confidence": confidence,
                "top_k_predictions_for_plot": top_k_preds_list,
                "lime_explanation": lime_data_for_json_file
            }

            predictions_to_return_for_api.append({
                "image_id": image_id,
                "prediction_id": prediction_id,
                "experiment_id": experiment_run_id_of_model,
                "predicted_class": predicted_name,
                "confidence": confidence,
            })

            prediction_artifact_base_path = PurePath("predictions") / username / str(image_id) / str(prediction_id)

            # Save JSON
            if persist_prediction_artifacts and self.artifact_repo and effective_results_detail_level > 0:
                pred_json_key = str((prediction_artifact_base_path / "prediction_details.json").as_posix())
                self.artifact_repo.save_json(single_prediction_json_content, pred_json_key)
                logger.info(f"Op {predict_op_run_id}: Prediction JSON for image {image_id} saved to: {pred_json_key}")

            # Generate LIME plot
            if generate_lime_explanations and lime_data_for_plotter and 'error' not in lime_data_for_plotter:
                if persist_prediction_artifacts and self.artifact_repo and plot_level > 0:
                    lime_plot_key = str((prediction_artifact_base_path / "plots" / "lime_explanation.png").as_posix())
                    ResultsPlotter.plot_lime_explanation_image(
                        original_pil_image=valid_pil_images[i],
                        lime_explanation_data=lime_data_for_plotter,
                        lime_num_features_to_display=lime_num_features_to_show_plot,
                        output_path=lime_plot_key,
                        repository_for_plots=self.artifact_repo,
                        show_plots=(plot_level == 2),
                        image_identifier=str(image_id)
                    )
            elif generate_lime_explanations and (not lime_data_for_plotter or 'error' in lime_data_for_plotter):
                logger.warning(
                    f"Op {predict_op_run_id}: LIME plot generation skipped for image {image_id} due to missing LIME data or LIME error.")

            # Probability plot
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
        Loads a trained model from the artifact repository into the pipeline's model adapter.

        This method loads both the model state dictionary and its architectural configuration
        from the specified path or object key. It supports loading from both local filesystem
        and remote storage (S3/MinIO). After loading, the model becomes the pipeline's active
        model for inference and evaluation tasks.

        Args:
            model_path_or_key: Path or key to the saved model file (.pt):
                              - For S3/MinIO: Relative path within the experiment prefix
                                (e.g., "run_id/model.pt")
                              - For local filesystem: Absolute or relative path to the model file

        Returns:
            None: Updates the pipeline's model adapter in-place

        Raises:
            FileNotFoundError: If the model file or its configuration cannot be found
            RuntimeError: If the model state dictionary cannot be loaded
            ValueError: If the loaded model architecture doesn't match the pipeline's configuration

        Note:
            The method automatically looks for a corresponding architecture configuration file
            with the naming pattern "{model_name}_arch_config.json" in the same directory.
            This config is used to ensure the correct model architecture is instantiated before
            loading the state dictionary.
        """
        run_id_for_log = f"load_model_op_{datetime.now().strftime('%H%M%S')}"
        logger.info(f"Operation {run_id_for_log}: Attempting to load model from: {model_path_or_key}")

        base_model_path_str = str(model_path_or_key)

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

        # Load config
        arch_config_dict: Optional[Dict[str, Any]] = None
        if self.artifact_repo:
            arch_config_dict = self.artifact_repo.load_json(full_arch_config_key_or_path)

        if arch_config_dict is None and Path(full_arch_config_key_or_path).is_file():
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
            if not self.model_adapter.initialized_:
                try:
                    self.model_adapter.initialize()
                except Exception as e:
                    raise RuntimeError(f"Op {run_id_for_log}: Default init failed (arch_config missing): {e}") from e
        else:
            logger.info(f"Op {run_id_for_log}: Loaded architecture config from: {full_arch_config_key_or_path}")
            loaded_model_type_str = arch_config_dict.pop('model_type')
            loaded_num_classes = arch_config_dict.pop('num_classes')
            loaded_img_size_h = arch_config_dict.pop('img_size_h', None)
            loaded_img_size_w = arch_config_dict.pop('img_size_w', None)

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

            for k_pop in ['patience_cfg', 'monitor_cfg', 'lr_policy_cfg', 'lr_patience_cfg']:
                current_pipeline_defaults.pop(k_pop, None)
            current_pipeline_defaults['dataset_handler_ref'] = self.dataset_handler
            current_pipeline_defaults['train_transform'] = self.dataset_handler.get_train_transform()
            current_pipeline_defaults['valid_transform'] = self.dataset_handler.get_eval_transform()
            current_pipeline_defaults['classes'] = np.arange(loaded_num_classes)

            if loaded_img_size_h is not None and loaded_img_size_w is not None:
                new_img_size = (loaded_img_size_h, loaded_img_size_w)
                if new_img_size != self.dataset_handler.img_size:
                    logger.info(f"Reconfiguring ImageDatasetHandler for loaded model's image size: {new_img_size}")
                    self.dataset_handler.img_size = new_img_size
                    current_pipeline_defaults[
                        'train_transform'] = self.dataset_handler.get_train_transform()
                    current_pipeline_defaults[
                        'valid_transform'] = self.dataset_handler.get_eval_transform()
                    self.model_adapter_config['train_transform'] = current_pipeline_defaults['train_transform']
                    self.model_adapter_config['valid_transform'] = current_pipeline_defaults['valid_transform']
                    self.model_adapter_config['img_size'] = new_img_size
                else:
                    logger.info(f"Loaded model's image size {new_img_size} matches current pipeline config.")
            else:
                logger.warning(
                    "Image size not found in loaded arch_config. Using current pipeline default image size for transforms.")

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

        # Load state dict
        state_dict: Optional[Dict] = None
        map_location = self.model_adapter.device

        if self.artifact_repo:
            state_dict = self.artifact_repo.load_model_state_dict(str(full_model_pt_key_or_path),
                                                                  map_location=map_location)
            if state_dict: logger.info(
                f"Op {run_id_for_log}: Model state_dict loaded via repo from: {full_model_pt_key_or_path}")

        if state_dict is None:
            local_pt_path = Path(full_model_pt_key_or_path)
            if not self.artifact_repo and not local_pt_path.is_file():
                local_pt_path = Path(model_path_or_key)

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
                is_skorch_module_prefixed = all(k.startswith("module.") for k in state_dict.keys())
                if is_skorch_module_prefixed and hasattr(self.model_adapter, 'module_') and isinstance(
                        self.model_adapter.module_, nn.Module):
                    logger.debug("Op {run_id_for_log}: Stripping 'module.' prefix from state_dict keys.")
                    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

                self.model_adapter.module_.load_state_dict(state_dict)
                self.model_adapter.module_.eval()
                logger.info(f"Op {run_id_for_log}: Model state_dict applied successfully from: {model_path_or_key}. Model image size context: {self.dataset_handler.img_size}")
            except Exception as e_apply:
                logger.error(f"Op {run_id_for_log}: Failed to apply state_dict: {e_apply}", exc_info=True)
                raise RuntimeError(f"Error applying state_dict from '{model_path_or_key}'.") from e_apply
        else:
            raise FileNotFoundError(
                f"Op {run_id_for_log}: Model state_dict could not be loaded from: {model_path_or_key}.")
