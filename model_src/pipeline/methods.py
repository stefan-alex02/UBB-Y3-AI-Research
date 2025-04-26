import os
import json
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import torch
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, cross_val_predict
)
from sklearn.metrics import make_scorer, accuracy_score # Used for CV scoring
from skorch.callbacks import Checkpoint # For saving best model during search

from utils import logger, set_seed
from datasets import DatasetHandler
from adapters import SkorchImageClassifier
from evaluation import calculate_metrics, format_metrics_log


def _get_output_dir(base_dir: str, dataset_name: str, model_name: str, method_config: Dict[str, Any]) -> str:
    """Creates and returns a unique directory for method results."""
    # Create a hash of the config for unique folder name (optional)
    config_str = json.dumps(method_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

    method_name = method_config.get('name', 'unnamed_method')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # dir_name = f"{method_name}_{config_hash}_{timestamp}" # Or simpler name
    dir_name = f"{method_name}_{config_hash}"

    output_path = os.path.join(base_dir, dataset_name, model_name, dir_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def _save_results(output_dir: str, metrics: Dict[str, Any], model: Optional[SkorchImageClassifier] = None, model_filename: str = "model.pt"):
    """Saves metrics (JSON) and optionally the model state."""
    metrics_path = os.path.join(output_dir, "metrics.json")
    try:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        logger.info(f"📊 Metrics saved to {metrics_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save metrics to {metrics_path}: {e}")

    if model is not None:
        model_path = os.path.join(output_dir, model_filename)
        try:
            # Save skorch model (includes optimizer state, history etc.)
            model.save_params(f_params=model_path)
            # Alternatively, save just the PyTorch module's state dict:
            # torch.save(model.module_.state_dict(), model_path.replace('.pt', '_statedict.pt'))
            logger.info(f"💾 Model saved to {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model to {model_path}: {e}")


# --- Method Implementations ---

def run_non_nested_cv(
    model_adapter: SkorchImageClassifier,
    dataset_handler: DatasetHandler,
    method_config: Dict[str, Any],
    output_base_dir: str,
    seed: int
) -> Optional[SkorchImageClassifier]:
    """
    Performs non-nested cross-validation (GridSearch or RandomizedSearch)
    to find the best hyperparameters on the training/validation set.

    Args:
        model_adapter (SkorchImageClassifier): The skorch model wrapper.
        dataset_handler (DatasetHandler): The dataset handler instance.
        method_config (Dict[str, Any]): Configuration for the method, including:
            'name': 'non_nested_cv',
            'search_type': 'GridSearchCV' or 'RandomizedSearchCV',
            'param_grid': Dictionary of hyperparameters to search.
            'cv_folds': Number of inner CV folds.
            'scoring': Scoring metric for search (e.g., 'accuracy').
            'n_iter': Number of iterations for RandomizedSearchCV.
            'val_size': Proportion for validation set if needed (usually handled by CV).
            'save_results': Boolean, whether to save metrics and best model.
            'model_name': Name of the model (for output path).
            'dataset_name': Name of the dataset (for output path).
        output_base_dir (str): Base directory for saving results.
        seed (int): Random seed for reproducibility.

    Returns:
        Optional[SkorchImageClassifier]: The best estimator found by the search,
                                         or None if the search fails.
    """
    logger.info("🚀 Starting Non-Nested CV (Hyperparameter Search)...")
    set_seed(seed) # Ensure reproducibility for CV splits and search

    search_type = method_config.get('search_type', 'GridSearchCV')
    param_grid = method_config.get('param_grid', {})
    cv_folds = method_config.get('cv_folds', 5)
    scoring = method_config.get('scoring', 'accuracy') # Skorch uses 'accuracy' by default if None
    n_iter = method_config.get('n_iter', 10) # For RandomizedSearchCV
    save_results = method_config.get('save_results', True)

    if not param_grid:
        logger.warning("⚠️ No param_grid provided for Non-Nested CV. Skipping search.")
        return model_adapter # Return the original adapter

    # Get training data (potentially split into train/val internally by CV)
    # We use the data designated as 'train' (from FIXED) or 'all' (from FLAT)
    # A validation set might be implicitly created *within* each CV fold by skorch (e.g., for EarlyStopping)
    try:
        train_paths, train_labels, _, _ = dataset_handler.get_train_val_data(val_size=0) # Get all 'trainable' data
        if not train_paths:
             raise ValueError("No training data available for Non-Nested CV.")
        X_train = np.array(train_paths) # Sklearn CV expects indexable X
        y_train = np.array(train_labels)
    except Exception as e:
        logger.error(f"❌ Error getting training data: {e}")
        return None

    # Setup CV strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Setup SearchCV object
    search_cls = GridSearchCV if search_type == 'GridSearchCV' else RandomizedSearchCV
    search_params = {
        'estimator': model_adapter,
        'param_grid': param_grid if search_type == 'GridSearchCV' else {},
        'param_distributions': param_grid if search_type == 'RandomizedSearchCV' else {},
        'cv': cv,
        'scoring': scoring,
        'refit': True, # Refit the best model on the whole training data
        'n_jobs': -1, # Use all available cores, careful with memory
        'verbose': 1, # Log search progress
        'error_score': 'raise' # Or np.nan
    }
    if search_type == 'RandomizedSearchCV':
        search_params['n_iter'] = n_iter
        search_params.pop('param_grid') # Remove param_grid for RandomizedSearchCV
    else:
         search_params.pop('param_distributions') # Remove param_distributions for GridSearchCV

    searcher = search_cls(**search_params)

    try:
        logger.info(f"🔍 Starting {search_type} with {cv_folds} folds...")
        searcher.fit(X_train, y_train) # Skorch adapter handles path loading via collate_fn

        logger.info(f"✅ Non-Nested CV completed.")
        logger.info(f"🏆 Best score ({scoring}): {searcher.best_score_:.4f}")
        logger.info(f"⚙️ Best params: {searcher.best_params_}")

        best_estimator = searcher.best_estimator_

        # --- Save Results (optional) ---
        if save_results:
            output_dir = _get_output_dir(
                output_base_dir,
                method_config.get('dataset_name', 'unknown_dataset'),
                method_config.get('model_name', 'unknown_model'),
                method_config # Pass the whole config for hashing/naming
            )
            results_data = {
                'method': 'non_nested_cv',
                'search_type': search_type,
                'best_score': searcher.best_score_,
                'best_params': searcher.best_params_,
                'cv_results': searcher.cv_results_ # Can be large
            }
            # Save detailed CV results and the best model
            _save_results(output_dir, results_data, best_estimator, model_filename="best_model_non_nested_cv.pt")

        return best_estimator

    except Exception as e:
        logger.error(f"❌ Non-Nested CV failed: {e}", exc_info=True) # Log traceback
        return None


def run_nested_cv(
    model_adapter: SkorchImageClassifier,
    dataset_handler: DatasetHandler,
    method_config: Dict[str, Any],
    output_base_dir: str,
    seed: int
) -> Dict[str, Any]:
    """
    Performs nested cross-validation to get an unbiased estimate of the
    generalization performance of the hyperparameter tuning process.

    Args:
        model_adapter (SkorchImageClassifier): Base skorch model wrapper (hyperparams will be searched).
        dataset_handler (DatasetHandler): Dataset handler instance.
        method_config (Dict[str, Any]): Configuration, including:
            'name': 'nested_cv',
            'search_type': 'GridSearchCV' or 'RandomizedSearchCV'.
            'param_grid': Hyperparameters for the inner search.
            'outer_cv_folds': Number of outer CV folds.
            'inner_cv_folds': Number of inner CV folds (for search).
            'scoring': Scoring metric for both inner search and outer evaluation (can differ).
            'n_iter': Number of iterations for RandomizedSearchCV.
            'save_results': Boolean, whether to save metrics.
            'model_name': Model name.
            'dataset_name': Dataset name.
        output_base_dir (str): Base directory for results.
        seed (int): Random seed.

    Returns:
        Dict[str, Any]: Dictionary containing the scores from the outer loop.
    """
    logger.info("🚀 Starting Nested CV (Performance Estimation)...")
    set_seed(seed)

    search_type = method_config.get('search_type', 'GridSearchCV')
    param_grid = method_config.get('param_grid', {})
    outer_cv_folds = method_config.get('outer_cv_folds', 5)
    inner_cv_folds = method_config.get('inner_cv_folds', 3)
    # Use multiple scorers if needed, e.g., ['accuracy', 'f1_macro']
    scoring = method_config.get('scoring', 'accuracy')
    n_iter = method_config.get('n_iter', 10)
    save_results = method_config.get('save_results', True)

    if not param_grid:
        logger.warning("⚠️ No param_grid provided for Nested CV. Skipping.")
        return {'error': 'No parameter grid provided.'}

    # Get the *entire* dataset for nested CV
    try:
        all_paths, all_labels = dataset_handler.get_all_data()
        if not all_paths:
            raise ValueError("No data available for Nested CV.")
        X = np.array(all_paths)
        y = np.array(all_labels)
    except Exception as e:
        logger.error(f"❌ Error getting data for Nested CV: {e}")
        return {'error': f'Failed to get data: {e}'}

    # Outer CV loop strategy
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=seed)
    # Inner CV loop strategy (for hyperparameter search within each outer fold)
    inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=seed + 1) # Use different seed

    # Setup the inner search object (estimator for cross_validate)
    search_cls = GridSearchCV if search_type == 'GridSearchCV' else RandomizedSearchCV
    search_params = {
        'estimator': model_adapter, # Pass the base adapter
        'param_grid': param_grid if search_type == 'GridSearchCV' else {},
        'param_distributions': param_grid if search_type == 'RandomizedSearchCV' else {},
        'cv': inner_cv,
        'scoring': scoring, # Score for selecting best params in inner loop
        'refit': True, # Refit best model on the inner training fold
        'n_jobs': 1, # Avoid nested parallelism issues, run outer loop sequentially maybe? Or manage resources carefully. Let's try 1 first.
        'verbose': 0, # Keep inner search quiet
        'error_score': 'raise' # Or np.nan
    }
    if search_type == 'RandomizedSearchCV':
        search_params['n_iter'] = n_iter
        search_params.pop('param_grid')
    else:
         search_params.pop('param_distributions')

    inner_searcher = search_cls(**search_params)

    try:
        logger.info(f"🔄 Starting Nested CV with {outer_cv_folds} outer folds and {inner_cv_folds} inner folds...")
        # Use cross_validate for potentially multiple metrics
        # The 'estimator' here is the *searcher* itself.
        # cross_validate will call fit/score on the searcher in each outer fold.
        nested_scores = cross_validate(
            inner_searcher,
            X=X,
            y=y,
            cv=outer_cv,
            scoring=scoring, # Scoring for the *outer* evaluation
            n_jobs=1, # Run outer folds sequentially to avoid resource conflicts
            verbose=1,
            return_train_score=False, # Usually not needed
            error_score='raise'
        )

        # Process results
        avg_test_score = np.mean(nested_scores['test_score'])
        std_test_score = np.std(nested_scores['test_score'])

        logger.info(f"✅ Nested CV completed.")
        logger.info(f"📊 Average Test Score ({scoring}) across outer folds: {avg_test_score:.4f} +/- {std_test_score:.4f}")

        results_data = {
            'method': 'nested_cv',
            'search_type': search_type,
            'outer_cv_folds': outer_cv_folds,
            'inner_cv_folds': inner_cv_folds,
            'scoring': scoring,
            'nested_scores': {k: v.tolist() for k, v in nested_scores.items()}, # Convert numpy arrays
            'average_test_score': avg_test_score,
            'std_test_score': std_test_score,
        }

        if save_results:
             output_dir = _get_output_dir(
                 output_base_dir,
                 method_config.get('dataset_name', 'unknown_dataset'),
                 method_config.get('model_name', 'unknown_model'),
                 method_config
             )
             # Don't save a model here, as nested CV estimates performance, doesn't produce a single final model
             _save_results(output_dir, results_data, model=None)

        return results_data

    except Exception as e:
        logger.error(f"❌ Nested CV failed: {e}", exc_info=True)
        return {'error': f'Nested CV failed: {e}'}


def run_cv_evaluation(
    model_adapter: SkorchImageClassifier,
    dataset_handler: DatasetHandler,
    method_config: Dict[str, Any],
    output_base_dir: str,
    seed: int
) -> Optional[Dict[str, Any]]:
    """
    Performs standard cross-validation on the *entire* dataset to evaluate
    a model with *fixed* hyperparameters. Cannot be used with FIXED datasets.

    Args:
        model_adapter (SkorchImageClassifier): Skorch wrapper with fixed hyperparameters.
        dataset_handler (DatasetHandler): Dataset handler.
        method_config (Dict[str, Any]): Config, including:
            'name': 'cv_evaluation',
            'cv_folds': Number of CV folds.
            'save_results': Boolean.
            'model_name': Model name.
            'dataset_name': Dataset name.
        output_base_dir (str): Base results directory.
        seed (int): Random seed.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with evaluation results across folds, or None.
    """
    logger.info("🚀 Starting Cross-Validation for Model Evaluation...")
    set_seed(seed)

    if dataset_handler.get_dataset_structure() == "FIXED":
        logger.error("❌ CV Evaluation method is incompatible with FIXED dataset structure (which has a predefined test set).")
        return None

    cv_folds = method_config.get('cv_folds', 5)
    save_results = method_config.get('save_results', True)

    # Get the entire dataset (only works for FLAT structure conceptually)
    try:
        all_paths, all_labels = dataset_handler.get_all_data()
        if not all_paths:
             raise ValueError("No data available for CV Evaluation.")
        X = np.array(all_paths)
        y = np.array(all_labels)
    except Exception as e:
        logger.error(f"❌ Error getting data for CV Evaluation: {e}")
        return None

    # CV strategy
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Use cross_val_predict to get out-of-fold predictions for the entire dataset
    # This allows calculating detailed metrics once on the full set of predictions.
    try:
        logger.info(f"🔄 Performing {cv_folds}-fold cross-validation prediction...")
        # Ensure the model adapter passed has fixed hyperparameters
        # Clone the estimator to avoid state leakage if it was already fitted
        # from sklearn.base import clone
        # cloned_adapter = clone(model_adapter) # Cloning might be complex with callbacks/transforms

        # We assume model_adapter is instantiated fresh or its state doesn't matter here
        # cross_val_predict fits a *new* model on each fold.

        y_pred_oof = cross_val_predict(model_adapter, X, y, cv=cv, n_jobs=1, verbose=1) # Out-of-fold predictions

        # Get probabilities (needs predict_proba method)
        # This might require fitting again or modifying how predict_proba works with CV
        # For simplicity, let's try getting probs - this might be slow as it runs predict_proba per fold
        try:
            y_prob_oof = cross_val_predict(model_adapter, X, y, cv=cv, method='predict_proba', n_jobs=1, verbose=1)
        except Exception as proba_e:
            logger.warning(f"⚠️ Could not get out-of-fold probabilities: {proba_e}. AUC/AUPRC metrics will be unavailable.")
            y_prob_oof = np.zeros((len(y), dataset_handler.get_num_classes())) # Placeholder

        logger.info("✅ Cross-validation prediction completed.")

        # Calculate metrics using the OOF predictions
        class_names = dataset_handler.get_classes()
        metrics = calculate_metrics(y_true=y, y_pred=y_pred_oof, y_prob=y_prob_oof, class_names=class_names)

        logger.info("📊 Aggregated Metrics from CV Evaluation (Out-of-Fold Predictions):")
        log_str = format_metrics_log(metrics, class_names)
        logger.info(log_str)

        results_data = {
            'method': 'cv_evaluation',
            'cv_folds': cv_folds,
            'metrics': metrics,
             # Optionally include raw OOF predictions (can be large)
            # 'y_pred_oof': y_pred_oof.tolist(),
            # 'y_prob_oof': y_prob_oof.tolist(),
        }

        if save_results:
            output_dir = _get_output_dir(
                output_base_dir,
                method_config.get('dataset_name', 'unknown_dataset'),
                method_config.get('model_name', 'unknown_model'),
                method_config
            )
            # No single model to save here, as multiple were trained during CV
            _save_results(output_dir, results_data, model=None)

        return results_data

    except Exception as e:
        logger.error(f"❌ CV Evaluation failed: {e}", exc_info=True)
        return None


def run_single_train(
    model_adapter: SkorchImageClassifier,
    dataset_handler: DatasetHandler,
    method_config: Dict[str, Any],
    output_base_dir: str,
    seed: int
) -> Optional[SkorchImageClassifier]:
    """
    Performs a single training run on a train/validation split.

    Args:
        model_adapter (SkorchImageClassifier): Skorch wrapper instance.
        dataset_handler (DatasetHandler): Dataset handler.
        method_config (Dict[str, Any]): Config, including:
            'name': 'single_train',
            'val_size': Validation set proportion (e.g., 0.2).
            'save_results': Boolean, whether to save the trained model and history.
            'model_name': Model name.
            'dataset_name': Dataset name.
        output_base_dir (str): Base results directory.
        seed (int): Random seed.

    Returns:
        Optional[SkorchImageClassifier]: The trained model adapter, or None if failed.
    """
    logger.info("🚀 Starting Single Train process...")
    set_seed(seed)

    val_size = method_config.get('val_size', 0.2)
    save_results = method_config.get('save_results', True)

    # Get train/validation split
    try:
        train_paths, train_labels, val_paths, val_labels = dataset_handler.get_train_val_data(val_size=val_size, stratify=True)
        if not train_paths:
             raise ValueError("No training data available for Single Train.")
        if val_size > 0 and not val_paths:
             logger.warning("⚠️ Requested validation set, but got empty validation data (maybe small dataset?). Proceeding without validation.")
             val_size = 0 # Treat as no validation

        X_train = np.array(train_paths)
        y_train = np.array(train_labels)
        X_val = np.array(val_paths) if val_paths else None
        y_val = np.array(val_labels) if val_labels else None

    except Exception as e:
        logger.error(f"❌ Error getting train/val data: {e}")
        return None

    try:
        logger.info(f"💪 Starting training on {len(X_train)} samples, validating on {len(X_val) if X_val is not None else 0} samples...")

        # Skorch handles validation data via fit_params or internal split.
        # We provide it explicitly using the `validation_data` argument format expected by our overridden `fit`.
        fit_params = {}
        if X_val is not None and y_val is not None:
            # Our adapter's fit method expects 'X_valid', 'y_valid' keys
            fit_params['X_valid'] = X_val.tolist() # Pass lists of paths/labels
            fit_params['y_valid'] = y_val.tolist()


        # Fit the model
        model_adapter.fit(X_train.tolist(), y_train.tolist(), **fit_params)

        logger.info(f"✅ Single Train completed.")
        # Log training history summary if available
        history = model_adapter.history
        if history:
             best_epoch = np.argmin(history[:, 'valid_loss']) # Assuming EarlyStopping loads best
             best_val_loss = history[best_epoch, 'valid_loss']
             train_loss_at_best = history[best_epoch, 'train_loss']
             logger.info(f"📉 Training history summary: Best validation loss {best_val_loss:.4f} at epoch {best_epoch+1} (Train loss: {train_loss_at_best:.4f})")


        if save_results:
            output_dir = _get_output_dir(
                output_base_dir,
                method_config.get('dataset_name', 'unknown_dataset'),
                method_config.get('model_name', 'unknown_model'),
                method_config
            )
            # Save history along with the model
            results_data = {
                'method': 'single_train',
                'training_history': history.to_list() # Skorch history can be complex, convert simply
            }
            _save_results(output_dir, results_data, model=model_adapter, model_filename="trained_model.pt")

        return model_adapter

    except Exception as e:
        logger.error(f"❌ Single Train failed: {e}", exc_info=True)
        return None


def run_single_eval(
    model_adapter: SkorchImageClassifier,
    dataset_handler: DatasetHandler,
    method_config: Dict[str, Any],
    output_base_dir: str,
    seed: int # Seed not strictly needed for eval, but kept for consistency
) -> Optional[Dict[str, Any]]:
    """
    Evaluates a trained model on the test set.

    Args:
        model_adapter (SkorchImageClassifier): The *trained* skorch model wrapper.
        dataset_handler (DatasetHandler): Dataset handler instance.
        method_config (Dict[str, Any]): Config, including:
            'name': 'single_eval',
            'save_results': Boolean.
            'model_name': Model name.
            'dataset_name': Dataset name.
        output_base_dir (str): Base results directory.
        seed (int): Random seed.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with evaluation metrics, or None.
    """
    logger.info("🚀 Starting Single Evaluation on Test Set...")
    set_seed(seed) # Set seed for deterministic dropout etc. if model uses it during eval

    save_results = method_config.get('save_results', True)

    # Get test data
    test_data = dataset_handler.get_test_data()
    if test_data is None:
        if dataset_handler.get_dataset_structure() == "FLAT":
             logger.error("❌ Cannot perform Single Eval: Dataset structure is FLAT, no predefined test set.")
        else: # FIXED structure, but test data wasn't loaded/found
             logger.error("❌ Cannot perform Single Eval: No test data found for FIXED dataset structure.")
        return None

    test_paths, test_labels = test_data
    X_test = np.array(test_paths)
    y_test = np.array(test_labels)

    if not model_adapter.initialized_:
         logger.error("❌ Cannot perform Single Eval: Model adapter is not fitted/initialized.")
         return None

    try:
        logger.info(f"🧪 Evaluating model on {len(X_test)} test samples...")

        # Set model to evaluation mode (handled by skorch predict/predict_proba)
        # Make predictions
        y_pred = model_adapter.predict(X_test.tolist()) # Pass list of paths

        # Get probabilities
        try:
            y_prob = model_adapter.predict_proba(X_test.tolist())
        except Exception as proba_e:
            logger.warning(f"⚠️ Could not get probabilities during evaluation: {proba_e}. AUC/AUPRC metrics will be unavailable.")
            y_prob = np.zeros((len(y_test), dataset_handler.get_num_classes())) # Placeholder

        # Calculate metrics
        class_names = dataset_handler.get_classes()
        metrics = calculate_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob, class_names=class_names)

        logger.info("📊 Evaluation Metrics on Test Set:")
        log_str = format_metrics_log(metrics, class_names)
        logger.info(log_str)

        results_data = {
            'method': 'single_eval',
            'metrics': metrics,
             # Optionally save predictions (can be large)
            # 'y_pred': y_pred.tolist(),
            # 'y_prob': y_prob.tolist(),
        }

        if save_results:
            output_dir = _get_output_dir(
                output_base_dir,
                method_config.get('dataset_name', 'unknown_dataset'),
                method_config.get('model_name', 'unknown_model'),
                method_config
            )
            # Don't save the model here, just the results
            _save_results(output_dir, results_data, model=None)

        return results_data

    except Exception as e:
        logger.error(f"❌ Single Evaluation failed: {e}", exc_info=True)
        return None

# Mapping method names to functions
METHOD_REGISTRY = {
    'non_nested_cv': run_non_nested_cv,
    'nested_cv': run_nested_cv,
    'cv_evaluation': run_cv_evaluation,
    'single_train': run_single_train,
    'single_eval': run_single_eval,
}
