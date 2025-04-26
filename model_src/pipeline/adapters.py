# --- START OF FILE adapters.py ---

import torch
from torch.utils.data.dataloader import default_collate
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Callback # Keep imports
from skorch.dataset import Dataset as SkorchDataset
from skorch.dataset import ValidSplit
from typing import Callable, List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import functools # Import functools

from utils import logger


# --- Helper for loading single image ---
def _load_and_transform_single(img_path: str, label: int, transform: Optional[Callable]):
    """Loads and transforms a single image, handling errors."""
    try:
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        # logger.debug(f"Successfully loaded image: {img_path}") # Optional: Very verbose
        return img, label
    except Exception as e:
        # Make sure full traceback is logged for loading errors during debug
        logger.error(f"❌ Error loading/transforming single image path='{img_path}' (label={label}): {e}", exc_info=True) # Log traceback for image errors
        return None, label # Signal failure


# --- Collate function for Train Data ---
def collate_train_data(batch: List[Tuple[Tuple[str, int], int]], transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for training data.
    Expects batch items structured as ((path, label), label).
    Uses the label paired with the path.
    """
    # logger.debug(f"Collate Train: Received batch structure: {batch[0] if batch else 'Empty'}") # Optional debug
    images = []
    labels = []
    processed_count = 0
    failed_count = 0
    for item_tuple, _ in batch:
        img_path, label = item_tuple
        img, lbl = _load_and_transform_single(img_path, label, transform)
        if img is not None:
             images.append(img)
             labels.append(lbl)
             processed_count += 1
        else:
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"⚠️ Collate Train: Failed to load {failed_count}/{len(batch)} images in this batch.")

    if not images:
        logger.error(f"❌ Collate Train: Returning COMPLETELY empty batch (all {len(batch)} image loads failed).")
        return torch.empty((0, 3, 224, 224), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    logger.debug(f"Collate Train: Processed {processed_count} images.")
    return default_collate(images), default_collate(labels)


# --- Collate function for Validation/Test Data ---
def collate_val_test_data(batch: List[Tuple[str, int]], transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for validation or test data.
    Expects batch items structured as (path, label).
    """
    # logger.debug(f"Collate Val/Test: Received batch structure: {batch[0] if batch else 'Empty'}") # Optional debug
    images = []
    labels = []
    processed_count = 0
    failed_count = 0
    for img_path, label in batch: # Direct unpacking
        img, lbl = _load_and_transform_single(img_path, label, transform)
        if img is not None:
             images.append(img)
             labels.append(lbl)
             processed_count += 1
        else:
             failed_count += 1

    if failed_count > 0:
        logger.warning(f"⚠️ Collate Val/Test: Failed to load {failed_count}/{len(batch)} images in this batch.")

    if not images:
        logger.error(f"❌ Collate Val/Test: Returning COMPLETELY empty batch (all {len(batch)} image loads failed).")
        return torch.empty((0, 3, 224, 224), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    # --- ADD DEBUG LOG ---
    logger.debug(f"Collate Val/Test: Successfully processed {processed_count} images for validation/test batch.")
    # --- END DEBUG LOG ---

    return default_collate(images), default_collate(labels)


# --- Skorch NeuralNetClassifier Wrapper ---

class SkorchImageClassifier(NeuralNetClassifier):
    """
    A skorch wrapper for PyTorch image classification models,
    handling transformations via custom collate functions.
    """
    # --- __init__ ---
    # Keep the version where EarlyStopping IS enabled
    def __init__(
            self,
            module: torch.nn.Module,
            criterion: torch.nn.Module = torch.nn.CrossEntropyLoss,
            optimizer: torch.optim.Optimizer = torch.optim.AdamW,
            lr: float = 1e-4,
            batch_size: int = 32,
            max_epochs: int = 10,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            callbacks: Optional[List[Any]] = None,
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            patience: int = 5,
            monitor: str = 'valid_loss',
            lr_scheduler: bool = False,
            lr_scheduler_patience: int = 3,
            *args, **kwargs
    ):
        if train_transform is None or val_transform is None:
            raise ValueError("Both train_transform and val_transform must be provided.")
        self.train_transform = train_transform
        self.val_transform = val_transform
        train_collate_partial = functools.partial(collate_train_data, transform=self.train_transform)
        val_test_collate_partial = functools.partial(collate_val_test_data, transform=self.val_transform)

        # Setup Callbacks (Restored)
        default_callbacks: List[Tuple[str, Callback]] = []
        if patience > 0:
            default_callbacks.append(
                ('early_stopping', EarlyStopping(monitor=monitor, patience=patience, lower_is_better=True, load_best=True))
            )
        else: logger.warning("⚠️ Early stopping disabled as patience <= 0.")
        if lr_scheduler:
            default_callbacks.append(
                ('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', patience=lr_scheduler_patience, factor=0.1, monitor='valid_loss'))
            )
        final_callbacks = default_callbacks
        if callbacks is not None: # Merge user-provided callbacks
            callback_dict = {name: cb for name, cb in default_callbacks}
            user_callbacks_processed = []
            for cb in callbacks:
                name = None; callback_instance = None
                if isinstance(cb, tuple): name, callback_instance = cb[0], cb[1]
                else: name, callback_instance = cb.__class__.__name__, cb
                if name in callback_dict: logger.warning(f"⚠️ Overriding default callback '{name}'...")
                processed_names = [c[0] for c in user_callbacks_processed if isinstance(c, tuple)] + \
                                  [c[1].__class__.__name__ for c in user_callbacks_processed if not isinstance(c, tuple)]
                if name in processed_names: logger.warning(f"⚠️ Duplicate user callback '{name}' provided. Using last one.")
                callback_dict[name] = callback_instance
                user_callbacks_processed.append((name, callback_instance))
            final_callbacks = list(callback_dict.items())

        logger.info(f"🤖 Initializing SkorchImageClassifier with device: {device}")
        logger.info(f"Callbacks: {[name for name, cb in final_callbacks]}")

        super().__init__(
            module=module,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            device=device,
            callbacks=final_callbacks, # Use the merged/default list
            iterator_train__collate_fn=train_collate_partial,
            iterator_valid__collate_fn=val_test_collate_partial,
            train_split=None,
            predict_nonlinearity='auto',
            *args, **kwargs
        )

    # --- fit method ---
    def fit(self, X, y=None, **fit_params):
        """ Overrides fit to handle path inputs and correctly pass validation data. """
        if y is None: raise ValueError("Target labels (y) are required during training fit.")
        if isinstance(X, np.ndarray): X_processed = X.tolist()
        elif isinstance(X, list): X_processed = X
        else: raise TypeError(f"Input X should be list/array of paths, got {type(X)}")
        if isinstance(y, torch.Tensor): y_for_check, y_for_tuples = y.cpu().numpy(), y.tolist()
        elif isinstance(y, list): y_for_check, y_for_tuples = np.array(y), y
        elif isinstance(y, np.ndarray): y_for_check, y_for_tuples = y, y.tolist()
        else: raise TypeError(f"Input y should be list/array/tensor, got {type(y)}")
        if len(X_processed) != len(y_for_check): raise ValueError(f"Input X length {len(X_processed)} and y length {len(y_for_check)} must match.")

        path_label_tuples = list(zip(X_processed, y_for_tuples))
        dataset = SkorchDataset(path_label_tuples, y=None)

        X_valid_processed, y_valid_processed = None, None
        if 'X_valid' in fit_params and 'y_valid' in fit_params:
            X_valid_raw, y_valid_raw = fit_params.pop('X_valid'), fit_params.pop('y_valid')
            if isinstance(X_valid_raw, np.ndarray): X_valid_paths = X_valid_raw.tolist()
            elif isinstance(X_valid_raw, list): X_valid_paths = X_valid_raw
            else: raise TypeError(f"Input X_valid should be list/array of paths, got {type(X_valid_raw)}")
            if isinstance(y_valid_raw, torch.Tensor): y_valid_processed = y_valid_raw.cpu().numpy()
            elif isinstance(y_valid_raw, list): y_valid_processed = np.array(y_valid_raw)
            elif isinstance(y_valid_raw, np.ndarray): y_valid_processed = y_valid_raw
            else: raise TypeError(f"Input y_valid should be list/array/tensor, got {type(y_valid_raw)}")
            if len(X_valid_paths) != len(y_valid_processed): raise ValueError(f"X_valid length {len(X_valid_paths)} and y_valid length {len(y_valid_processed)} must match.")
            X_valid_processed = X_valid_paths
            logger.info("Passing validation data (paths and labels) directly to super().fit.")
        else: logger.info("Fitting without explicit validation data.")

        # --- ADD DEBUG PRINT ---
        x_valid_type = type(X_valid_processed) if X_valid_processed is not None else None
        y_valid_type = type(y_valid_processed) if y_valid_processed is not None else None
        x_valid_len = len(X_valid_processed) if X_valid_processed is not None else 0
        y_valid_len = len(y_valid_processed) if y_valid_processed is not None else 0
        logger.debug(f"Calling super().fit with: "
                     f"X=SkorchDataset(len={len(dataset)}), "
                     f"y=np.array(shape={y_for_check.shape}), "
                     f"X_valid={x_valid_type}(len={x_valid_len}), "
                     f"y_valid={y_valid_type}(len={y_valid_len})")
        # --- END DEBUG PRINT ---

        super().fit(
            X=dataset,
            y=y_for_check,
            X_valid=X_valid_processed,
            y_valid=y_valid_processed,
            **fit_params
        )
        return self

    # --- infer method ---
    def infer(self, x, **fit_params):
        """ Override infer to manually remove keys related to validation data """
        filtered_fit_params = fit_params.copy()
        filtered_fit_params.pop('X_valid', None)
        filtered_fit_params.pop('y_valid', None)
        filtered_fit_params.pop('validation_data', None)
        return super().infer(x, **filtered_fit_params)

    # --- get_params / set_params ---
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if not deep:
            params['train_transform'] = self.train_transform
            params['val_transform'] = self.val_transform
        return params

    def set_params(self, **params):
        if 'train_transform' in params:
            self.train_transform = params.pop('train_transform')
        if 'val_transform' in params:
            self.val_transform = params.pop('val_transform')
        super().set_params(**params)
        return self

# --- END OF FILE adapters.py ---