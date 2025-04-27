# --- START OF FILE adapters.py ---

import torch
from torch.utils.data.dataloader import default_collate
# Import the base Dataset class from skorch
from skorch.dataset import Dataset as SkorchDatasetBase
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Callback
from skorch.dataset import ValidSplit # Keep import just in case needed elsewhere
from typing import Callable, List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np
import functools # Import functools

from utils import logger


# --- Custom Skorch-compatible Dataset ---
class PathLabelDataset(SkorchDatasetBase):
    """
    Custom skorch Dataset that correctly handles lists of paths for X
    and determines length based on X. Iteration yields (X[i], y[i]).
    """
    def __init__(self, X, y=None, length=None):
        self.X = X
        self.y = y
        if length is not None: self._len = length
        elif X is not None: self._len = len(X)
        elif y is not None: self._len = len(y)
        else: self._len = 0
        if X is not None and y is not None:
             # Allow y to be None during length check if X is primary data source
             # This happens during predict when only X is passed.
             if y is not None and len(X) != len(y):
                  raise ValueError(f"X and y have inconsistent lengths ({len(X)} != {len(y)}).")
    def __len__(self): return self._len
    def __getitem__(self, i):
        # Yields (path, label) tuple
        Xi = self.X[i] if self.X is not None else None
        yi = self.y[i] if self.y is not None else None
        return Xi, yi

# --- Helper for loading single image ---
def _load_and_transform_single(img_path: str, label: int, transform: Optional[Callable]):
    """Loads and transforms a single image, handling errors."""
    try:
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        return img, label
    except Exception as e:
        logger.error(f"❌ Error loading/transforming single image path='{img_path}' (label={label}): {e}", exc_info=True)
        return None, label

# --- Unified Collate function for (path, label) data ---
def collate_path_label_data(batch: List[Tuple[str, int]], transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Collate function for (path, label) tuples yielded by PathLabelDataset. """
    images = []
    labels = []
    processed_count = 0
    failed_count = 0
    for img_path, label in batch:
        if not isinstance(img_path, str):
             logger.error(f"Collate Error: Expected string path, got {type(img_path)}. Skipping.")
             failed_count+=1
             continue
        # Handle None label case gracefully if it ever occurs, though unlikely here
        current_label = label if label is not None else -1 # Use a placeholder if label is None

        img, lbl = _load_and_transform_single(img_path, current_label, transform)
        if img is not None:
             images.append(img)
             labels.append(lbl)
             processed_count += 1
        else:
             failed_count += 1

    if failed_count > 0: logger.warning(f"⚠️ Collate Path/Label: Failed {failed_count}/{len(batch)} loads.")
    if not images:
        logger.error(f"❌ Collate Path/Label: Returning empty batch.")
        h, w = (128, 128) # Use IMAGE_SIZE from main? Needs better way to get size
        return torch.empty((0, 3, h, w), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    images_tensor = default_collate(images)
    labels_tensor = default_collate(labels)
    labels_tensor = labels_tensor.long() # Ensure long type

    # logger.debug(f"Collate Path/Label: Processed {processed_count}. Img shape: {images_tensor.shape}, Lbl shape: {labels_tensor.shape}, Lbl dtype: {labels_tensor.dtype}")
    return images_tensor, labels_tensor


# --- Skorch NeuralNetClassifier Wrapper ---
class SkorchImageClassifier(NeuralNetClassifier):
    """
    Skorch wrapper using PathLabelDataset and path-loading collate_fn.
    Includes necessary overrides for fit and infer.
    """
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
        if train_transform is None or val_transform is None: raise ValueError("Transforms required.")
        self.train_transform = train_transform
        self.val_transform = val_transform
        train_collate_partial = functools.partial(collate_path_label_data, transform=self.train_transform)
        val_test_collate_partial = functools.partial(collate_path_label_data, transform=self.val_transform)

        # Setup Callbacks
        default_callbacks: List[Tuple[str, Callback]] = []
        if patience > 0: default_callbacks.append(('early_stopping', EarlyStopping(monitor=monitor, patience=patience, lower_is_better=True, load_best=True)))
        else: logger.warning("⚠️ Early stopping disabled.")
        if lr_scheduler: default_callbacks.append(('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', patience=lr_scheduler_patience, factor=0.1, monitor='valid_loss')))
        final_callbacks = default_callbacks
        if callbacks is not None: # Merge user callbacks
            # Placeholder for actual merging logic if needed
            pass

        logger.info(f"🤖 Initializing SkorchImageClassifier with device: {device}")
        logger.info(f"Callbacks: {[name for name, cb in final_callbacks]}")

        super().__init__(
            module=module, criterion=criterion, optimizer=optimizer, lr=lr, batch_size=batch_size,
            max_epochs=max_epochs, device=device, callbacks=final_callbacks,
            dataset=PathLabelDataset, # Use custom dataset
            iterator_train__collate_fn=train_collate_partial,
            iterator_valid__collate_fn=val_test_collate_partial,
            iterator_valid__batch_size=batch_size, # Explicitly set valid batch size
            train_split=None, # Keep as None
            predict_nonlinearity='auto',
            *args, **kwargs
        )

    # --- fit method ---
    def fit(self, X, y=None, **fit_params):
        """
        Overrides fit to handle path inputs (as list/array) and validation data,
        passing them directly to super().fit() for internal PathLabelDataset creation.
        """
        if y is None: raise ValueError("Labels (y) required.")
        # Process Train Data
        if isinstance(X, np.ndarray): X_train_processed = X.tolist()
        elif isinstance(X, list): X_train_processed = X
        else: raise TypeError(f"Train X type: {type(X)}")
        if isinstance(y, torch.Tensor): y_train_processed = y.cpu().numpy()
        elif isinstance(y, list): y_train_processed = np.array(y)
        elif isinstance(y, np.ndarray): y_train_processed = y
        else: raise TypeError(f"Train y type: {type(y)}")
        if len(X_train_processed) != len(y_train_processed): raise ValueError("Train X/y length mismatch.")

        # Process Validation Data
        X_valid_processed, y_valid_processed = None, None
        if 'X_valid' in fit_params and 'y_valid' in fit_params:
            X_valid_raw, y_valid_raw = fit_params.pop('X_valid'), fit_params.pop('y_valid')
            if isinstance(X_valid_raw, np.ndarray): X_valid_processed = X_valid_raw.tolist()
            elif isinstance(X_valid_raw, list): X_valid_processed = X_valid_raw
            else: raise TypeError(f"Val X type: {type(X_valid_raw)}")
            if isinstance(y_valid_raw, torch.Tensor): y_valid_processed = y_valid_raw.cpu().numpy()
            elif isinstance(y_valid_raw, list): y_valid_processed = np.array(y_valid_raw)
            elif isinstance(y_valid_raw, np.ndarray): y_valid_processed = y_valid_raw
            else: raise TypeError(f"Val y type: {type(y_valid_raw)}")
            if len(X_valid_processed) != len(y_valid_processed): raise ValueError("Val X/y length mismatch.")
            logger.info("Passing validation data (paths and labels) directly to super().fit.")
        else: logger.info("Fitting without explicit validation data.")

        # Call Parent Fit Method
        logger.debug(f"Calling super().fit with: "
                     f"X=List(len={len(X_train_processed)}), "
                     f"y=np.array(shape={y_train_processed.shape}), "
                     f"X_valid={type(X_valid_processed)}(len={len(X_valid_processed) if X_valid_processed else 0}), "
                     f"y_valid={type(y_valid_processed)}(shape={y_valid_processed.shape if y_valid_processed is not None else None})")

        super().fit(
            X=X_train_processed, y=y_train_processed,
            X_valid=X_valid_processed, y_valid=y_valid_processed,
            **fit_params
        )
        return self


    # --- RESTORED infer method ---
    def infer(self, x, **fit_params):
        """
        Override infer to manually remove keys related to validation data
        ('X_valid', 'y_valid') from fit_params before passing them to the
        underlying module via super().infer().
        This prevents TypeError in the module's forward method when external
        validation data is provided to the fit method.
        """
        filtered_fit_params = fit_params.copy()
        # Remove keys that should not reach the module's forward()
        filtered_fit_params.pop('X_valid', None)
        filtered_fit_params.pop('y_valid', None)
        filtered_fit_params.pop('validation_data', None) # Still good practice
        # logger.debug(f"Infer called. Filtered fit_params: {filtered_fit_params.keys()}") # Optional debug
        return super().infer(x, **filtered_fit_params)

    # --- get_params / set_params ---
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if not deep:
            params['train_transform'] = self.train_transform
            params['val_transform'] = self.val_transform
        return params

    def set_params(self, **params):
        if 'train_transform' in params: self.train_transform = params.pop('train_transform')
        if 'val_transform' in params: self.val_transform = params.pop('val_transform')
        super().set_params(**params)
        return self

# --- END OF FILE adapters.py ---