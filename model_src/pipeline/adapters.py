import torch
from torch.utils.data.dataloader import default_collate
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset as SkorchDataset  # Avoid name collision
from typing import Callable, List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np

from utils import logger


# --- Collate Functions ---

def load_and_transform(batch: List[Tuple[str, int]], transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads images from paths in a batch and applies transformations.
    Used by the skorch iterators via the collate_fn.

    Args:
        batch (List[Tuple[str, int]]): A list of (image_path, label) tuples.
        transform (Callable): The torchvision transform to apply.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed images tensor and labels tensor.
    """
    images = []
    labels = []
    for img_path, label in batch:
        try:
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
            labels.append(label)
        except Exception as e:
            logger.error(f"❌ Error loading/transforming image {img_path} in collate_fn: {e}")
            # Handle error: skip image, use placeholder, or raise?
            # Using a placeholder might skew training/validation. Skipping changes batch size.
            # Let's log and skip for now. A better approach might be pre-filtering bad images.
            continue  # Skip this sample

    if not images:  # If all images in the batch failed
        return torch.empty(0), torch.empty(0)  # Return empty tensors

    # Stack images and convert labels to tensor
    images_tensor = default_collate(images)  # Stacks tensors correctly
    labels_tensor = default_collate(labels)  # Handles list of ints -> tensor
    return images_tensor, labels_tensor


# --- Skorch NeuralNetClassifier Wrapper ---

class SkorchImageClassifier(NeuralNetClassifier):
    """
    A skorch wrapper for PyTorch image classification models,
    handling transformations via custom collate functions.
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
            callbacks: Optional[List[Any]] = None,  # Specify skorch callbacks type later if needed
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            patience: int = 5,  # for EarlyStopping
            monitor: str = 'valid_loss',  # for EarlyStopping
            lr_scheduler: bool = False,  # Add CosineAnnealingLR?
            lr_scheduler_patience: int = 3,  # for ReduceLROnPlateau
            *args, **kwargs
    ):
        """
        Args:
            module (torch.nn.Module): The PyTorch model.
            criterion: The loss function.
            optimizer: The optimizer class.
            lr (float): Learning rate.
            batch_size (int): Training and validation batch size.
            max_epochs (int): Maximum number of training epochs.
            device (str): 'cuda' or 'cpu'.
            callbacks (Optional[List]): List of skorch callbacks. EarlyStopping is added by default.
            train_transform (Optional[Callable]): Transformations for the training set.
            val_transform (Optional[Callable]): Transformations for the validation/test set.
            patience (int): Patience for EarlyStopping.
            monitor (str): Metric to monitor for EarlyStopping.
            lr_scheduler (bool): Whether to use ReduceLROnPlateau scheduler.
            lr_scheduler_patience (int): Patience for ReduceLROnPlateau scheduler.
        """
        if train_transform is None or val_transform is None:
            raise ValueError("Both train_transform and val_transform must be provided.")

        self.train_transform = train_transform
        self.val_transform = val_transform

        # --- Define Collate Functions ---
        # These functions capture the respective transforms
        def train_collate_fn(batch):
            return load_and_transform(batch, self.train_transform)

        def val_collate_fn(batch):
            return load_and_transform(batch, self.val_transform)

        # --- Setup Callbacks ---
        default_callbacks = [
            ('early_stopping', EarlyStopping(monitor=monitor, patience=patience, lower_is_better=True, load_best=True)),
            # Add other default callbacks if needed
        ]
        if lr_scheduler:
            # Using ReduceLROnPlateau - monitors validation loss
            default_callbacks.append(
                ('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', patience=lr_scheduler_patience, factor=0.1,
                                             monitor='valid_loss'))
            )
            # Note: CosineAnnealingLR might be better but needs T_max (total steps)

        if callbacks is not None:
            # Allow user to override defaults by name or add new ones
            callback_dict = {name: cb for name, cb in default_callbacks}
            for cb in callbacks:
                if isinstance(cb, tuple):  # User provided (name, callback)
                    callback_dict[cb[0]] = cb[1]
                else:  # User provided callback instance, generate name
                    callback_dict[cb.__class__.__name__] = cb
            final_callbacks = list(callback_dict.items())
        else:
            final_callbacks = default_callbacks

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
            callbacks=final_callbacks,
            # Pass the collate functions to skorch iterators
            iterator_train__collate_fn=train_collate_fn,
            iterator_valid__collate_fn=val_collate_fn,
            # Skorch needs predict_nonlinearity for predict_proba
            predict_nonlinearity='auto',  # 'auto' usually works (softmax for classification)
            *args, **kwargs
        )

    def fit(self, X, y=None, **fit_params):
        """
        Overrides fit to handle path inputs.
        Sklearn CV methods will pass subsets of X (which are paths) here.
        We package X and y into a Skorch Dataset of tuples.
        """
        # Skorch expects X, y. We have paths in X, labels in y.
        # Create a dataset of (path, label) tuples.
        # Skorch Dataset can handle various inputs, including lists.
        # Ensure y is included if provided (needed for training)
        if y is not None:
            # Ensure y is list or np.array for indexing compatibility with Skorch Dataset
            if isinstance(y, torch.Tensor):
                y_processed = y.tolist()
            else:
                y_processed = y  # Assume list or np.array

            # Check length consistency
            if len(X) != len(y_processed):
                raise ValueError(
                    f"Input X (paths) and y (labels) must have the same length. Got {len(X)} and {len(y_processed)}")

            # Package as list of tuples: [(path1, label1), (path2, label2), ...]
            path_label_tuples = list(zip(X, y_processed))
            dataset = SkorchDataset(path_label_tuples, y=None)  # y=None because labels are inside X now

            # Handle validation split if provided in fit_params
            # skorch expects validation data in fit_params['X_valid'], fit_params['y_valid']
            # We need to adapt if user passes validation paths/labels
            if 'X_valid' in fit_params and 'y_valid' in fit_params:
                X_valid_paths = fit_params.pop('X_valid')
                y_valid_labels = fit_params.pop('y_valid')
                if isinstance(y_valid_labels, torch.Tensor):
                    y_valid_processed = y_valid_labels.tolist()
                else:
                    y_valid_processed = y_valid_labels

                valid_path_label_tuples = list(zip(X_valid_paths, y_valid_processed))
                valid_dataset = SkorchDataset(valid_path_label_tuples, y=None)
                fit_params['validation_data'] = valid_dataset  # Use skorch's validation_data parameter

            # Call parent fit method with the structured dataset
            # super().fit(X=dataset, y=None, **fit_params) # Pass y=None as labels are inside dataset
            # Updated skorch versions might prefer passing validation data differently
            # Check skorch docs for `validation_data` or alternative validation split methods
            # Using `train_split=skorch.dataset.ValidSplit(...)` with `validation_data` might be preferred

            # Let's use the standard skorch train_split=None and pass validation data explicitly if needed
            # Skorch handles validation split internally if train_split is not None
            # If we provide validation_data, train_split should typically be None

            # If using skorch > 0.11, pass validation data directly to fit:
            if 'validation_data' in fit_params:
                logger.info("Fitting with provided validation data.")
                super().fit(X=dataset, y=None, **fit_params)
            else:
                # If no explicit validation data, let skorch handle internal splitting if train_split is configured
                # By default, skorch uses a 20% validation split if train_split is not specified and no validation_data
                # We rely on EarlyStopping using this internal validation set.
                logger.info("Fitting without explicit validation data. Skorch may use internal validation split.")
                super().fit(X=dataset, y=None, **fit_params)

        else:
            # If y is None (e.g., during prediction or certain CV scenarios where y is inferred)
            # Create dataset with dummy labels (or handle appropriately based on context)
            # This case needs careful handling depending on *why* y is None.
            # For typical sklearn fit, y should not be None.
            # For predict, fit is not called.
            logger.warning("⚠️ SkorchImageClassifier.fit called with y=None. This might be unexpected during training.")
            # Assuming X contains paths, we still need to pass *something* skorch Dataset expects
            # Passing dummy labels might be necessary if skorch requires a target, even if unused later.
            dummy_labels = [0] * len(X)
            path_label_tuples = list(zip(X, dummy_labels))
            dataset = SkorchDataset(path_label_tuples, y=None)
            super().fit(X=dataset, y=None, **fit_params)

        return self

    # No need to override predict/predict_proba - parent methods work with the collate_fn
    # when iterating over test data passed as X (which will be paths).
    # The iterator_valid with val_collate_fn will be used automatically.

    def get_params(self, deep=True):
        # Ensure transforms are not deep copied if they are complex objects or closures
        params = super().get_params(deep=deep)
        if not deep:
            # Prevent deep copying of transforms if they cause issues
            params['train_transform'] = self.train_transform
            params['val_transform'] = self.val_transform
        return params

    def set_params(self, **params):
        # Handle setting transform parameters if needed, otherwise rely on parent
        if 'train_transform' in params:
            self.train_transform = params.pop('train_transform')
        if 'val_transform' in params:
            self.val_transform = params.pop('val_transform')
        super().set_params(**params)
        return self
