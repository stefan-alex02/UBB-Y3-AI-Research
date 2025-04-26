import torch
from torch.utils.data.dataloader import default_collate
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Callback
from skorch.dataset import Dataset as SkorchDataset  # Avoid name collision
from skorch.dataset import ValidSplit
from typing import Callable, List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np

from utils import logger


# --- Collate Functions ---

def load_and_transform(batch: List[Tuple[Tuple[str, int], int]], transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads images from paths in a batch and applies transformations.
    Assumes batch items are structured as ((path, label), label) due to
    how SkorchDataset combines X=(path, label) and y=label during iteration.

    Args:
        batch (List[Tuple[Tuple[str, int], int]]): A list of items like ((path, label), label).
        transform (Callable): The torchvision transform to apply.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transformed images tensor and labels tensor.
    """
    images = []
    labels = []
    # Unpack the ((path, inner_label), outer_label) structure
    for item_tuple, _ in batch: # Ignore outer_label, we use the one bundled with path
        img_path, label = item_tuple # Unpack the inner tuple to get path and label
        try:
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
            labels.append(label) # Use the label that was originally paired with the path
        except Exception as e:
            # Log the specific path that failed
            logger.error(f"❌ Error loading/transforming image path='{img_path}' (label={label}) in collate_fn: {e}", exc_info=True)
            continue  # Skip this sample

    if not images:  # If all images in the batch failed
        # Return empty tensors with correct number of dimensions but size 0
        # Assuming RGB images and integer labels
        return torch.empty((0, 3, 224, 224)), torch.empty((0,), dtype=torch.long)

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
        callbacks: Optional[List[Any]] = None,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        patience: int = 5,
        monitor: str = 'valid_loss',
        lr_scheduler: bool = False,
        lr_scheduler_patience: int = 3,
        # Add train_split parameter here if you want it configurable,
        # but for this use case, setting it to None is usually correct.
        # train_split = ValidSplit(5) # Example if you wanted internal split
        *args, **kwargs # Pass other skorch args
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
        default_callbacks: List[Tuple[str, Callback]] = [
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

        final_callbacks = default_callbacks # Simplified for snippet

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
            # --- Explicitly set train_split=None ---
            # This tells skorch *not* to perform its own internal validation split.
            # We handle validation either via sklearn CV or by passing
            # 'validation_data' in fit_params.
            train_split=None,
            # Skorch needs predict_nonlinearity for predict_proba
            predict_nonlinearity='auto',  # 'auto' usually works (softmax for classification)
            *args, **kwargs # Pass other skorch args
        )

    def fit(self, X, y=None, **fit_params):
        """
        Overrides fit to handle path inputs.
        Sklearn CV methods will pass subsets of X (which are paths) here.
        We package X and y into a Skorch Dataset of tuples.
        """
        # --- Input Validation and Processing ---
        if y is None:
            # This case should ideally not happen for training via sklearn CV/fit
            logger.error("❌ SkorchImageClassifier.fit called with y=None during training setup. Labels are required.")
            raise ValueError("Target labels (y) are required during training fit.")

        # Ensure X is a list of paths
        if isinstance(X, np.ndarray):
             X_processed = X.tolist() # Convert numpy array of paths to list
        elif isinstance(X, list):
             X_processed = X # Already a list
        else:
            raise TypeError(f"Input X should be a list or numpy array of paths, got {type(X)}")

        # Ensure y is suitable for skorch checks (numpy array) and for tuples (list/basic types)
        if isinstance(y, torch.Tensor):
            y_for_check = y.numpy() # Numpy array for skorch checks
            y_for_tuples = y.tolist() # List for zipping
        elif isinstance(y, list):
             y_for_check = np.array(y) # Numpy array for skorch checks
             y_for_tuples = y       # Keep as list for zipping
        elif isinstance(y, np.ndarray):
             y_for_check = y       # Already numpy array for skorch checks
             y_for_tuples = y.tolist() # List for zipping
        else:
            raise TypeError(f"Input y should be a list, numpy array, or torch tensor, got {type(y)}")

        # Check length consistency
        if len(X_processed) != len(y_for_check):
             raise ValueError(f"Input X (paths) length {len(X_processed)} and y (labels) length {len(y_for_check)} must match.")

        # Package paths and labels as list of tuples for the Dataset used by collate_fn
        path_label_tuples = list(zip(X_processed, y_for_tuples))

        # Create the dataset where X contains the bundled data for iteration
        # y=None here tells SkorchDataset not to expect a separate y target during iteration
        dataset = SkorchDataset(path_label_tuples, y=None)

        # --- Handle Validation Data ---
        # Skorch handles validation data passed via fit_params['validation_data']
        # We need to ensure it's also packaged correctly if provided
        if 'X_valid' in fit_params and 'y_valid' in fit_params:
            # Get raw validation data
            X_valid_raw = fit_params.pop('X_valid')
            y_valid_raw = fit_params.pop('y_valid')

            # Process validation X (paths)
            if isinstance(X_valid_raw, np.ndarray):
                X_valid_paths = X_valid_raw.tolist()
            elif isinstance(X_valid_raw, list):
                X_valid_paths = X_valid_raw
            else:
                 raise TypeError(f"Input X_valid should be a list or numpy array of paths, got {type(X_valid_raw)}")

            # Process validation y (labels)
            if isinstance(y_valid_raw, torch.Tensor):
                 y_valid_labels = y_valid_raw.tolist()
            elif isinstance(y_valid_raw, np.ndarray):
                 y_valid_labels = y_valid_raw.tolist()
            elif isinstance(y_valid_raw, list):
                 y_valid_labels = y_valid_raw
            else:
                 raise TypeError(f"Input y_valid should be a list, numpy array, or torch tensor, got {type(y_valid_raw)}")

            # Check lengths
            if len(X_valid_paths) != len(y_valid_labels):
                 raise ValueError(f"Input X_valid length {len(X_valid_paths)} and y_valid length {len(y_valid_labels)} must match.")

            # Package validation data into tuples and a SkorchDataset
            valid_path_label_tuples = list(zip(X_valid_paths, y_valid_labels))
            valid_dataset = SkorchDataset(valid_path_label_tuples, y=None)
            # Pass the packaged validation dataset to skorch's fit method
            fit_params['validation_data'] = valid_dataset
            logger.info("Fitting with provided validation data (packaged as SkorchDataset).")
        else:
             logger.info("Fitting without explicit validation data. Skorch may use internal validation split if train_split is not None.")


        # --- Call Parent Fit Method ---
        # Pass our custom dataset (containing tuples) as X.
        # Pass the original labels (y_for_check as numpy array) separately as y.
        # Skorch uses the separate y for internal checks (like class inference)
        # but iterates over the X dataset (our tuples) for training batches.
        logger.debug(f"Calling super().fit with X=SkorchDataset, y=np.array(shape={y_for_check.shape})")
        super().fit(X=dataset, y=y_for_check, **fit_params)

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
