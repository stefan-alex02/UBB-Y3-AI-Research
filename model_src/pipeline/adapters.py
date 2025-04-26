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
            *args, **kwargs  # Pass other skorch args
    ):
        if train_transform is None or val_transform is None:
            raise ValueError("Both train_transform and val_transform must be provided.")

        self.train_transform = train_transform
        self.val_transform = val_transform

        # Define Collate Functions
        def train_collate_fn(batch):
            # Pass the updated load_and_transform function
            return load_and_transform(batch, self.train_transform)

        def val_collate_fn(batch):
            # Pass the updated load_and_transform function
            return load_and_transform(batch, self.val_transform)

        # Setup Callbacks (ensure this logic correctly handles merging)
        default_callbacks: List[Tuple[str, Callback]] = [
            ('early_stopping', EarlyStopping(monitor=monitor, patience=patience, lower_is_better=True, load_best=True)),
        ]
        if lr_scheduler:
            default_callbacks.append(
                ('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', patience=lr_scheduler_patience, factor=0.1,
                                             monitor='valid_loss'))
            )

        final_callbacks = default_callbacks  # Start with defaults
        if callbacks is not None:
            callback_dict = {name: cb for name, cb in default_callbacks}
            user_callbacks_processed = []  # Keep track of user callbacks by name/type
            for cb in callbacks:
                name = None
                callback_instance = None
                if isinstance(cb, tuple):
                    name, callback_instance = cb[0], cb[1]
                else:
                    name = cb.__class__.__name__  # Generate name
                    callback_instance = cb

                if name in callback_dict:
                    logger.warning(f"⚠️ Overriding default callback '{name}' with user-provided callback.")
                elif name in [c[0] for c in user_callbacks_processed] or name in [c[1].__class__.__name__ for c in
                                                                                  user_callbacks_processed if
                                                                                  not isinstance(c, tuple)]:
                    logger.warning(f"⚠️ Duplicate user callback '{name}' provided. Using the last one.")

                callback_dict[name] = callback_instance
                user_callbacks_processed.append((name, callback_instance))  # Track processed user callbacks

            final_callbacks = list(callback_dict.items())  # Use the merged dictionary

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
            iterator_train__collate_fn=train_collate_fn,
            iterator_valid__collate_fn=val_collate_fn,
            train_split=None,  # Keep this as None
            predict_nonlinearity='auto',
            *args, **kwargs
        )

    def fit(self, X, y=None, **fit_params):
        """
        Overrides fit to handle path inputs and correctly pass validation data.
        """
        # --- Input Validation and Processing (Train Data) ---
        # (Keep the processing for X and y -> X_processed, y_for_check, y_for_tuples)
        if y is None:
            logger.error("❌ SkorchImageClassifier.fit called with y=None during training setup. Labels are required.")
            raise ValueError("Target labels (y) are required during training fit.")

        if isinstance(X, np.ndarray): X_processed = X.tolist()
        elif isinstance(X, list): X_processed = X
        else: raise TypeError(f"Input X should be list/array of paths, got {type(X)}")

        if isinstance(y, torch.Tensor): y_for_check, y_for_tuples = y.cpu().numpy(), y.tolist() # Ensure numpy conversion happens on CPU
        elif isinstance(y, list): y_for_check, y_for_tuples = np.array(y), y
        elif isinstance(y, np.ndarray): y_for_check, y_for_tuples = y, y.tolist()
        else: raise TypeError(f"Input y should be list/array/tensor, got {type(y)}")

        if len(X_processed) != len(y_for_check): raise ValueError(f"Input X length {len(X_processed)} and y length {len(y_for_check)} must match.")

        # Package TRAINING paths and labels for the Dataset used by collate_fn
        path_label_tuples = list(zip(X_processed, y_for_tuples))
        dataset = SkorchDataset(path_label_tuples, y=None) # Training dataset

        # --- Handle Validation Data ---
        X_valid_processed = None
        y_valid_processed = None # This should be the numpy array for skorch internal checks
        if 'X_valid' in fit_params and 'y_valid' in fit_params:
            # Get raw validation data from fit_params (as passed by run_single_train)
            X_valid_raw = fit_params.pop('X_valid') # Pop them from fit_params
            y_valid_raw = fit_params.pop('y_valid')

            # Process validation X (paths) -> Should be List[str]
            if isinstance(X_valid_raw, np.ndarray): X_valid_paths = X_valid_raw.tolist()
            elif isinstance(X_valid_raw, list): X_valid_paths = X_valid_raw
            else: raise TypeError(f"Input X_valid should be list/array of paths, got {type(X_valid_raw)}")

            # Process validation y -> Should be np.ndarray for skorch checks
            if isinstance(y_valid_raw, torch.Tensor): y_valid_processed = y_valid_raw.cpu().numpy()
            elif isinstance(y_valid_raw, list): y_valid_processed = np.array(y_valid_raw)
            elif isinstance(y_valid_raw, np.ndarray): y_valid_processed = y_valid_raw
            else: raise TypeError(f"Input y_valid should be list/array/tensor, got {type(y_valid_raw)}")

            if len(X_valid_paths) != len(y_valid_processed): raise ValueError(f"X_valid length {len(X_valid_paths)} and y_valid length {len(y_valid_processed)} must match.")

            # --- IMPORTANT: Package validation paths/labels for the Validation *Dataset* ---
            # Skorch needs the validation data packaged correctly for its iterator too.
            # We pass the *raw* paths and labels to fit, but must also ensure
            # that skorch uses our path-loading mechanism for validation batches.
            # Option 1 (Implicit via collate_fn): If iterator_valid__collate_fn is set, skorch
            # might automatically use it when creating the internal validation dataloader from
            # the X_valid (paths) and y_valid (labels) we pass below. This is the hope.
            # Option 2 (Explicit): We could potentially create a validation SkorchDataset here
            # and pass it, but the API expects X_valid, y_valid directly. Let's rely on Option 1.

            # Set X_valid_processed to the list of paths for the super().fit call
            X_valid_processed = X_valid_paths
            logger.info("Passing validation data (paths and labels) directly to super().fit.")

        else:
            logger.info("Fitting without explicit validation data.")

        # --- Call Parent Fit Method ---
        logger.debug(f"Calling super().fit with X=SkorchDataset (train tuples), y=np.array(shape={y_for_check.shape})")
        # Pass the training dataset as X, training labels as y.
        # Pass the *processed* validation paths as X_valid and validation labels as y_valid.
        super().fit(
            X=dataset, # Training data (tuples for collate_fn)
            y=y_for_check, # Training labels (numpy for skorch checks)
            X_valid=X_valid_processed, # Validation paths (list) - skorch will build internal dataset
            y_valid=y_valid_processed, # Validation labels (numpy)
            **fit_params # Pass any remaining fit_params
        )

        return self

    def infer(self, x, **fit_params):
        """
        Override infer to manually remove keys related to validation data
        ('X_valid', 'y_valid') from fit_params before passing them to the
        underlying module via super().infer().
        This prevents TypeError in the module's forward method when external
        validation data is provided to the fit method.
        """
        # Create a copy of fit_params.
        filtered_fit_params = fit_params.copy()

        # Remove the standard skorch keys for validation data if they exist.
        # These keys are used by skorch internally but should not be passed
        # to the module's forward method.
        filtered_fit_params.pop('X_valid', None)
        filtered_fit_params.pop('y_valid', None)
        # Also remove the old key just in case (belt and suspenders)
        filtered_fit_params.pop('validation_data', None)

        # Call the original infer method with the filtered parameters.
        return super().infer(x, **filtered_fit_params)

    # --- get_params and set_params (should be okay as they were) ---
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
