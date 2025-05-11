from typing import List, Tuple, Callable, Type, Optional

import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from skorch.utils import to_numpy
from torch.utils.data import DataLoader

from ..config import DEVICE
from ..dataset_utils import PathImageDataset
from ..logger_utils import logger


class SkorchModelAdapter(NeuralNetClassifier):
    """
    Skorch adapter using PathImageDataset.
    Uses overridden get_split_datasets and get_iterator to ensure
    correct train/eval transforms are applied during fit and predict/eval.
    Includes train_acc logging.
    """

    def __init__(
            self,
            *args,
            module: Optional[Type[nn.Module]] = None,
            criterion: Type[nn.Module] = nn.CrossEntropyLoss,
            optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
            lr: float = 0.001,
            max_epochs: int = 20,
            batch_size: int = 32,
            device: str = DEVICE,
            # 'callbacks' now expects None or a list directly from the caller
            callbacks: Optional[List[Tuple[str, Callback]]] = None,
            train_transform: Optional[Callable] = None,
            valid_transform: Optional[Callable] = None,
            train_split: Optional[Callable] = None,
            iterator_train__shuffle: bool = True,
            verbose: int = 1,
            **kwargs
    ):
        if train_transform is None or valid_transform is None:
            raise ValueError("Both train_transform and valid_transform must be provided")

        # Store transforms directly
        self.train_transform = train_transform
        self.valid_transform = valid_transform

        # Callback config args (patience etc.) should have been handled by the caller
        # We no longer process 'default' here.

        # Add Collate Functions to kwargs if not provided by caller
        kwargs.setdefault('iterator_train__collate_fn', PathImageDataset.collate_fn)
        kwargs.setdefault('iterator_valid__collate_fn', PathImageDataset.collate_fn)

        # Initialize the parent class, passing callbacks list/None directly
        super().__init__(
            *args, module=module, criterion=criterion, optimizer=optimizer, lr=lr,
            max_epochs=max_epochs, batch_size=batch_size, device=device,
            callbacks=callbacks,  # Pass the provided list/None
            train_split=train_split, iterator_train__shuffle=iterator_train__shuffle,
            verbose=verbose, **kwargs
        )

    # --- Override get_split_datasets ---
    def get_split_datasets(self, X, y=None, **fit_params):
        """
        Splits paths/labels using self.train_split based on indices and y,
        then creates separate PathImageDatasets with appropriate train/valid transforms.
        """
        # Ensure y is available and numpy array
        if y is None: raise ValueError("y must be provided to fit when using train_split.")
        y_arr = to_numpy(y)

        # Ensure X is paths and get length
        if not isinstance(X, (list, tuple, np.ndarray)): raise TypeError(f"X must be sequence, got {type(X)}")
        if isinstance(X, np.ndarray) and X.ndim > 1: raise ValueError("X must be 1D sequence of paths")
        X_len = len(X)
        if X_len == 0: logger.warning("Input X is empty."); return None, None
        X_paths_np = np.asarray(X)  # Keep original paths safe

        # 1. Check if a train_split strategy is defined
        if self.train_split:
            try:
                # --- MODIFICATION ---
                # Pass indices array (representing X) and the actual y array to the splitter
                # ValidSplit(cv=float, stratified=True) uses train_test_split which works with indices.
                # ValidSplit(cv=KFold, stratified=?) KFold split works on indices.
                indices = np.arange(X_len)
                ds_train_split, ds_valid_split = self.train_split(indices, y=y_arr, **fit_params)
                # --- END MODIFICATION ---

                # Extract indices from the returned split datasets
                if hasattr(ds_train_split, 'indices'):
                    train_indices = ds_train_split.indices
                else:
                    raise TypeError(f"Could not extract indices from train split result type {type(ds_train_split)}")
                train_indices = np.asarray(train_indices)

                valid_indices = None
                if ds_valid_split is not None and len(ds_valid_split) > 0:
                    if hasattr(ds_valid_split, 'indices'):
                        valid_indices = ds_valid_split.indices
                        valid_indices = np.asarray(valid_indices)
                    else:
                        raise TypeError(
                            f"Could not extract indices from valid split result type {type(ds_valid_split)}")

                # Create datasets using indices on original paths/labels AND correct transforms
                ds_train = PathImageDataset(
                    paths=X_paths_np[train_indices].tolist(),
                    labels=y_arr[train_indices].tolist(),
                    transform=self.train_transform
                )

                ds_valid = None
                if len(valid_indices) > 0:
                    ds_valid = PathImageDataset(
                        paths=X_paths_np[valid_indices].tolist(),
                        labels=y_arr[valid_indices].tolist(),
                        transform=self.valid_transform
                    )
                    logger.debug(f"Split created: {len(ds_train)} train, {len(ds_valid)} validation.")
                else:
                    logger.debug(f"Split created: {len(ds_train)} train, 0 validation.")

                return ds_train, ds_valid

            except Exception as e:
                logger.error(f"Error applying train_split in get_split_datasets: {e}", exc_info=True)
                logger.warning("Falling back to using all data for training.")
                ds_train = PathImageDataset(X_paths_np.tolist(), y_arr.tolist(), transform=self.train_transform)
                return ds_train, None
        else:
            # No train_split defined
            logger.debug(f"No train_split defined. Using all {X_len} samples for training.")
            ds_train = PathImageDataset(X_paths_np.tolist(), y_arr.tolist(), transform=self.train_transform)
            return ds_train, None

    def get_iterator(self, dataset, training=False):
        """
        Override to ensure PathImageDataset with correct transform is used,
        and DataLoader is configured correctly with batch_size and collate_fn.
        """
        # Ensure 'dataset' is PathImageDataset
        if not isinstance(dataset, PathImageDataset):
            if hasattr(dataset, 'X') and hasattr(dataset, 'y'):
                X_paths = dataset.X
                y_labels = dataset.y
                if isinstance(X_paths, np.ndarray): X_paths = X_paths.tolist()
                if isinstance(y_labels, np.ndarray): y_labels = y_labels.tolist()
                transform = self.train_transform if training else self.valid_transform
                logger.debug(
                    f"get_iterator creating PathImageDataset for {'training' if training else 'evaluation/prediction'}.")
                dataset = PathImageDataset(X_paths, y_labels, transform=transform)
            else:
                logger.warning(f"get_iterator received unexpected dataset type {type(dataset)}, fallback to super.")
                return super().get_iterator(dataset, training=training)

        # --- Refined DataLoader Configuration ---
        collate_fn = getattr(dataset, 'collate_fn', None)
        if collate_fn is None:
            logger.warning("PathImageDataset instance missing collate_fn attribute.")
            # Optionally fall back to default collate, but might fail on None items
            collate_fn = torch.utils.data.dataloader.default_collate

        # Get relevant iterator parameters directly from self
        # Use skorch's convention for parameter naming
        shuffle = self.iterator_train__shuffle if training else False  # Only shuffle train iterator
        batch_size = self.batch_size  # Use the main batch_size parameter

        # Get other potential DataLoader args like num_workers, pin_memory if set via kwargs
        loader_kwargs = {}
        if hasattr(self, 'iterator__num_workers'):
            loader_kwargs['num_workers'] = self.iterator__num_workers
        if hasattr(self, 'iterator__pin_memory'):
            loader_kwargs['pin_memory'] = self.iterator__pin_memory
        # Add any other relevant DataLoader args you might configure via skorch kwargs

        logger.debug(
            f"Creating DataLoader: size={len(dataset)}, batch_size={batch_size}, shuffle={shuffle}, collate_fn={'Assigned' if collate_fn else 'None'}, other_kwargs={loader_kwargs}")

        return DataLoader(
            dataset,
            batch_size=batch_size,  # Pass explicitly
            shuffle=shuffle,
            collate_fn=collate_fn,
            **loader_kwargs
        )

    # --- train_step_single / validation_step handle batches from PathImageDataset ---
    def train_step_single(self, batch, **fit_params):
        self.module_.train()
        Xi, yi = batch  # Already transformed tensors
        yi = yi.to(dtype=torch.long)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {'loss': loss, 'y_pred': y_pred}  # y_pred might be needed by scoring callback implicitly

    def validation_step(self, batch, **fit_params):
        self.module_.eval()
        Xi, yi = batch # Already transformed tensors from PathImageDataset
        yi = yi.to(dtype=torch.long)
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        # Return y_pred so skorch can calculate valid_acc etc.
        return {'loss': loss, 'y_pred': y_pred}
