from pathlib import Path
from typing import List, Tuple, Callable, Type, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from skorch.utils import to_numpy
from torch.utils.data import DataLoader, Dataset as PyTorchDataset

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
            show_first_batch_augmentation: bool = False,
            # 'callbacks' now expects None or a list directly from the caller
            callbacks: Optional[List[Tuple[str, Callback]]] = None,
            train_transform: Optional[Callable] = None,
            valid_transform: Optional[Callable] = None,
            train_split: Optional[Callable] = None,
            iterator_train__shuffle: bool = True,
            verbose: int = 1,
            **kwargs
    ):
        self.show_first_batch_augmentation = show_first_batch_augmentation
        self._first_train_batch_shown = False
        self._first_valid_batch_shown = False

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

    # --- Nested class for the plotting iterator ---
    # Defined before get_iterator so it's available
    class _BatchPlottingIterator:
        def __init__(self, base_dataloader: DataLoader, is_train_iterator: bool, adapter_ref: 'SkorchModelAdapter'):
            self.base_dataloader = base_dataloader
            self.base_iter = None  # Will be initialized in __iter__
            self.is_train_iterator = is_train_iterator
            self.adapter_ref = adapter_ref  # Reference to the SkorchModelAdapter instance
            self.is_first_batch_this_epoch = True  # Tracks first batch per epoch iterator

        def __iter__(self):
            self.base_iter = iter(self.base_dataloader)
            self.is_first_batch_this_epoch = True  # Reset for each new epoch's iterator
            return self

        def __next__(self):
            try:
                # Get (X, y) or (X, y, sample_weight) etc. from the base DataLoader
                batch_data = next(self.base_iter)
            except StopIteration:
                raise StopIteration

            # Ensure batch_data[0] is the image tensor
            if not isinstance(batch_data, (tuple, list)) or not len(batch_data) >= 1 or not isinstance(
                    batch_data[0], torch.Tensor):
                logger.warning(
                    f"PlottingIterator: Unexpected batch data format: {type(batch_data)}. Skipping plot.")
                return batch_data  # Return as is

            images_tensor = batch_data[0]

            if self.is_first_batch_this_epoch:
                if self.is_train_iterator and not self.adapter_ref._first_train_batch_shown:
                    self._plot_batch(images_tensor, "First Training Batch (Augmented)")
                    self.adapter_ref._first_train_batch_shown = True
                elif not self.is_train_iterator and not self.adapter_ref._first_valid_batch_shown:
                    self._plot_batch(images_tensor, "First Validation Batch (No Augmentation)")
                    self.adapter_ref._first_valid_batch_shown = True
                self.is_first_batch_this_epoch = False
            return batch_data

        @staticmethod
        def _plot_batch(images_tensor_to_plot: torch.Tensor, title: str):  # Changed param name for clarity
            try:
                from model_src.plotter import ResultsPlotter  # Adjust path as per your structure

                ResultsPlotter.plot_image_batch(
                    images_tensor_to_plot.detach().clone(),  # <<< USE THE PARAMETER NAME
                    title=title,
                    output_path=None,
                    repository_for_plots=None,
                    show_plots=True
                )
            except ImportError:
                logger.error("ResultsPlotter not found (ImportError). Cannot plot debug batch.")
            except Exception as e:
                logger.error(f"Failed to plot debug batch '{title}': {e}", exc_info=True)

        def __len__(self):
            return len(self.base_dataloader)

    def get_iterator(self, dataset, training=False):
        actual_dataset_to_use: PyTorchDataset

        current_transform = self.train_transform if training else self.valid_transform

        if isinstance(dataset, PathImageDataset):
            if dataset.transform != current_transform:
                logger.debug(f"get_iterator: Updating transform for existing PathImageDataset (training={training}).")
                actual_dataset_to_use = PathImageDataset(paths=dataset.paths, labels=dataset.labels,
                                                         transform=current_transform)
            else:
                actual_dataset_to_use = dataset
        elif hasattr(dataset, 'X') and hasattr(dataset, 'y') and \
             not isinstance(dataset.X, torch.Tensor) and \
             (isinstance(dataset.X, (list, tuple, np.ndarray)) and
              (len(dataset.X) == 0 or isinstance(dataset.X[0], (str, Path, Image.Image, bytes)))):
            X_data, y_data = dataset.X, dataset.y
            # This assumes X_data will be paths for PathImageDataset.
            # If predict_images passes InMemoryPILDataset directly, this branch might not be hit,
            # or if InMemoryPILDataset is wrapped by skorch Dataset, X_data would be PIL images.
            if isinstance(X_data, (list, tuple, np.ndarray)) and \
                    (len(X_data) == 0 or isinstance(X_data[0], (str, Path))):  # Check if X_data are paths

                paths_for_ds = [Path(p) for p in X_data]
                labels_for_ds = list(y_data) if y_data is not None else None
                logger.debug(
                    f"get_iterator: Creating PathImageDataset from X (paths), y for {'training' if training else 'eval'}.")
                actual_dataset_to_use = PathImageDataset(paths=paths_for_ds, labels=labels_for_ds,
                                                         transform=current_transform)
            else:  # X_data is not paths (e.g., already tensors, or PIL images list for InMemoryPILDataset)
                # or it's an unknown type.
                if isinstance(dataset, TorchDataset):  # If it's already a PyTorch Dataset
                    logger.debug(f"get_iterator: Using provided PyTorch Dataset {type(dataset).__name__} as is.")
                    actual_dataset_to_use = dataset  # Assume it has correct transform or handles it
                else:
                    logger.warning(
                        f"get_iterator: X in dataset is not paths. Falling back to super().get_iterator(). Type of X: {type(X_data[0]) if X_data else 'Empty'}")
                    # Fallback to skorch's default iterator creation if dataset.X is not paths.
                    # This might happen if skorch is passed already-loaded tensors.
                    # Our plotting wrapper won't apply if super().get_iterator() is used unless super calls this again.
                    return super().get_iterator(dataset, training=training)
        elif isinstance(dataset, TorchDataset):  # It's some other PyTorch dataset (e.g. InMemoryPILDataset)
            logger.debug(f"get_iterator: Using provided PyTorch Dataset {type(dataset).__name__} directly.")
            actual_dataset_to_use = dataset
        else:
            logger.error(f"get_iterator: Unsupported dataset type: {type(dataset)}. Fallback to super().")
            return super().get_iterator(dataset, training=training)

        # --- DataLoader Configuration ---
        collate_fn_to_use = getattr(actual_dataset_to_use, 'collate_fn', None)
        if collate_fn_to_use is None:
            if hasattr(PathImageDataset, 'collate_fn'):  # Good default if dataset has no specific one
                collate_fn_to_use = PathImageDataset.collate_fn
                logger.debug(f"Using PathImageDataset.collate_fn for {type(actual_dataset_to_use).__name__}.")
            else:  # Absolute fallback
                collate_fn_to_use = torch.utils.data.dataloader.default_collate
                logger.warning("Using torch default collate_fn.")

        shuffle = self.iterator_train__shuffle if training else False
        batch_size_to_use = self.batch_size

        loader_kwargs = {
            'num_workers': getattr(self, 'iterator__num_workers', 0),
            'pin_memory': getattr(self, 'iterator__pin_memory', False)
        }
        # Ensure num_workers is 0 if on Windows and not in __main__ for multiprocessing safety with DataLoader
        import sys
        if sys.platform == "win32" and loader_kwargs['num_workers'] > 0:
            # This check is tricky for library code. Best to set num_workers=0 by default.
            # logger.debug("Setting num_workers to 0 for Windows platform in DataLoader.")
            # loader_kwargs['num_workers'] = 0
            pass  # Assume user has configured NUM_WORKERS appropriately at pipeline level

        logger.debug(
            f"Creating DataLoader for {type(actual_dataset_to_use).__name__}: "
            f"size={len(actual_dataset_to_use) if hasattr(actual_dataset_to_use, '__len__') else 'unknown'}, "
            f"batch_size={batch_size_to_use}, shuffle={shuffle}, "
            f"collate_fn_type='{getattr(collate_fn_to_use, '__name__', str(type(collate_fn_to_use)))}', other_kwargs={loader_kwargs}"
        )

        base_dataloader = DataLoader(
            actual_dataset_to_use,
            batch_size=batch_size_to_use,
            shuffle=shuffle,
            collate_fn=collate_fn_to_use,
            **loader_kwargs
        )

        # --- Optionally wrap the DataLoader for plotting ---
        if self.show_first_batch_augmentation and hasattr(self, '_BatchPlottingIterator'):
            logger.debug(
                f"Wrapping DataLoader with _BatchPlottingIterator for {'training' if training else 'validation'}.")
            return self._BatchPlottingIterator(base_dataloader, is_train_iterator=training, adapter_ref=self)
        else:
            return base_dataloader

    # --- train_step_single / validation_step handle batches from PathImageDataset ---
    def train_step_single(self, batch, **fit_params):
        self.module_.train()
        Xi, yi = batch  # Already transformed tensors
        # TODO: add optional visualisation of the batch
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
