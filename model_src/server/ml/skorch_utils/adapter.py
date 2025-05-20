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
                from ..plotter import ResultsPlotter  # Adjust path as per your structure

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
        """
        Override to ensure PathImageDataset with correct transform is used,
        and DataLoader is configured correctly with batch_size and collate_fn.
        Optionally wraps the iterator to plot the first batch for debugging.
        """
        current_dataset: Optional[torch.utils.data.Dataset] = None

        if isinstance(dataset, PathImageDataset):
            current_dataset = dataset
            # Ensure the transform is correct for the phase (training vs eval)
            # PathImageDataset created by get_split_datasets already has the correct transform.
            # This is more for when skorch calls get_iterator with a dataset not from get_split_datasets.
            expected_transform = self.train_transform if training else self.valid_transform
            if current_dataset.transform != expected_transform:
                logger.debug(
                    f"get_iterator: Updating transform on existing PathImageDataset for {'training' if training else 'eval'}.")
                current_dataset.transform = expected_transform

        elif hasattr(dataset, 'X'):  # Likely a skorch.dataset.Dataset from predict/score
            X_input = dataset.X
            y_labels = getattr(dataset, 'y', None)  # y might not exist or be None

            # Check if X_input are paths
            is_path_data = False
            if isinstance(X_input, (list, tuple, np.ndarray)) and len(X_input) > 0:
                # Check first element to infer type
                first_el = X_input[0]
                if isinstance(X_input, np.ndarray) and X_input.ndim > 1 and X_input.shape[
                    0] > 0:  # Handle multi-dim np array by checking first row, first element
                    first_el = X_input[
                        0, 0] if X_input.size > 0 else None  # Needs more robust check if X can be complex
                if isinstance(first_el, (str, Path)):
                    is_path_data = True

            if is_path_data:
                X_paths_list = list(X_input) if not isinstance(X_input, np.ndarray) else X_input.tolist()
                y_labels_list = list(y_labels) if y_labels is not None and not isinstance(y_labels, np.ndarray) else (
                    y_labels.tolist() if y_labels is not None else None)

                transform_to_use = self.train_transform if training else self.valid_transform
                logger.debug(
                    f"get_iterator: Wrapping skorch Dataset's X (paths) with PathImageDataset for {'training' if training else 'eval'}.")
                current_dataset = PathImageDataset(paths=X_paths_list, labels=y_labels_list, transform=transform_to_use)
            else:
                # If X is not path data, assume `dataset` is already a compatible torch.utils.data.Dataset
                # (e.g., if user passed a TensorDataset or your InMemoryPILDataset for predict_images)
                if isinstance(dataset, torch.utils.data.Dataset):
                    logger.debug(f"get_iterator: Using provided Dataset of type {type(dataset)} directly.")
                    current_dataset = dataset
                else:
                    logger.warning(
                        f"get_iterator received skorch Dataset with non-path X data. Falling back to super().get_iterator().")
                    base_iterator = super().get_iterator(dataset, training=training)
                    if self.show_first_batch_augmentation:
                        return self._BatchPlottingIterator(base_iterator, is_train_iterator=training, adapter_ref=self)
                    return base_iterator
        elif isinstance(dataset, torch.utils.data.Dataset):  # It's already a torch Dataset
            current_dataset = dataset
        else:
            logger.warning(
                f"get_iterator received unexpected dataset type {type(dataset)}. Falling back to super().get_iterator().")
            base_iterator = super().get_iterator(dataset, training=training)
            if self.show_first_batch_augmentation:
                return self._BatchPlottingIterator(base_iterator, is_train_iterator=training, adapter_ref=self)
            return base_iterator

        if current_dataset is None:  # Should have been handled by fallbacks above
            raise RuntimeError("get_iterator: Could not determine a valid torch Dataset.")

        # --- DataLoader Configuration using current_dataset ---
        # Always use PathImageDataset.collate_fn if the dataset is one of ours or compatible
        collate_fn_to_use = PathImageDataset.collate_fn
        if hasattr(current_dataset,
                   'collate_fn') and current_dataset.collate_fn is not None:  # If dataset has its own specific collate
            collate_fn_to_use = current_dataset.collate_fn
        elif not isinstance(current_dataset, PathImageDataset) and not hasattr(current_dataset, 'paths'):
            # If it's some other torch.utils.data.Dataset that isn't our PathImageDataset
            # and doesn't have a collate_fn, use default torch collate.
            collate_fn_to_use = torch.utils.data.dataloader.default_collate
            logger.debug(f"Using default torch collate_fn for dataset type: {type(current_dataset)}")

        shuffle = self.iterator_train__shuffle if training else False
        batch_size = self.batch_size

        loader_kwargs = {}
        iterator_params = self.get_params()  # Skorch way to get settable params
        num_workers_key = 'iterator_train__num_workers' if training else 'iterator_valid__num_workers'
        pin_memory_key = 'iterator_train__pin_memory' if training else 'iterator_valid__pin_memory'

        # Use configured num_workers if available, else try direct attribute, else default to 0
        loader_kwargs['num_workers'] = iterator_params.get(num_workers_key, getattr(self, 'iterator__num_workers', 0))
        loader_kwargs['pin_memory'] = iterator_params.get(pin_memory_key, getattr(self, 'iterator__pin_memory', False))
        loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

        logger.debug(
            f"Creating DataLoader: dataset_len={len(current_dataset)}, batch_size={batch_size}, shuffle={shuffle}, "
            f"collate_fn={'Custom (PathImageDataset style)' if collate_fn_to_use == PathImageDataset.collate_fn else ('Dataset Specific' if hasattr(current_dataset, 'collate_fn') else 'Torch Default')}, "
            f"loader_kwargs={loader_kwargs}"
        )

        actual_dataloader = DataLoader(
            current_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_to_use,
            **loader_kwargs
        )

        if self.show_first_batch_augmentation:
            logger.debug(
                f"Wrapping DataLoader with _BatchPlottingIterator for {'training' if training else 'validation/evaluation'}.")
            return self._BatchPlottingIterator(
                actual_dataloader,
                is_train_iterator=training,
                adapter_ref=self
            )

        return actual_dataloader

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
