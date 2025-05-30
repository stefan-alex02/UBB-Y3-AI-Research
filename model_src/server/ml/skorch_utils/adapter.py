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
from .augmentations_utils import cutmix_data, rand_bbox


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
            cutmix_alpha: float = 0.0,       # Corresponds to 'beta' in Kaggle script; Alpha for Beta distribution. If 0, CutMix is disabled.
            cutmix_probability: float = 0.0, # Probability of applying CutMix per batch. If 0, CutMix is disabled.
            gradient_clip_value: Optional[float] = None,  # For gradient clipping
            verbose: int = 1,
            **kwargs
    ):
        self.show_first_batch_augmentation = show_first_batch_augmentation
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_probability = cutmix_probability
        self.gradient_clip_value = gradient_clip_value

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

        if self.gradient_clip_value is not None and self.gradient_clip_value > 0:
            self.gradient_clip_fn_ = lambda iterator: torch.nn.utils.clip_grad_norm_( # Corrected: was self.module_.parameters()
                self.module_.parameters(), max_norm=self.gradient_clip_value
            )
        else:
            self.gradient_clip_fn_ = None

    def initialize(self):
        super().initialize()
        # Flags to ensure each type of plot happens only once per fit() call
        self._first_original_train_batch_plotted_this_fit = False
        self._first_cutmixed_train_batch_plotted_this_fit = False
        self._first_validation_batch_plotted_this_fit = False
        # logger.debug("SkorchModelAdapter plotting flags reset for new fit.")
        return self

    @staticmethod
    def _plot_debug_batch(images_tensor_to_plot: torch.Tensor, title: str):
        try:
            from ..plotter import ResultsPlotter
            # logger.info(f"Plotting debug batch: {title}")
            ResultsPlotter.plot_image_batch(
                images_tensor_to_plot.detach().clone().cpu(),
                title=title, show_plots=True
            )
        except Exception as e:
            logger.error(f"Error in _plot_debug_batch: {e}", exc_info=True)

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
        and DataLoader is configured correctly.
        Plotting of the first batch is now handled within train_step (for training)
        and can be added to validation_step (for validation) if needed.
        """
        current_dataset: Optional[torch.utils.data.Dataset] = None

        if isinstance(dataset, PathImageDataset):
            current_dataset = dataset
            expected_transform = self.train_transform if training else self.valid_transform
            if current_dataset.transform != expected_transform:  # Should ideally not happen if get_split_datasets is used
                logger.debug(
                    f"get_iterator: Updating transform on existing PathImageDataset for {'training' if training else 'eval'}.")
                current_dataset.transform = expected_transform

        elif hasattr(dataset, 'X'):  # Likely a skorch.dataset.Dataset from predict/score calls
            X_input = dataset.X
            y_labels = getattr(dataset, 'y', None)

            is_path_data = False
            if isinstance(X_input, (list, tuple, np.ndarray)) and len(X_input) > 0:
                first_el = X_input[0]
                if isinstance(X_input, np.ndarray) and X_input.ndim > 1 and X_input.shape[0] > 0:
                    first_el = X_input[0, 0] if X_input.size > 0 else None
                if isinstance(first_el, (str, Path)):
                    is_path_data = True

            if is_path_data:
                X_paths_list = list(X_input) if not isinstance(X_input, np.ndarray) else X_input.tolist()
                y_labels_list = list(y_labels) if y_labels is not None and not isinstance(y_labels,
                                                                                          np.ndarray) else (
                    y_labels.tolist() if y_labels is not None else None)
                transform_to_use = self.train_transform if training else self.valid_transform
                current_dataset = PathImageDataset(paths=X_paths_list, labels=y_labels_list,
                                                   transform=transform_to_use)
            elif isinstance(dataset,
                            torch.utils.data.Dataset):  # X is not path data, but dataset is already a torch Dataset
                current_dataset = dataset
            else:  # Fallback to super for unknown skorch.dataset.Dataset content
                logger.warning(
                    f"get_iterator received skorch Dataset with non-path X data of type {type(X_input)}. "
                    f"Falling back to super().get_iterator(). First batch plotting relies on train/validation_step.")
                return super().get_iterator(dataset, training=training)

        elif isinstance(dataset, torch.utils.data.Dataset):  # Already a torch Dataset passed directly
            current_dataset = dataset
        else:  # Truly unexpected dataset type
            logger.warning(
                f"get_iterator received unexpected dataset type {type(dataset)}. "
                f"Falling back to super().get_iterator(). First batch plotting relies on train/validation_step.")
            return super().get_iterator(dataset, training=training)

        if current_dataset is None:
            # This case should ideally be unreachable if logic above is complete
            raise RuntimeError("get_iterator: Could not determine or create a valid torch.utils.data.Dataset.")

        # --- DataLoader Configuration ---
        collate_fn_to_use = PathImageDataset.collate_fn  # Default for our PathImageDataset
        if not isinstance(current_dataset, PathImageDataset) and not hasattr(current_dataset, 'paths'):
            # If it's some other torch.utils.data.Dataset that isn't our PathImageDataset
            # (e.g., a TensorDataset, or user-provided custom Dataset for predict_images)
            # and it doesn't have a `collate_fn` attribute, use default torch collate.
            if hasattr(current_dataset, 'collate_fn') and current_dataset.collate_fn is not None:
                collate_fn_to_use = current_dataset.collate_fn
            else:
                collate_fn_to_use = torch.utils.data.dataloader.default_collate
                logger.debug(f"Using default torch collate_fn for dataset type: {type(current_dataset)}")

        shuffle = self.iterator_train__shuffle if training else False
        # batch_size is a direct attribute from __init__ or set_params

        loader_kwargs = {}
        # get_params() retrieves all skorch-settable parameters
        iterator_params = self.get_params()

        # Keys for num_workers and pin_memory can be specific to train/valid iterators in skorch
        num_workers_key = 'iterator_train__num_workers' if training else 'iterator_valid__num_workers'
        pin_memory_key = 'iterator_train__pin_memory' if training else 'iterator_valid__pin_memory'

        # Get num_workers, falling back to a generic iterator__num_workers if specific isn't set, then 0
        loader_kwargs['num_workers'] = iterator_params.get(
            num_workers_key,
            iterator_params.get('iterator__num_workers', 0)  # Check for generic skorch param
        )
        # Get pin_memory, similar fallback
        loader_kwargs['pin_memory'] = iterator_params.get(
            pin_memory_key,
            iterator_params.get('iterator__pin_memory', False)  # Check for generic skorch param
        )
        # Remove None values, as DataLoader doesn't like num_workers=None
        loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

        logger.debug(
            f"Creating DataLoader for {'training' if training else 'validation/evaluation'}: "
            f"dataset_len={len(current_dataset)}, batch_size={self.batch_size}, shuffle={shuffle}, "
            f"collate_fn_type='{type(collate_fn_to_use).__name__}', "
            f"loader_kwargs={loader_kwargs}"
        )

        return DataLoader(
            current_dataset,
            batch_size=self.batch_size,  # Use self.batch_size directly
            shuffle=shuffle,
            collate_fn=collate_fn_to_use,
            **loader_kwargs
        )

    def train_step(self, batch, **fit_params):
        self.module_.train()
        Xi_original, yi_original = batch  # Keep original batch for potential plotting

        # Move to device
        Xi = Xi_original.to(self.device)
        yi = yi_original.to(self.device)
        yi = yi.to(dtype=torch.long)

        inputs_for_model = Xi  # This will hold the batch actually fed to self.infer()
        is_cutmix_applied_this_batch = False
        loss_lambda_for_plot = 1.0

        # Plot original first training batch if enabled and not yet plotted
        if self.show_first_batch_augmentation and not self._first_original_train_batch_plotted_this_fit:
            # Use Xi_original which is on CPU from DataLoader before explicit move
            # Or, if you always move then plot, use Xi.cpu()
            SkorchModelAdapter._plot_debug_batch(Xi_original, "First Original Training Batch (from DataLoader)")
            self._first_original_train_batch_plotted_this_fit = True

        # Apply CutMix
        if self.module_.training and self.cutmix_alpha > 0 and torch.rand(1).item() < self.cutmix_probability:
            inputs_cutmix, targets_a, targets_b, lam = cutmix_data(
                Xi, yi, alpha=self.cutmix_alpha, device=self.device
            )
            inputs_for_model = inputs_cutmix
            is_cutmix_applied_this_batch = True
            loss_lambda_for_plot = lam

            y_pred = self.infer(inputs_for_model, **fit_params)
            loss = loss_lambda_for_plot * self.get_loss(y_pred, targets_a, X=inputs_for_model, training=True) + \
                   (1 - loss_lambda_for_plot) * self.get_loss(y_pred, targets_b, X=inputs_for_model, training=True)
        else:
            y_pred = self.infer(inputs_for_model, **fit_params)  # inputs_for_model is original Xi here
            loss = self.get_loss(y_pred, yi, X=inputs_for_model, training=True)

        # Plot first CutMixed batch if it was applied and not yet plotted (and original wasn't cutmixed)
        if self.show_first_batch_augmentation and \
                is_cutmix_applied_this_batch and \
                not self._first_cutmixed_train_batch_plotted_this_fit:
            # If the very first batch was cutmixed, _first_original_train_batch_plotted_this_fit is True.
            # We only want to plot cutmix separately if the *original* plot didn't already show a cutmixed version.
            # This means if the first batch was cutmixed, the "original" plot will actually show the cutmixed one.
            # Let's refine:
            # The "original" plot always shows the batch *before* cutmix decision.
            # The "cutmixed" plot *only* shows if cutmix was applied AND it's the first time cutmix was applied.

            SkorchModelAdapter._plot_debug_batch(inputs_for_model,  # This is inputs_cutmix
                                                 f"First CutMixed Training Batch (lam~{loss_lambda_for_plot:.2f})")
            self._first_cutmixed_train_batch_plotted_this_fit = True

        self.optimizer_.zero_grad()
        loss.backward()
        if self.gradient_clip_fn_:
            self.gradient_clip_fn_(self.module_.parameters())
        self.optimizer_.step()

        return {'loss': loss, 'y_pred': y_pred}

    def validation_step(self, batch, **fit_params):
        self.module_.eval()
        Xi_original_val, yi_original_val = batch  # Keep original

        Xi = Xi_original_val.to(self.device)
        yi = yi_original_val.to(self.device)
        yi = yi.to(dtype=torch.long)

        if self.show_first_batch_augmentation and not self._first_validation_batch_plotted_this_fit:
            SkorchModelAdapter._plot_debug_batch(Xi_original_val, "First Validation Batch (Clean)")
            self._first_validation_batch_plotted_this_fit = True

        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {'loss': loss, 'y_pred': y_pred}

    # --- Add dummy method for the custom notification ---
    def on_train_batch_transformed(self, net, **kwargs):
        """
        Dummy method to satisfy skorch's notify trying to call this on the net itself.
        The actual logic is in the FirstBatchPlotterCallback.
        'net' will be self here.
        """
        pass # Does nothing

    def notify(self, method_name: str, **cb_kwargs):
        # First, call on self if the method exists (for standard skorch behavior and your dummy method)
        if hasattr(self, method_name):
            # The method on 'self' (the net) is typically called with 'self' (the net) as the first arg,
            # and then **cb_kwargs. Skorch's internal methods might also pass the callbacks list.
            # For simplicity, let's match the signature of callback methods: (net, **kwargs)
            # So, self.on_train_batch_transformed(self, **cb_kwargs)
            getattr(self, method_name)(self, **cb_kwargs) # Call the dummy method on self

        # Then, iterate through registered callbacks
        for cb_name, cb_instance in self.callbacks_: # self.callbacks_ holds (name, instance) tuples
            if hasattr(cb_instance, method_name):
                # Call the method on the callback instance, passing self (the net) and kwargs
                getattr(cb_instance, method_name)(self, **cb_kwargs)
            # If a callback doesn't have the method, we simply skip it (no AttributeError)
