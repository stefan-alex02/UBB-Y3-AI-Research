from pathlib import Path
from typing import List, Tuple, Callable, Type, Optional

import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from skorch.utils import to_numpy
from torch.utils.data import DataLoader

from .augmentations_utils import cutmix_data
from ..config import DEVICE
from ..dataset_utils import PathImageDataset, ImageDatasetHandler
from ..logger_utils import logger


class SkorchModelAdapter(NeuralNetClassifier):
    """
    Adapter class that extends Skorch's NeuralNetClassifier to work with image datasets.

    This class provides specialized functionality for image classification tasks:
    - Uses PathImageDataset for handling image data from file paths
    - Supports separate train and validation transforms
    - Handles offline augmented data
    - Implements CutMix data augmentation
    - Provides gradient clipping
    - Includes debugging visualization of augmented batches

    Inherits all functionality from NeuralNetClassifier while adding image-specific capabilities.
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
            callbacks: Optional[List[Tuple[str, Callback]]] = None,
            train_transform: Optional[Callable] = None,
            valid_transform: Optional[Callable] = None,
            use_offline_augmented_data: bool = False,
            dataset_handler_ref: Optional[ImageDatasetHandler] = None,
            train_split: Optional[Callable] = None,
            iterator_train__shuffle: bool = True,
            cutmix_alpha: float = 0.0,
            cutmix_probability: float = 0.0,
            gradient_clip_value: Optional[float] = None,
            verbose: int = 1,
            **kwargs
    ):
        """
        Initializes the SkorchModelAdapter with configuration for image classification.

        Args:
            module: Neural network module to use
            criterion: Loss function class
            optimizer: Optimizer class
            lr: Learning rate
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training and validation
            device: Device to use ('cuda', 'cpu', etc.)
            show_first_batch_augmentation: If True, plots first batches for debugging
            callbacks: List of (name, callback) tuples for Skorch callback system
            train_transform: Transform pipeline for training data
            valid_transform: Transform pipeline for validation data
            use_offline_augmented_data: If True, includes pre-generated augmented images
            dataset_handler_ref: Reference to ImageDatasetHandler for offline augmentations
            train_split: Function to split data into train/validation sets
            iterator_train__shuffle: Whether to shuffle training data
            cutmix_alpha: Alpha parameter for CutMix Beta distribution (0 disables CutMix)
            cutmix_probability: Probability of applying CutMix per batch
            gradient_clip_value: Maximum norm for gradient clipping (None disables)
            verbose: Verbosity level
            **kwargs: Additional arguments passed to NeuralNetClassifier

        Raises:
            ValueError: If train_transform or valid_transform is None
            ValueError: If use_offline_augmented_data is True but dataset_handler_ref is None
        """
        self.show_first_batch_augmentation = show_first_batch_augmentation
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_probability = cutmix_probability
        self.gradient_clip_value = gradient_clip_value

        if train_transform is None or valid_transform is None:
            raise ValueError("Both train_transform and valid_transform must be provided")

        self.train_transform = train_transform
        self.valid_transform = valid_transform

        self.use_offline_augmented_data = use_offline_augmented_data
        self.dataset_handler_ref = dataset_handler_ref
        if self.use_offline_augmented_data and self.dataset_handler_ref is None:
            raise ValueError("dataset_handler_ref must be provided if use_offline_augmented_data is True")

        # Collate Functions
        kwargs.setdefault('iterator_train__collate_fn', PathImageDataset.collate_fn)
        kwargs.setdefault('iterator_valid__collate_fn', PathImageDataset.collate_fn)

        super().__init__(
            *args, module=module, criterion=criterion, optimizer=optimizer, lr=lr,
            max_epochs=max_epochs, batch_size=batch_size, device=device,
            callbacks=callbacks,
            train_split=train_split, iterator_train__shuffle=iterator_train__shuffle,
            verbose=verbose, **kwargs
        )

        if self.gradient_clip_value is not None and self.gradient_clip_value > 0:
            self.gradient_clip_fn_ = lambda iterator: torch.nn.utils.clip_grad_norm_(
                self.module_.parameters(), max_norm=self.gradient_clip_value
            )
        else:
            self.gradient_clip_fn_ = None

    def initialize(self):
        """
        Initializes or resets the model state at the beginning of training.

        This method extends the parent's initialize method to reset flags
        that track whether debug visualization has occurred for different
        batch types during the current training run.

        Returns:
            self: Returns self for method chaining
        """
        super().initialize()
        self._first_original_train_batch_plotted_this_fit = False
        self._first_cutmixed_train_batch_plotted_this_fit = False
        self._first_validation_batch_plotted_this_fit = False
        return self

    @staticmethod
    def _plot_debug_batch(images_tensor_to_plot: torch.Tensor, title: str):
        """
        Plots a batch of images for debugging and visualization purposes.

        Args:
            images_tensor_to_plot: Tensor containing batch of images to plot
            title: Title for the plot

        Notes:
            - Uses ResultsPlotter.plot_image_batch from the plotter module
            - Silently catches and logs any exceptions during plotting
            - Detaches and clones tensor to ensure it doesn't affect training
        """
        try:
            from ..plotter import ResultsPlotter
            ResultsPlotter.plot_image_batch(
                images_tensor_to_plot.detach().clone().cpu(),
                title=title, show_plots=True
            )
        except Exception as e:
            logger.error(f"Error in _plot_debug_batch: {e}", exc_info=True)

    def get_split_datasets(self, X, y=None, **fit_params):
        """
        Creates training and validation datasets from input data.

        This method handles:
        - Creating appropriate train/validation splits based on indices
        - Applying correct transforms to each dataset
        - Incorporating offline augmented data if enabled
        - Ensuring augmented samples only appear in the training set

        Args:
            X: List of image paths
            y: List of corresponding labels
            **fit_params: Additional parameters for dataset creation

        Returns:
            Tuple[PyTorchDataset, Optional[PyTorchDataset]]:
                - Training dataset with appropriate transforms
                - Validation dataset with appropriate transforms (or None)

        Raises:
            ValueError: If y is None
        """
        if y is None: raise ValueError("y must be provided.")
        y_arr_orig_pool = to_numpy(y)
        X_paths_np_orig_pool = np.asarray(X)

        ds_train_final = None
        ds_valid_final = None

        if self.train_split:
            indices_in_orig_pool = np.arange(len(X_paths_np_orig_pool))
            ds_train_indices_wrapper, ds_valid_indices_wrapper = \
                self.train_split(indices_in_orig_pool, y=y_arr_orig_pool, **fit_params)

            train_indices_for_this_fold = np.asarray(ds_train_indices_wrapper.indices)

            # Create VALIDATION dataset from ORIGINAL data ONLY
            if ds_valid_indices_wrapper is not None and len(ds_valid_indices_wrapper) > 0:
                valid_indices_for_this_fold = np.asarray(ds_valid_indices_wrapper.indices)
                valid_paths = X_paths_np_orig_pool[valid_indices_for_this_fold].tolist()
                valid_labels = y_arr_orig_pool[valid_indices_for_this_fold].tolist()
                ds_valid_final = PathImageDataset(paths=valid_paths, labels=valid_labels,
                                                  transform=self.valid_transform)
                logger.debug(f"Validation split for fold created with {len(ds_valid_final)} original samples.")

            # Create TRAINING dataset
            current_fold_train_paths_orig = X_paths_np_orig_pool[train_indices_for_this_fold].tolist()
            current_fold_train_labels_orig = y_arr_orig_pool[train_indices_for_this_fold].tolist()

            current_fold_train_original_basenames = set()
            for p_orig in current_fold_train_paths_orig:
                current_fold_train_original_basenames.add(Path(p_orig).stem)

            combined_train_paths_for_fold = current_fold_train_paths_orig[:]
            combined_train_labels_for_fold = current_fold_train_labels_orig[:]

            if self.use_offline_augmented_data and self.dataset_handler_ref:
                logger.debug("Original training set created with "
                             f"{len(combined_train_paths_for_fold)} samples. "
                             "Now checking for relevant offline augmentations...")

                all_aug_paths, all_aug_labels, all_aug_original_basenames = \
                    self.dataset_handler_ref.get_offline_augmented_paths_labels_with_originals()

                added_aug_count = 0
                if all_aug_paths:
                    for aug_path, aug_label, aug_orig_basename in zip(all_aug_paths, all_aug_labels,
                                                                      all_aug_original_basenames):
                        if aug_orig_basename in current_fold_train_original_basenames:
                            combined_train_paths_for_fold.append(aug_path)
                            combined_train_labels_for_fold.append(aug_label)
                            added_aug_count += 1

                    # TODO maybe add param for force sorting
                    # combined_train_paths_for_fold, combined_train_labels_for_fold = \
                    #     zip(*sorted(zip(combined_train_paths_for_fold, combined_train_labels_for_fold),
                    #                   key=lambda x: Path(x[0]).stem))

                    logger.debug(
                        f"Added {added_aug_count} relevant offline augmented samples to current training fold.")
                else:
                    logger.warning("No offline augmented paths found in dataset_handler_ref.")
            else:
                logger.debug("No offline augmentations will be applied.")

            ds_train_final = PathImageDataset(
                paths=combined_train_paths_for_fold,
                labels=combined_train_labels_for_fold,
                transform=self.train_transform
            )
            logger.debug(
                f"Total training split for fold created with {len(ds_train_final)} samples.")

        else:  # No train_split
            logger.debug(
                "No train_split defined by skorch. Using all provided X,y for training, plus all offline augmentations.")
            combined_train_paths = X_paths_np_orig_pool.tolist()
            combined_train_labels = y_arr_orig_pool.tolist()
            if self.use_offline_augmented_data and self.dataset_handler_ref:
                aug_paths, aug_labels, _ = self.dataset_handler_ref.get_offline_augmented_paths_labels_with_originals()
                if aug_paths:
                    combined_train_paths.extend(aug_paths)
                    combined_train_labels.extend(aug_labels)
                    logger.debug(
                        f"Added {len(aug_paths)} offline augmented samples to full training set (no validation split).")
            ds_train_final = PathImageDataset(paths=combined_train_paths, labels=combined_train_labels,
                                              transform=self.train_transform)

        return ds_train_final, ds_valid_final

    def get_iterator(self, dataset, training=False):
        """
        Creates and configures a DataLoader for the given dataset.

        This method ensures:
        - Correct transforms are applied based on training/validation mode
        - Appropriate collate_fn is used based on dataset type
        - DataLoader parameters (batch_size, num_workers, etc.) are properly set

        Args:
            dataset: The dataset to create an iterator for
            training: Whether this is for training (True) or evaluation (False)

        Returns:
            DataLoader: Configured DataLoader for the given dataset

        Raises:
            RuntimeError: If no valid torch.utils.data.Dataset can be determined
        """
        current_dataset: Optional[torch.utils.data.Dataset] = None

        if isinstance(dataset, PathImageDataset):
            current_dataset = dataset
            expected_transform = self.train_transform if training else self.valid_transform
            if current_dataset.transform != expected_transform:
                logger.debug(
                    f"get_iterator: Updating transform on existing PathImageDataset for {'training' if training else 'eval'}.")
                current_dataset.transform = expected_transform

        elif hasattr(dataset, 'X'):
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
            elif isinstance(dataset, torch.utils.data.Dataset):
                current_dataset = dataset
            else:
                logger.warning(
                    f"get_iterator received skorch Dataset with non-path X data of type {type(X_input)}. "
                    f"Falling back to super().get_iterator(). First batch plotting relies on train/validation_step.")
                return super().get_iterator(dataset, training=training)

        elif isinstance(dataset, torch.utils.data.Dataset):
            current_dataset = dataset
        else:
            logger.warning(
                f"get_iterator received unexpected dataset type {type(dataset)}. "
                f"Falling back to super().get_iterator(). First batch plotting relies on train/validation_step.")
            return super().get_iterator(dataset, training=training)

        if current_dataset is None:
            raise RuntimeError("get_iterator: Could not determine or create a valid torch.utils.data.Dataset.")

        # DataLoader
        collate_fn_to_use = PathImageDataset.collate_fn
        if not isinstance(current_dataset, PathImageDataset) and not hasattr(current_dataset, 'paths'):
            if hasattr(current_dataset, 'collate_fn') and current_dataset.collate_fn is not None:
                collate_fn_to_use = current_dataset.collate_fn
            else:
                collate_fn_to_use = torch.utils.data.dataloader.default_collate
                logger.debug(f"Using default torch collate_fn for dataset type: {type(current_dataset)}")

        shuffle = self.iterator_train__shuffle if training else False

        loader_kwargs = {}
        iterator_params = self.get_params()

        num_workers_key = 'iterator_train__num_workers' if training else 'iterator_valid__num_workers'
        pin_memory_key = 'iterator_train__pin_memory' if training else 'iterator_valid__pin_memory'

        loader_kwargs['num_workers'] = iterator_params.get(
            num_workers_key,
            iterator_params.get('iterator__num_workers', 0)
        )
        loader_kwargs['pin_memory'] = iterator_params.get(
            pin_memory_key,
            iterator_params.get('iterator__pin_memory', False)
        )
        loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

        drop_last_flag = False # Default
        if training:
            drop_last_flag = getattr(self, 'iterator_train__drop_last', False)
        else: # For validation/prediction
            drop_last_flag = getattr(self, 'iterator_valid__drop_last', False)

        logger.debug(
            f"Creating DataLoader for {'training' if training else 'validation/evaluation'}: "
            f"dataset_len={len(current_dataset)}, batch_size={self.batch_size}, shuffle={shuffle}, "
            f"drop_last={drop_last_flag}, "
            f"collate_fn_type='{type(collate_fn_to_use).__name__}', "
            f"loader_kwargs={loader_kwargs}"
        )

        return DataLoader(
            current_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn_to_use,
            drop_last=drop_last_flag,
            **loader_kwargs
        )

    def train_step(self, batch, **fit_params):
        """
        Performs a single training step (forward pass, loss calculation, backward pass).

        This method:
        - Handles regular training and CutMix-augmented training
        - Applies gradient clipping if configured
        - Plots debug visualization of batches if enabled
        - Properly computes mixed loss for CutMix batches

        Args:
            batch: Tuple of (inputs, targets) from the DataLoader
            **fit_params: Additional parameters for training

        Returns:
            Dict[str, Any]: Dictionary containing 'loss' and 'y_pred' values
        """
        self.module_.train()
        Xi_original, yi_original = batch

        # Move to device
        Xi = Xi_original.to(self.device)
        yi = yi_original.to(self.device)
        yi = yi.to(dtype=torch.long)

        inputs_for_model = Xi
        is_cutmix_applied_this_batch = False
        loss_lambda_for_plot = 1.0

        if self.show_first_batch_augmentation and not self._first_original_train_batch_plotted_this_fit:
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
            y_pred = self.infer(inputs_for_model, **fit_params)
            loss = self.get_loss(y_pred, yi, X=inputs_for_model, training=True)

        if self.show_first_batch_augmentation and is_cutmix_applied_this_batch and \
                not self._first_cutmixed_train_batch_plotted_this_fit:

            SkorchModelAdapter._plot_debug_batch(inputs_for_model,
                                                 f"First CutMixed Training Batch (lam~{loss_lambda_for_plot:.2f})")
            self._first_cutmixed_train_batch_plotted_this_fit = True

        self.optimizer_.zero_grad()
        loss.backward()
        if self.gradient_clip_fn_:
            self.gradient_clip_fn_(self.module_.parameters())
        self.optimizer_.step()

        return {'loss': loss, 'y_pred': y_pred}

    def validation_step(self, batch, **fit_params):
        """
        Performs a single validation step (forward pass and loss calculation).

        This method:
        - Sets the model to evaluation mode
        - Plots the first validation batch if visualization is enabled
        - Computes loss and predictions without gradient tracking

        Args:
            batch: Tuple of (inputs, targets) from the DataLoader
            **fit_params: Additional parameters for validation

        Returns:
            Dict[str, Any]: Dictionary containing 'loss' and 'y_pred' values
        """
        self.module_.eval()
        Xi_original_val, yi_original_val = batch

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

    def on_train_batch_transformed(self, net, **kwargs):
        """
        Dummy method to satisfy skorch's notify trying to call this on the net itself.
        """
        pass

    def notify(self, method_name: str, **cb_kwargs):
        """
        Calls the specified method on the net and all registered callbacks.
        """
        if hasattr(self, method_name):
            getattr(self, method_name)(self, **cb_kwargs)

        for cb_name, cb_instance in self.callbacks_:
            if hasattr(cb_instance, method_name):
                getattr(cb_instance, method_name)(self, **cb_kwargs)
