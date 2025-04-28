# START OF FILE code_v5_revised.py

import os
import logging
import emoji
import json
import torch
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Matplotlib potentially used for plotting results later, keep import
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Callable, Any, Type
from enum import Enum
from pathlib import Path

import torch.nn as nn
from skorch.helper import SliceDataset
# import torch.nn.functional as F # F not used directly, can be removed if desired
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from torchvision import transforms, datasets, models
# from torchvision.transforms import v2 # v2 not explicitly used, stick to v1 for now

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, train_test_split
    # cross_val_score, cross_val_predict # Not used, cross_validate is preferred
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, make_scorer
)
# from sklearn.preprocessing import label_binarize # Not directly needed, handled by metric functions
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint
from skorch.dataset import Dataset as SkorchDataset # To distinguish from torch.utils.data.Dataset

# --- Global Configurations ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4 # Heuristic for num_workers


# --- Configure Logging ---
class CustomFormatter(logging.Formatter):
    """Custom formatter to include emojis, consistent formatting, and level names."""

    # Define formats for different levels
    log_formats = {
        logging.DEBUG:   "%(asctime)s [%(levelname)s] " + emoji.emojize(":magnifying_glass_tilted_left:") + " %(message)s",
        logging.INFO:    "%(asctime)s [%(levelname)s] " + emoji.emojize(":information:") + " %(message)s",
        logging.WARNING: "%(asctime)s [%(levelname)s] " + emoji.emojize(":warning:") + " %(message)s",
        logging.ERROR:   "%(asctime)s [%(levelname)s] " + emoji.emojize(":red_exclamation_mark:") + " %(message)s",
        logging.CRITICAL:"%(asctime)s [%(levelname)s] " + emoji.emojize(":skull:") + " %(message)s",
    }
    date_format = '%Y-%m-%d %H:%M:%S'

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record according to the level.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message.
        """
        log_fmt = self.log_formats.get(record.levelno, self.log_formats[logging.INFO])
        formatter = logging.Formatter(log_fmt, datefmt=self.date_format)
        return formatter.format(record)


def setup_logger(name: str, log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with custom formatting for console and optional file output.

    Args:
        name (str): The name for the logger.
        log_file (Optional[Union[str, Path]]): Path to the log file. If None, only console logging is enabled.
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Prevent duplicate messages if root logger is configured

    # Remove existing handlers to avoid duplication if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
        try:
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(CustomFormatter())
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create file handler for {log_path}: {e}")


    return logger

# Create logger instance
logger = setup_logger('image_classification', Path('logs') / 'classification.log')


# --- Dataset Handling ---

class DatasetStructure(Enum):
    """Enum to represent the structure of the image dataset directory."""
    FLAT = "flat"  # Root dir contains class folders
    FIXED = "fixed" # Root dir contains 'train' and 'test' subdirs, each with class folders


class TransformedSubset(Dataset):
    """
    A Dataset wrapper that applies a specific transform to a subset of another Dataset.
    Useful for applying data augmentation only to the training subset when the original
    dataset object is shared (e.g., in FLAT structure split).

    Attributes:
        dataset (Dataset): The original dataset (expected to be like ImageFolder).
        indices (List[int]): Indices defining the subset within the original dataset.
        transform (Optional[Callable]): The transform to apply to the images in this subset.
    """
    def __init__(self, dataset: Dataset, indices: List[int], transform: Optional[Callable]):
        """
        Initializes the TransformedSubset.

        Args:
            dataset (Dataset): The base dataset (e.g., an ImageFolder instance).
            indices (List[int]): The list of indices from the base dataset that form this subset.
            transform (Optional[Callable]): The transformation pipeline to apply to images
                                             retrieved from this subset.
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        # We need direct access to ImageFolder's specific attributes if that's the base dataset
        if isinstance(dataset, datasets.ImageFolder):
            self.loader = dataset.loader
            self.samples = dataset.samples # List of (filepath, class_index) tuples
        else:
            # Attempt generic access, might need adjustment for other dataset types
            logger.warning(f"TransformedSubset base dataset is type {type(dataset)}, not ImageFolder. "
                           f"Assuming it has 'samples' and 'loader' attributes or similar structure.")
            self.loader = getattr(dataset, 'loader', lambda x: x) # Default loader if not found
            self.samples = getattr(dataset, 'samples', []) # Default samples if not found
            if not self.samples:
                 logger.error("Base dataset for TransformedSubset lacks 'samples'. Getitem might fail.")


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves the item at the given index within this subset, applying the specific transform.

        Args:
            idx (int): The index within the subset (not the original dataset).

        Returns:
            Tuple[torch.Tensor, int]: The transformed image tensor and its label.

        Raises:
            IndexError: If the index is out of bounds for the subset.
            AttributeError: If the base dataset structure is incompatible.
        """
        if idx >= len(self.indices):
            raise IndexError("Index out of bounds for TransformedSubset")

        original_idx = self.indices[idx]
        try:
            path, target = self.samples[original_idx]
            sample = self.loader(path)
        except Exception as e:
             logger.error(f"Error loading sample at original index {original_idx} (subset index {idx}): {e}")
             # Return dummy data or re-raise? Let's re-raise for now.
             raise e

        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        """
        Returns the number of samples in this subset.

        Returns:
            int: The size of the subset.
        """
        return len(self.indices)

    @property
    def targets(self) -> List[int]:
        """
        Returns the targets (labels) for the samples in this subset.
        Necessary for compatibility with some skorch/sklearn functions.

        Returns:
            List[int]: A list of class indices for the subset.
        """
        # Efficiently get targets for the subset indices
        all_targets = getattr(self.dataset, 'targets', None)
        if all_targets is not None:
            try:
                return [all_targets[i] for i in self.indices]
            except Exception as e:
                logger.warning(f"Could not extract subset targets using indices: {e}. Falling back.")

        # Fallback: slower method if direct targets array isn't available/compatible
        logger.warning("Falling back to slower target extraction for TransformedSubset.")
        subset_targets = []
        for i in self.indices:
             try:
                 _, target = self.samples[i]
                 subset_targets.append(target)
             except Exception as e:
                 logger.error(f"Error getting target for sample at original index {i}: {e}")
                 # Handle error - append a placeholder or raise? Append -1 for now.
                 subset_targets.append(-1)
        return subset_targets


class ImageDatasetHandler:
    """
    Handles loading and preparing image datasets from disk, detecting their structure
    (FLAT or FIXED) and providing access to train, validation, and test subsets/dataloaders
    with appropriate transforms applied.

    Attributes:
        root_path (Path): Path to the dataset root directory.
        img_size (Tuple[int, int]): Target image size (height, width).
        val_split_ratio (float): Fraction of training data to use for validation.
        data_augmentation (bool): Whether to apply augmentation to the training set.
        structure (DatasetStructure): Detected structure of the dataset.
        classes (List[str]): List of class names found in the dataset.
        num_classes (int): Number of unique classes.
        train_dataset (Dataset): The training dataset subset (potentially augmented).
        val_dataset (Dataset): The validation dataset subset (no augmentation).
        test_dataset (Dataset): The test dataset subset (no augmentation).
        train_transform (Callable): Transforms applied to the training data.
        eval_transform (Callable): Transforms applied to validation and test data.
        full_dataset (Optional[Dataset]): The complete dataset (only for FLAT structure).
    """

    def __init__(self,
                 root_path: Union[str, Path],
                 img_size: Tuple[int, int] = (224, 224),
                 val_split_ratio: float = 0.2,
                 data_augmentation: bool = True):
        """
        Initializes the dataset handler.

        Args:
            root_path (Union[str, Path]): Path to the dataset root directory.
            img_size (Tuple[int, int]): Target size to resize images to (height, width).
            val_split_ratio (float): Fraction of training data to reserve for validation (0.0 to 1.0).
            data_augmentation (bool): If True, applies augmentation transforms to the training set.

        Raises:
            FileNotFoundError: If the root_path does not exist or is not a directory.
            ValueError: If val_split_ratio is not between 0 and 1.
        """
        self.root_path = Path(root_path)
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Dataset root path not found or not a directory: {self.root_path}")

        if not 0.0 <= val_split_ratio < 1.0:
            raise ValueError("val_split_ratio must be between 0.0 and 1.0 (exclusive of 1.0)")

        self.img_size = img_size
        self.val_split_ratio = val_split_ratio
        self.data_augmentation = data_augmentation
        self.full_dataset: Optional[datasets.ImageFolder] = None
        self.train_dataset_raw: Optional[Subset] = None # Train subset without augmentation
        self.train_dataset_aug: Optional[TransformedSubset] = None # Train subset with augmentation
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Union[datasets.ImageFolder, Subset]] = None

        # Detect dataset structure
        self.structure = self._detect_structure()
        logger.info(f"Detected dataset structure: {self.structure.value}")

        # Setup transforms
        self.train_transform = self._setup_train_transform()
        self.eval_transform = self._setup_eval_transform()

        # Load dataset based on structure
        try:
            self._load_dataset()
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.root_path}: {e}", exc_info=True)
            raise  # Re-raise after logging

        # Get class information
        self.classes = self._get_classes()
        if not self.classes:
             raise ValueError(f"Could not determine classes for dataset at {self.root_path}")
        self.num_classes = len(self.classes)
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.classes)}")

        # Assign the correct training dataset based on augmentation flag
        self.train_dataset = self.train_dataset_aug if self.data_augmentation else self.train_dataset_raw

    def _detect_structure(self) -> DatasetStructure:
        """
        Detects the dataset structure (FLAT or FIXED) based on subdirectories.

        Returns:
            DatasetStructure: The detected structure.
        """
        root_subdirs = [d.name for d in self.root_path.iterdir() if d.is_dir()]

        if 'train' in root_subdirs and 'test' in root_subdirs:
            train_path = self.root_path / 'train'
            test_path = self.root_path / 'test'
            train_class_dirs = {d.name for d in train_path.iterdir() if d.is_dir()}
            test_class_dirs = {d.name for d in test_path.iterdir() if d.is_dir()}

            if train_class_dirs and train_class_dirs == test_class_dirs:
                logger.debug("Found 'train' and 'test' directories with matching class subdirectories. Assuming FIXED structure.")
                return DatasetStructure.FIXED
            else:
                 logger.warning("Found 'train' and 'test' directories, but class subdirectories don't match or are missing. Assuming FLAT structure.")

        logger.debug("Did not find standard 'train'/'test' structure. Assuming FLAT structure.")
        return DatasetStructure.FLAT

    def _setup_train_transform(self) -> transforms.Compose:
        """Sets up data augmentation transforms for the training data."""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), # Increased rotation slightly
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Slightly stronger jitter
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Added scaling
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet stats
        ])

    def _setup_eval_transform(self) -> transforms.Compose:
        """Sets up transforms for evaluation data (validation/test) without augmentation."""
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self) -> None:
        """Loads the dataset based on the detected structure."""
        if self.structure == DatasetStructure.FLAT:
            self._load_flat_dataset()
        else:  # FIXED
            self._load_fixed_dataset()

    def _load_flat_dataset(self) -> None:
        """Loads a FLAT dataset, creating train/val/test splits."""
        logger.info(f"Loading FLAT dataset from {self.root_path}...")

        # Load the entire dataset once using the eval_transform.
        # We will wrap the training subset later to apply train_transform.
        try:
            full_dataset_obj = datasets.ImageFolder(str(self.root_path), transform=self.eval_transform)
            self.full_dataset = full_dataset_obj # Store the full dataset
            targets = np.array(full_dataset_obj.targets)
            if len(full_dataset_obj) == 0:
                raise ValueError("Loaded dataset is empty.")
            logger.info(f"Full dataset loaded: {len(full_dataset_obj)} samples.")
        except Exception as e:
            logger.error(f"Error loading ImageFolder for FLAT dataset: {e}", exc_info=True)
            raise

        # Ensure there's enough data for splits
        if len(full_dataset_obj) < 3: # Need at least 1 for train, val, test
             raise ValueError("Dataset too small for train/val/test split.")

        indices = np.arange(len(targets))

        # Create initial Train / Test split (e.g., 80% train+val, 20% test)
        test_split_ratio = 0.20
        try:
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=test_split_ratio,
                stratify=targets,
                random_state=RANDOM_SEED
            )
            logger.debug(f"Initial split: {len(train_val_indices)} train+val, {len(test_indices)} test indices.")
        except ValueError as e: # Handle cases where stratification isn't possible (e.g., < 2 members per class)
            logger.warning(f"Stratified train/test split failed ({e}). Attempting non-stratified split.")
            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=test_split_ratio,
                random_state=RANDOM_SEED
            )

        # Split Train into actual Train and Validation
        if self.val_split_ratio > 0 and len(train_val_indices) > 1:
            train_val_targets = targets[train_val_indices]
            try:
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=self.val_split_ratio, # Ratio applied to the train_val set
                    stratify=train_val_targets,
                    random_state=RANDOM_SEED
                )
                logger.debug(f"Train/Val split: {len(train_indices)} train, {len(val_indices)} validation indices.")
            except ValueError as e:
                logger.warning(f"Stratified train/val split failed ({e}). Attempting non-stratified split.")
                train_indices, val_indices = train_test_split(
                    train_val_indices,
                    test_size=self.val_split_ratio,
                    random_state=RANDOM_SEED
                )
        elif len(train_val_indices) <= 1:
             logger.warning("Train+Val set too small for validation split. Using all for training.")
             train_indices = train_val_indices
             val_indices = []
        else: # val_split_ratio is 0
             logger.info("Validation split ratio is 0. No validation set created.")
             train_indices = train_val_indices
             val_indices = []

        # Create Subset objects for val and test (using eval_transform inherited from full_dataset_obj)
        self.val_dataset = Subset(full_dataset_obj, val_indices) if val_indices.size > 0 else None
        self.test_dataset = Subset(full_dataset_obj, test_indices)

        # Create Subset for raw train data (no augmentation)
        self.train_dataset_raw = Subset(full_dataset_obj, train_indices)

        # Create TransformedSubset for augmented train data
        self.train_dataset_aug = TransformedSubset(
            full_dataset_obj, train_indices, transform=self.train_transform
        )

        val_count = len(self.val_dataset) if self.val_dataset else 0
        logger.info(f"FLAT Dataset splits: {len(self.train_dataset_raw)} train, "
                    f"{val_count} validation, "
                    f"{len(self.test_dataset)} test samples.")

    def _load_fixed_dataset(self) -> None:
        """Loads a FIXED dataset with predefined train/test splits."""
        train_path = self.root_path / 'train'
        test_path = self.root_path / 'test'
        logger.info(f"Loading FIXED dataset from {train_path} (train) and {test_path} (test)...")

        # Load the training dataset (with eval_transform initially)
        try:
            train_val_dataset_obj = datasets.ImageFolder(str(train_path), transform=self.eval_transform)
            targets = np.array(train_val_dataset_obj.targets)
            if len(train_val_dataset_obj) == 0:
                 raise ValueError("Loaded train dataset is empty.")
            logger.info(f"Train+Validation dataset loaded: {len(train_val_dataset_obj)} samples.")
        except Exception as e:
            logger.error(f"Error loading ImageFolder for FIXED train dataset: {e}", exc_info=True)
            raise

        indices = np.arange(len(targets))

        # Split Train into actual Train and Validation
        if self.val_split_ratio > 0 and len(indices) > 1:
            try:
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.val_split_ratio,
                    stratify=targets,
                    random_state=RANDOM_SEED
                )
                logger.debug(f"Train/Val split: {len(train_indices)} train, {len(val_indices)} validation indices.")
            except ValueError as e:
                logger.warning(f"Stratified train/val split failed ({e}). Attempting non-stratified split.")
                train_indices, val_indices = train_test_split(
                    indices,
                    test_size=self.val_split_ratio,
                    random_state=RANDOM_SEED
                )
        elif len(indices) <= 1:
             logger.warning("Train set too small for validation split. Using all for training.")
             train_indices = indices
             val_indices = []
        else: # val_split_ratio is 0
             logger.info("Validation split ratio is 0. No validation set created.")
             train_indices = indices
             val_indices = []

        # Create Subset for validation (using eval_transform)
        self.val_dataset = Subset(train_val_dataset_obj, val_indices) if val_indices.size > 0 else None

        # Create Subset for raw train data (no augmentation)
        self.train_dataset_raw = Subset(train_val_dataset_obj, train_indices)

        # Create TransformedSubset for augmented train data
        self.train_dataset_aug = TransformedSubset(
            train_val_dataset_obj, train_indices, transform=self.train_transform
        )

        # Load the test dataset (always with eval_transform)
        try:
            self.test_dataset = datasets.ImageFolder(str(test_path), transform=self.eval_transform)
            if len(self.test_dataset) == 0:
                 logger.warning("Loaded test dataset is empty.")
            logger.info(f"Test dataset loaded: {len(self.test_dataset)} samples.")
        except Exception as e:
            logger.error(f"Error loading ImageFolder for FIXED test dataset: {e}", exc_info=True)
            raise

        val_count = len(self.val_dataset) if self.val_dataset else 0
        logger.info(f"FIXED Dataset loaded: {len(self.train_dataset_raw)} train, "
                    f"{val_count} validation, "
                    f"{len(self.test_dataset)} test samples.")


    def _get_classes(self) -> List[str]:
        """
        Retrieves the list of class names from the loaded dataset.

        Returns:
            List[str]: Sorted list of class names.

        Raises:
            RuntimeError: If classes cannot be determined from any loaded dataset part.
        """
        if self.structure == DatasetStructure.FLAT and self.full_dataset:
            return sorted(self.full_dataset.classes)
        elif self.structure == DatasetStructure.FIXED:
            # Try train_dataset_raw first (as it's derived from train ImageFolder)
            if self.train_dataset_raw and isinstance(self.train_dataset_raw.dataset, datasets.ImageFolder):
                 return sorted(self.train_dataset_raw.dataset.classes)
            # Fallback to test dataset if train failed or wasn't ImageFolder based
            elif self.test_dataset and isinstance(self.test_dataset, datasets.ImageFolder):
                 logger.warning("Getting classes from test dataset as train dataset info wasn't available.")
                 return sorted(self.test_dataset.classes)
        # Fallback: try scanning directories if objects don't have class info
        logger.warning("Attempting to determine classes by scanning directories as dataset objects lacked class info.")
        source_path = self.root_path
        if self.structure == DatasetStructure.FIXED:
            source_path = self.root_path / 'train' # Prefer train dir for FIXED
        try:
            classes = sorted([d.name for d in source_path.iterdir() if d.is_dir() and not d.name.lower() in ['train', 'test']])
            if classes:
                return classes
        except Exception as e:
            logger.error(f"Failed to scan directories for class names in {source_path}: {e}")

        raise RuntimeError("Could not determine class names for the dataset.")


    def get_train_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Gets a DataLoader for the training dataset. Applies augmentation if enabled.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data at every epoch.

        Returns:
            DataLoader: DataLoader for the training data.

        Raises:
            ValueError: If the training dataset is not loaded.
        """
        if self.train_dataset is None:
             raise ValueError("Training dataset is not loaded or available.")
        logger.debug(f"Creating train DataLoader (augmented: {self.data_augmentation}) with batch_size={batch_size}, shuffle={shuffle}")
        return DataLoader(
            self.train_dataset, # This points to augmented or raw based on init flag
            batch_size=batch_size,
            shuffle=shuffle,
            # num_workers=NUM_WORKERS,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=True # Drop last incomplete batch for potentially smoother training
        )

    def get_val_dataloader(self, batch_size: int = 32) -> Optional[DataLoader]:
        """
        Gets a DataLoader for the validation dataset (no augmentation).

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            Optional[DataLoader]: DataLoader for the validation data, or None if no validation set exists.
        """
        if self.val_dataset is None:
            logger.debug("No validation dataset available, returning None for DataLoader.")
            return None
        logger.debug(f"Creating validation DataLoader with batch_size={batch_size}")
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle validation data
            # num_workers=NUM_WORKERS,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    def get_test_dataloader(self, batch_size: int = 32) -> DataLoader:
        """
        Gets a DataLoader for the test dataset (no augmentation).

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            DataLoader: DataLoader for the test data.

        Raises:
            ValueError: If the test dataset is not loaded.
        """
        if self.test_dataset is None:
             raise ValueError("Test dataset is not loaded or available.")
        logger.debug(f"Creating test DataLoader with batch_size={batch_size}")
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False, # No need to shuffle test data
            # num_workers=NUM_WORKERS,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    def get_full_dataset(self) -> Dataset:
        """
        Gets the full dataset object (only available for FLAT structure).

        Returns:
            Dataset: The complete dataset object (likely an ImageFolder).

        Raises:
            ValueError: If called on a FIXED structure dataset or if the full dataset is not available.
        """
        if self.structure == DatasetStructure.FLAT:
            if self.full_dataset is None:
                 raise ValueError("Full dataset requested for FLAT structure, but it was not loaded.")
            return self.full_dataset
        else:
            raise ValueError("Full dataset is only available for FLAT structure datasets.")

    def get_train_dataset(self) -> Dataset:
        """Gets the training dataset (potentially augmented based on initialization)."""
        if self.train_dataset is None:
             raise ValueError("Training dataset is not available.")
        return self.train_dataset

    def get_val_dataset(self) -> Optional[Dataset]:
        """Gets the validation dataset."""
        return self.val_dataset

    def get_test_dataset(self) -> Dataset:
        """Gets the test dataset."""
        if self.test_dataset is None:
             raise ValueError("Test dataset is not available.")
        return self.test_dataset


# --- Model Definitions ---

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for image classification.

    Consists of three convolutional layers with ReLU activation and max pooling,
    followed by a fully connected classifier head with dropout.

    Attributes:
        num_classes (int): The number of output classes for the classification task.
    """
    def __init__(self, num_classes: int):
        """
        Initializes the SimpleCNN model.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Input: 3 channels (RGB)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Halves spatial dimensions
            # Output: 32 x H/2 x W/2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Halves spatial dimensions again
            # Output: 64 x H/4 x W/4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Halves spatial dimensions again
            # Output: 128 x H/8 x W/8
        )
        # Adaptive average pooling reduces spatial dimensions to a fixed size (e.g., 7x7)
        # regardless of the input image size (after feature extraction).
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Output: 128 x 7 x 7 = 6272 features

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Regularization
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"SimpleCNN initialized with {num_classes} output classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the SimpleCNN.

        Args:
            x (torch.Tensor): Input batch of images (BatchSize x Channels x Height x Width).

        Returns:
            torch.Tensor: Output logits (BatchSize x num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.classifier(x)
        return x


class SimpleViT(nn.Module):
    """
    A Vision Transformer (ViT) model for image classification, using a pre-trained
    ViT-Base (vit_b_16) from torchvision and replacing the classification head.

    Freezes most pre-trained layers for faster fine-tuning.

    Attributes:
        num_classes (int): The number of output classes for the classification task.
        model (nn.Module): The underlying pre-trained ViT model with modified head.
    """
    def __init__(self, num_classes: int):
        """
        Initializes the SimpleViT model.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        # Load pre-trained ViT-Base-16 model
        logger.debug("Loading pre-trained vit_b_16 model...")
        vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        logger.debug("Pre-trained model loaded.")

        # Replace the classification head
        original_hidden_dim = vit_model.heads.head.in_features # Access hidden dim correctly
        vit_model.heads.head = nn.Linear(original_hidden_dim, self.num_classes)
        logger.debug(f"Replaced ViT classification head with a new one for {num_classes} classes.")

        # Freeze most layers for transfer learning
        # Unfreeze only the classification head and maybe the last few transformer blocks
        num_layers_to_unfreeze = 4 # Example: Unfreeze head + last few layers/norm
        total_params = len(list(vit_model.parameters()))
        for i, param in enumerate(vit_model.parameters()):
            if i < total_params - num_layers_to_unfreeze:
                 param.requires_grad = False
            else:
                 param.requires_grad = True
                 # logger.debug(f"Unfreezing ViT parameter {i+1}/{total_params}") # Very verbose

        logger.info(f"SimpleViT: Froze first {total_params - num_layers_to_unfreeze} parameter groups, "
                    f"unfroze last {num_layers_to_unfreeze} for fine-tuning.")

        self.model = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the SimpleViT.

        Args:
            x (torch.Tensor): Input batch of images (BatchSize x Channels x Height x Width).

        Returns:
            torch.Tensor: Output logits (BatchSize x num_classes).
        """
        return self.model(x)


class DiffusionClassifier(nn.Module):
    """
    Placeholder for a diffusion-inspired classifier.
    Currently implemented using a pre-trained ResNet50 backbone with a modified
    classifier head. This does *not* implement an actual diffusion process for classification
    but serves as a structural placeholder for a potentially more complex model.

    Attributes:
        num_classes (int): The number of output classes.
        backbone (nn.Module): The feature extraction backbone (ResNet50).
        diffusion_head (nn.Module): The classification head layers.
    """
    def __init__(self, num_classes: int):
        """
        Initializes the DiffusionClassifier model.

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        logger.debug("Loading pre-trained ResNet50 model as backbone for DiffusionClassifier...")
        # Using ResNet50 as a strong backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        logger.debug("Pre-trained ResNet50 loaded.")

        # Remove the original fully connected layer
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity() # Remove final layer, just output features
        self.backbone = resnet

        # Freeze most backbone layers (optional, adjust as needed)
        num_layers_to_unfreeze = 5 # e.g., unfreeze last layer + fc (which is identity)
        total_params = len(list(self.backbone.parameters()))
        for i, param in enumerate(self.backbone.parameters()):
            if i < total_params - num_layers_to_unfreeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        logger.info(f"DiffusionClassifier (ResNet50): Froze first {total_params - num_layers_to_unfreeze} backbone param groups.")


        # Define a new classifier head (simulating a "diffusion" inspired structure perhaps)
        # Input dimension matches the output features of the ResNet backbone (2048 for ResNet50)
        self.diffusion_head = nn.Sequential(
            nn.Linear(in_features, 1024), # Reduce dimensionality
            nn.BatchNorm1d(1024), # Add normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.4), # Slightly increased dropout
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"DiffusionClassifier initialized with ResNet50 backbone and custom head for {num_classes} classes.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the DiffusionClassifier.

        Args:
            x (torch.Tensor): Input batch of images (BatchSize x Channels x Height x Width).

        Returns:
            torch.Tensor: Output logits (BatchSize x num_classes).
        """
        features = self.backbone(x) # Get features from ResNet (B x 2048)
        # Flatten features isn't needed if backbone ends with AdaptiveAvgPool (like ResNet)
        # features = torch.flatten(features, 1)
        logits = self.diffusion_head(features) # Pass features through the custom head
        return logits


# --- Skorch Model Adapter ---

class SkorchModelAdapter(NeuralNetClassifier):
    """
    A skorch NeuralNetClassifier adapter for PyTorch image classification models,
    making them compatible with scikit-learn workflows (GridSearch, cross_validate).

    Handles model instantiation, optimizer setup, standard callbacks (EarlyStopping,
    LR Scheduling, Checkpointing), and device management.

    Crucially, it's configured with `train_split=None` to prevent skorch from creating
    its own validation split, allowing external control via CV iterators or explicit
    validation data passed to `fit`.
    """
    def __init__(
        self,
        module: Optional[Type[nn.Module]] = None,
        module__num_classes: Optional[int] = None,
        criterion: Type[nn.Module] = nn.CrossEntropyLoss,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr: float = 0.001,
        optimizer__weight_decay: float = 0.01,
        max_epochs: int = 20,
        batch_size: int = 32,
        device: str = DEVICE,
        callbacks: Optional[List[Tuple[str, skorch.callbacks.Callback]]] = 'default',
        patience: int = 10,
        monitor: str = 'valid_loss',
        lr_scheduler_policy: str = 'ReduceLROnPlateau',
        lr_scheduler_patience: int = 5,
        # --- Use ValidSplit for internal validation ---
        train_split: Optional[Callable] = skorch.dataset.ValidSplit(cv=0.2, stratified=True, random_state=RANDOM_SEED),
        classes = None, # Pass classes for scoring compatibility
        verbose: int = 1, # Set to 1 or 2 to see skorch epoch logs
        **kwargs
    ):
        """
        Initializes the SkorchModelAdapter.

        Args:
            module (Optional[Type[nn.Module]]): The PyTorch nn.Module class to wrap.
            module__num_classes (Optional[int]): Number of classes, passed to the module's constructor.
                                                 Use `module__<param_name>` for other module args.
            criterion (Type[nn.Module]): The loss function class.
            optimizer (Type[torch.optim.Optimizer]): The optimizer class.
            lr (float): Learning rate for the optimizer.
            optimizer__weight_decay (float): Weight decay for the optimizer.
            max_epochs (int): Maximum number of training epochs.
            batch_size (int): Number of samples per batch.
            device (str): The device to run computations on ('cuda', 'cpu').
            callbacks (Optional[List[Tuple[str, skorch.callbacks.Callback]]]): List of skorch callbacks or 'default'.
            patience (int): Patience for the EarlyStopping callback.
            monitor (str): Metric to monitor for EarlyStopping and Checkpoint.
            lr_scheduler_policy (str): Policy for the LRScheduler callback (e.g., 'ReduceLROnPlateau').
            lr_scheduler_patience (int): Patience for the learning rate scheduler.
            train_split (Optional[Callable]): Should be None to disable skorch's internal split.
                                              Validation is handled by CV or explicit `fit(X, y, X_val, y_val)`.
            **kwargs: Additional arguments passed to the skorch.NeuralNetClassifier parent class.
        """
        self.module_class = module # Store the class for potential re-instantiation if needed
        self.module_init_kwargs = {k.split('__', 1)[1]: v for k, v in kwargs.items() if k.startswith('module__')}
        if module__num_classes is not None:
            self.module_init_kwargs['num_classes'] = module__num_classes

        # Setup default callbacks (same as before)
        if callbacks == 'default':
            callbacks = [
                # ... (callbacks definition remains the same) ...
                ('early_stopping', EarlyStopping(monitor=monitor, patience=patience, load_best=True, lower_is_better=monitor.endswith('_loss'))),
                ('checkpoint', Checkpoint(monitor=f'{monitor}_best', f_params='best_model.pt', dirname=f"skorch_cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}", load_best=False)),
                ('lr_scheduler', LRScheduler(policy=lr_scheduler_policy, monitor=monitor, mode='min' if monitor.endswith('_loss') else 'max', patience=lr_scheduler_patience, factor=0.1))
            ]
        elif callbacks is None:
             callbacks = []

        # --- Initialize NeuralNetClassifier ---
        super().__init__(
            module=module,
            module__num_classes=module__num_classes,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            optimizer__weight_decay=optimizer__weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            callbacks=callbacks,
            train_split=train_split, # Pass ValidSplit instance
            classes=classes,         # Pass classes
            verbose=verbose,         # Pass verbose
            **kwargs
        )

    # Override get_dataset to handle the specific case where
    # X is a Dataset and y is provided (by sklearn CV)
    # def get_dataset(self, X, y=None):
    #     """
    #     Override to handle the specific case where X is a list of
    #     (feature, target) tuples (from sklearn CV) and ensure targets
    #     are LongTensors for CrossEntropyLoss.
    #     """
    #     if isinstance(X, list) and X and isinstance(X[0], (tuple, list)) and len(X[0]) == 2:
    #         # Case 1: X is a list of (feature, target) tuples (from sklearn CV slice).
    #         logger.debug("[DEBUG] Adapter.get_dataset: X is list of tuples. Manual unpacking and stacking.")
    #         try:
    #             X_tensors = [item[0] for item in X]
    #             if not X_tensors:
    #                 X_features_stacked = torch.empty((0, *X_tensors[0].shape) if X else (0,))
    #             else:
    #                 X_features_stacked = torch.stack(X_tensors, dim=0)
    #
    #             y_targets = [item[1] for item in X] # List of Python ints
    #
    #             # --- Convert targets to LongTensor ---
    #             # Create a torch tensor directly with the correct dtype
    #             y_targets_tensor = torch.tensor(y_targets, dtype=torch.long)
    #             # --- End Conversion ---
    #
    #             logger.debug(f"[DEBUG] Adapter.get_dataset: Instantiating SkorchDataset with stacked tensor "
    #                          f"(shape={X_features_stacked.shape}) and target tensor (shape={y_targets_tensor.shape}, dtype={y_targets_tensor.dtype}).")
    #             # Pass Tensor X and Tensor y
    #             return SkorchDataset(X_features_stacked, y_targets_tensor)
    #
    #         except Exception as e:
    #              logger.error(f"Error during manual unpacking/stacking or SkorchDataset instantiation: {e}", exc_info=True)
    #              raise RuntimeError("Failed to process list of tuples input during CV split.") from e
    #     else:
    #         # Case 2: Default behavior for other input types
    #         logger.debug(f"[DEBUG] Adapter.get_dataset: X is not list of tuples. Using default super().get_dataset(X, y).")
    #         # Ensure y=None is passed if X is a Dataset that yields targets
    #         if isinstance(X, (Dataset, SkorchDataset)) and y is not None:
    #              logger.debug("[DEBUG] Adapter.get_dataset: X is Dataset and y is not None (default case). Ignoring y.")
    #              return super().get_dataset(X, y=None)
    #         # If y is provided (e.g., X and y are numpy arrays), ensure y becomes LongTensor
    #         elif y is not None and not isinstance(y, torch.Tensor):
    #              logger.debug("[DEBUG] Adapter.get_dataset: Converting provided y to LongTensor.")
    #              y = torch.tensor(np.array(y), dtype=torch.long) # Ensure numpy first for safety, then tensor
    #         elif y is not None and isinstance(y, torch.Tensor) and y.dtype != torch.long:
    #              logger.debug(f"[DEBUG] Adapter.get_dataset: Converting provided y tensor from {y.dtype} to LongTensor.")
    #              y = y.to(dtype=torch.long)
    #
    #         return super().get_dataset(X, y)

    # def fit(self, X, y=None, **fit_params):
    #     """
    #     Fits the model to the training data.
    #
    #     Handles PyTorch Datasets directly. If `X_val` and `y_val` are provided in
    #     `fit_params`, they are used for validation (requires `train_split=None`).
    #
    #     Args:
    #         X (Dataset or np.ndarray or torch.Tensor): Training data. Can be a PyTorch Dataset.
    #         y (Optional[Any]): Training targets. Should be None if X is a Dataset that yields (data, target).
    #         **fit_params: Additional parameters passed to the underlying skorch fit method.
    #                       Crucially, can include `X_val` and `y_val` for validation data.
    #
    #     Returns:
    #         self: The fitted estimator instance.
    #     """
    #     module_name = self.module_class.__name__ if self.module_class else "Model"
    #     logger.info(f"Starting training for {module_name}...")
    #     # No need to modify y here, get_split_datasets will handle it.
    #     super().fit(X, y, **fit_params)
    #     logger.info(f"Finished training for {module_name}.")
    #     return self
    #
    # def infer(self, x, **fit_params):
    #     """
    #     Perform inference. Assumes 'x' is a Tensor.
    #     Filters fit_params and moves data to device.
    #     """
    #     if not isinstance(x, torch.Tensor):
    #          # Add a check here just in case, but the fix should be upstream
    #          logger.error(f"[ERROR] Infer received non-Tensor input: {type(x)}. Upstream issue likely.")
    #          raise TypeError(f"SkorchModelAdapter.infer received input of type {type(x)}, expected torch.Tensor.")
    #
    #     module_forward_params = {
    #         k: v for k, v in fit_params.items()
    #         if k not in ['X_val', 'y_val']
    #     }
    #     # Move to device - x is confirmed to be a Tensor here
    #     x = x.to(self.device)
    #     return self.module_(x, **module_forward_params)
    #
    # # Override validation_step for debugging and ensuring execution
    # def validation_step(self, batch, **fit_params):
    #     """Perform a validation step."""
    #     logger.debug("[DEBUG] Adapter.validation_step: Called.")
    #     self.module_.eval()
    #
    #     # Move the whole batch structure to the device first
    #     try:
    #         batch = skorch.utils.to_device(batch, self.device)
    #     except Exception as e:
    #         # Catch potential errors if batch structure is very unexpected
    #         logger.error(f"Failed to move batch to device in validation_step. Batch type: {type(batch)}. Error: {e}")
    #         raise RuntimeError("Error moving batch to device in validation_step.") from e
    #
    #     # Now unpack the batch (which should contain tensors on the correct device)
    #     try:
    #         Xi, yi = batch
    #         # Optional: Add explicit check that Xi is now a Tensor
    #         if not isinstance(Xi, torch.Tensor):
    #             raise TypeError(f"Xi unpacked from batch is not a Tensor, but {type(Xi)}")
    #         if not isinstance(yi, torch.Tensor):
    #              logger.warning(f"yi unpacked from batch is not a Tensor ({type(yi)}). Loss calculation might fail.")
    #
    #     except (ValueError, TypeError) as e:
    #          logger.error(f"Failed to unpack batch in validation_step after moving to device. Batch type: {type(batch)}. Error: {e}")
    #          raise ValueError("Could not unpack batch in validation_step. Expected (Tensor, Tensor) structure.") from e
    #
    #     # Perform inference and loss calculation
    #     with torch.no_grad():
    #         # Pass the Tensor Xi to infer
    #         y_pred = self.infer(Xi, **fit_params)
    #         # Calculate loss using device-local tensors y_pred and yi
    #         loss = self.get_loss(y_pred, yi, X=Xi, training=False)
    #
    #     logger.debug(f"[DEBUG] Adapter.validation_step: Loss calculated: {loss.item():.4f}")
    #
    #     return {
    #         'loss': loss,
    #         'y_pred': y_pred,
    #     }
    #
    # def get_split_datasets(self, X, y=None, **fit_params):
    #     """
    #     Override to handle different types of X input, especially the
    #     list of tuples format coming from sklearn CV slicing PyTorch Datasets,
    #     and explicit validation data passed via fit_params.
    #     Manually unpacks list of tuples before creating SkorchDataset.
    #     """
    #     # --- Process Training Data ---
    #     if isinstance(X, (Dataset, SkorchDataset)):
    #         # Case 1: X is already a Dataset. Ignore y, let SkorchDataset handle it.
    #         logger.debug("[DEBUG] Adapter.get_split_datasets: X is Dataset, using get_dataset(X, y=None).")
    #         dataset_train = self.get_dataset(X, y=None)
    #     elif isinstance(X, list) and X and isinstance(X[0], (tuple, list)) and len(X[0]) == 2:
    #         # Case 2: X is a list of (feature, target) tuples (from sklearn CV slice).
    #         # Manually unpack features and targets BEFORE creating SkorchDataset.
    #         logger.debug("[DEBUG] Adapter.get_split_datasets: X is list of tuples. Manual unpacking.")
    #         try:
    #             X_features = [item[0] for item in X] # List of feature tensors
    #             y_targets = [item[1] for item in X]  # List of targets
    #             # Create dataset with separated features and targets
    #             dataset_train = self.get_dataset(X_features, y_targets)
    #             logger.debug(f"[DEBUG] Adapter.get_split_datasets: Created SkorchDataset from unpacked list (len={len(X_features)}).")
    #         except Exception as e:
    #              logger.error(f"Error during manual unpacking or SkorchDataset creation from list: {e}", exc_info=True)
    #              raise RuntimeError("Failed to process list of tuples input during CV split.") from e
    #     else:
    #         # Case 3: X is likely features (e.g., numpy array), use passed y.
    #         logger.debug("[DEBUG] Adapter.get_split_datasets: X is assumed features, using get_dataset(X, y).")
    #         dataset_train = self.get_dataset(X, y)
    #
    #     # --- Process Validation Data ---
    #     dataset_valid = None
    #     if 'X_val' in fit_params:
    #         logger.debug("[DEBUG] Adapter.get_split_datasets: Found X_val in fit_params. Processing validation set.")
    #         X_val = fit_params['X_val']
    #         y_val = fit_params.get('y_val')
    #
    #         # Apply similar logic for validation data
    #         if isinstance(X_val, (Dataset, SkorchDataset)):
    #             logger.debug("[DEBUG] Adapter.get_split_datasets: X_val is Dataset, using get_dataset(X_val, y=None).")
    #             dataset_valid = self.get_dataset(X_val, y=None)
    #         elif isinstance(X_val, list) and X_val and isinstance(X_val[0], (tuple, list)) and len(X_val[0]) == 2:
    #              logger.debug("[DEBUG] Adapter.get_split_datasets: X_val is list of tuples. Manual unpacking.")
    #              try:
    #                  X_val_features = [item[0] for item in X_val]
    #                  y_val_targets = [item[1] for item in X_val]
    #                  dataset_valid = self.get_dataset(X_val_features, y_val_targets)
    #                  logger.debug(f"[DEBUG] Adapter.get_split_datasets: Created validation SkorchDataset from unpacked list (len={len(X_val_features)}).")
    #              except Exception as e:
    #                  logger.error(f"Error during manual unpacking or SkorchDataset creation from validation list: {e}", exc_info=True)
    #                  raise RuntimeError("Failed to process list of tuples input for validation data.") from e
    #         else:
    #             logger.debug("[DEBUG] Adapter.get_split_datasets: X_val is assumed features, using get_dataset(X_val, y_val).")
    #             dataset_valid = self.get_dataset(X_val, y_val)
    #     elif self.train_split:
    #          # Fallback if no X_val provided, but train_split is active (shouldn't happen)
    #          logger.warning("[DEBUG] Adapter.get_split_datasets: No X_val, using self.train_split (unexpected).")
    #          initial_dataset_for_split = self.get_dataset(X, y)
    #          dataset_train, dataset_valid = self.train_split(initial_dataset_for_split, **fit_params)
    #
    #
    #     return dataset_train, dataset_valid

    # No need to override predict, predict_proba, score etc. unless specific logic is needed.
    # Skorch handles passing Datasets to these methods correctly.


# --- Classification Pipeline ---

class ClassificationPipeline:
    """
    Manages the end-to-end image classification process, including data loading,
    model selection, training, hyperparameter tuning, evaluation, and results saving.

    Integrates ImageDatasetHandler and SkorchModelAdapter with scikit-learn's
    cross-validation and search tools.

    Attributes:
        dataset_handler (ImageDatasetHandler): Handler for accessing dataset splits/loaders.
        model_adapter (SkorchModelAdapter): Skorch wrapper for the PyTorch model.
        model_type (str): Identifier for the model architecture ('cnn', 'vit', 'diffusion').
        results_dir (Path): Directory where results (metrics, logs, models) are saved.
    """

    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: str = 'cnn',
                 model_load_path: Optional[Union[str, Path]] = None,
                 img_size: Tuple[int, int] = (224, 224),
                 results_dir: Union[str, Path] = 'results',
                 val_split_ratio: float = 0.2,
                 data_augmentation: bool = True,
                 # Skorch adapter params (can be overridden by specific methods)
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10):
        """
        Initializes the classification pipeline.

        Args:
            dataset_path (Union[str, Path]): Path to the root directory of the image dataset.
            model_type (str): Type of model architecture ('cnn', 'vit', or 'diffusion').
            model_load_path (Optional[Union[str, Path]]): Path to a pre-trained model state_dict
                                                          to load. If None, a new model is created.
            img_size (Tuple[int, int]): Target image size (height, width).
            results_dir (Union[str, Path]): Base directory to save results and artifacts.
            val_split_ratio (float): Fraction of training data to use for validation.
            data_augmentation (bool): Whether to apply augmentation to the training set.
            lr (float): Default learning rate for the model adapter.
            max_epochs (int): Default maximum epochs for the model adapter.
            batch_size (int): Default batch size for the model adapter.
            patience (int): Default patience for early stopping.

        Raises:
            FileNotFoundError: If dataset_path doesn't exist.
            ValueError: If model_type is unsupported or val_split_ratio is invalid.
        """
        self.dataset_path = Path(dataset_path)
        self.model_type = model_type.lower()
        logger.info(f"Initializing Classification Pipeline:")
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Model Type: {self.model_type}")

        # Initialize dataset handler
        self.dataset_handler = ImageDatasetHandler(
            root_path=self.dataset_path,
            img_size=img_size,
            val_split_ratio=val_split_ratio,
            data_augmentation=data_augmentation
        )

        # Set up results directory path
        base_results_dir = Path(results_dir)
        dataset_name = self.dataset_path.name
        self.results_dir = base_results_dir / dataset_name / self.model_type
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Results will be saved to: {self.results_dir}")

        # Select model class based on type
        model_class = self._get_model_class(self.model_type)

        # Initialize model adapter
        self.model_adapter = SkorchModelAdapter(
            module=model_class,
            module__num_classes=self.dataset_handler.num_classes,
            classes=np.arange(self.dataset_handler.num_classes),
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience, # Passed to SkorchModelAdapter for EarlyStopping default
            # Other SkorchModelAdapter defaults (optimizer, criterion, etc.) are used
            # Add verbose=1 if you want to see epoch progress during CV fits
            verbose=3
        )
        logger.info(f"  Model Adapter: Initialized with {model_class.__name__}")

        # Load pre-trained weights if specified
        if model_load_path:
            self.load_model(model_load_path)

        logger.info(f"Pipeline initialized successfully for {self.model_type} model and "
                    f"{self.dataset_handler.structure.value} dataset structure.")

    def _get_model_class(self, model_type_str: str) -> Type[nn.Module]:
        """Gets the PyTorch model class based on the model type string."""
        model_mapping = {
            'cnn': SimpleCNN,
            'vit': SimpleViT,
            'diffusion': DiffusionClassifier # Placeholder name
        }
        model_class = model_mapping.get(model_type_str)
        if model_class is None:
            raise ValueError(f"Unsupported model type: '{model_type_str}'. Choose from {list(model_mapping.keys())}.")
        return model_class

    def _compute_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes a comprehensive set of classification metrics.

        Calculates overall accuracy and macro-averaged precision, recall, specificity,
        F1-score. If y_score (probabilities) are provided, also calculates macro-averaged
        ROC AUC and Precision-Recall AUC (AUPRC). Handles multi-class cases using OvR strategy.

        Args:
            y_true (np.ndarray): Ground truth labels (integer class indices).
            y_pred (np.ndarray): Predicted labels (integer class indices).
            y_score (Optional[np.ndarray]): Predicted probabilities or scores, shape (n_samples, n_classes).
                                             Required for AUC calculations.

        Returns:
            Dict[str, Any]: A dictionary containing computed metrics, including overall accuracy,
                            macro averages, and potentially per-class details (commented out for brevity).
        """
        if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
        if y_score is not None and not isinstance(y_score, np.ndarray): y_score = np.array(y_score)


        metrics = {'accuracy': accuracy_score(y_true, y_pred)}
        class_labels = sorted(list(np.unique(y_true))) # Use actual labels present
        n_classes = len(class_labels)

        # Use class indices consistent with dataset_handler if possible
        class_names = self.dataset_handler.classes
        if len(class_names) != self.dataset_handler.num_classes:
             logger.warning("Mismatch between dataset handler classes and number of classes. Using unique labels from y_true.")
             class_names = [f"Class {lbl}" for lbl in class_labels]

        # Mapping from actual label value to its index in the sorted unique list
        label_to_idx = {label: i for i, label in enumerate(class_labels)}

        per_class_metrics = {name: {} for name in class_names}
        all_precisions, all_recalls, all_specificities, all_f1s = [], [], [], []
        all_roc_aucs, all_pr_aucs = [], []

        can_compute_auc = y_score is not None and y_score.shape == (len(y_true), self.dataset_handler.num_classes)
        if y_score is not None and not can_compute_auc:
             logger.warning(f"y_score shape {y_score.shape} incompatible with y_true len {len(y_true)} and "
                            f"num_classes {self.dataset_handler.num_classes}. Cannot compute AUCs.")


        for i, class_label in enumerate(class_labels):
            class_name = class_names[class_label] # Assuming class_label matches index in handler.classes
            true_is_class = (y_true == class_label).astype(int)
            pred_is_class = (y_pred == class_label).astype(int)

            # --- Calculate Basic Metrics ---
            precision = precision_score(true_is_class, pred_is_class, zero_division=0)
            recall = recall_score(true_is_class, pred_is_class, zero_division=0) # Sensitivity
            f1 = f1_score(true_is_class, pred_is_class, zero_division=0)

            # Specificity = True Negatives / (True Negatives + False Positives)
            # This is recall of the negative class (1 - true_is_class, 1 - pred_is_class)
            specificity = recall_score(1 - true_is_class, 1 - pred_is_class, zero_division=0)

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_specificities.append(specificity)
            all_f1s.append(f1)

            # Store per-class (optional, can make results dict large)
            # per_class_metrics[class_name]['precision'] = precision
            # per_class_metrics[class_name]['recall'] = recall
            # per_class_metrics[class_name]['specificity'] = specificity
            # per_class_metrics[class_name]['f1'] = f1

            # --- Calculate AUC Metrics (if scores available) ---
            roc_auc, pr_auc = np.nan, np.nan # Default to NaN
            if can_compute_auc:
                # Ensure we have scores corresponding to the current class_label index
                # The column index in y_score should match the class_label value
                try:
                    score_for_class = y_score[:, class_label] # Assumes y_score columns align with class indices 0..N-1
                except IndexError:
                    logger.warning(f"Cannot get y_score column for class label {class_label}. Skipping AUC calculation for this class.")
                    score_for_class = None

                if score_for_class is not None and len(np.unique(true_is_class)) > 1: # AUC requires both classes present
                    try:
                        roc_auc = roc_auc_score(true_is_class, score_for_class)
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC for class {class_name}: {e}")
                        roc_auc = np.nan # Ensure NaN on error

                    try:
                        prec, rec, _ = precision_recall_curve(true_is_class, score_for_class)
                        # Handle cases where precision/recall curves might be degenerate
                        if len(rec) > 1 and len(prec) > 1:
                            # Sort recall in ascending order for AUC calculation
                            order = np.argsort(rec)
                            pr_auc = auc(rec[order], prec[order])
                        else:
                            pr_auc = np.nan # Cannot compute AUC if curve is invalid
                    except Exception as e:
                        logger.warning(f"Could not calculate PR AUC for class {class_name}: {e}")
                        pr_auc = np.nan # Ensure NaN on error
                elif len(np.unique(true_is_class)) <= 1:
                     logger.debug(f"Skipping AUC calculation for class {class_name}: only one class present in y_true.")

            all_roc_aucs.append(roc_auc)
            all_pr_aucs.append(pr_auc)
            # per_class_metrics[class_name]['roc_auc'] = roc_auc
            # per_class_metrics[class_name]['pr_auc'] = pr_auc

        # --- Calculate Macro Averages ---
        # Use nanmean to ignore NaN values (e.g., from AUC calculations that failed)
        metrics['macro_avg'] = {
            'precision': float(np.nanmean(all_precisions)),
            'recall': float(np.nanmean(all_recalls)),
            'specificity': float(np.nanmean(all_specificities)),
            'f1': float(np.nanmean(all_f1s)),
            'roc_auc': float(np.nanmean(all_roc_aucs)) if can_compute_auc else np.nan,
            'pr_auc': float(np.nanmean(all_pr_aucs)) if can_compute_auc else np.nan
        }

        # metrics['per_class'] = per_class_metrics # Optionally include per-class details

        # Log summary
        logger.debug(f"Computed Metrics: Acc={metrics['accuracy']:.4f}, "
                     f"Macro F1={metrics['macro_avg']['f1']:.4f}, "
                     f"Macro ROC AUC={metrics['macro_avg']['roc_auc']:.4f}, "
                     f"Macro PR AUC={metrics['macro_avg']['pr_auc']:.4f}")

        return metrics

    def _save_results(self,
                      results_data: Dict[str, Any],
                      method_name: str,
                      params: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves evaluation results (metrics, history, parameters) to JSON and updates a summary CSV.

        Args:
            results_data (Dict[str, Any]): Dictionary containing the results to save.
                                            Expected to have keys like 'accuracy', 'macro_avg', etc.
            method_name (str): Name of the method that generated the results (e.g., 'single_eval').
            params (Optional[Dict[str, Any]]): Dictionary of parameters used for the run,
                                                included in the filename and summary.
        """
        params = params or {}
        # Create results filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        params_str = '_'.join([f"{k}={v}" for k, v in sorted(params.items()) if k != 'self'])
        filename_base = f"{method_name}_{params_str}_{timestamp}" if params_str else f"{method_name}_{timestamp}"
        json_filepath = self.results_dir / f"{filename_base}.json"
        csv_filepath = self.results_dir / 'summary_results.csv'

        # --- Save detailed JSON ---
        try:
            # Custom serializer to handle numpy types and NaN
            def json_serializer(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj) if not np.isnan(obj) else None # Convert NaN to None
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, datetime):
                     return obj.isoformat()
                # Add other types if needed
                return str(obj) # Default fallback

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=4, default=json_serializer)
            logger.info(f"Detailed results saved to: {json_filepath}")
        except Exception as e:
            logger.error(f"Failed to save detailed results to JSON {json_filepath}: {e}", exc_info=True)

        # --- Prepare and save summary CSV ---
        try:
            summary = {
                'method': method_name,
                'timestamp': timestamp,
                'model_type': self.model_type,
                'dataset_name': self.dataset_handler.root_path.name,
                'dataset_structure': self.dataset_handler.structure.value,
                'accuracy': results_data.get('accuracy', results_data.get('mean_test_accuracy', np.nan)), # Handle different keys
                **{f"macro_{k}": v for k, v in results_data.get('macro_avg', {}).items()}, # Flatten macro averages
                **params # Add run parameters
            }
            # Convert potential NaNs to None or empty string for CSV
            for key, value in summary.items():
                if isinstance(value, float) and np.isnan(value):
                    summary[key] = None # Or ''

            df_summary = pd.DataFrame([summary])

            if csv_filepath.exists():
                # Append without header
                df_summary.to_csv(csv_filepath, mode='a', header=False, index=False, encoding='utf-8')
            else:
                # Create new file with header
                df_summary.to_csv(csv_filepath, mode='w', header=True, index=False, encoding='utf-8')
            logger.info(f"Summary results updated in: {csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to save summary results to CSV {csv_filepath}: {e}", exc_info=True)

    def _get_targets_from_dataset(self, dataset: Dataset) -> np.ndarray:
        """Extracts target labels from various PyTorch Dataset types."""
        if isinstance(dataset, Subset):
            # Get targets from the original dataset using subset indices
            original_targets = np.array(dataset.dataset.targets)
            return original_targets[dataset.indices]
        elif isinstance(dataset, TransformedSubset):
            # Use the custom targets property we added
            return np.array(dataset.targets)
        elif isinstance(dataset, datasets.ImageFolder):
            # Directly access targets from ImageFolder
            return np.array(dataset.targets)
        elif isinstance(dataset, SkorchDataset):
             # If it's already a SkorchDataset, access its 'y' attribute
             if dataset.y is None:
                 raise ValueError("SkorchDataset has y=None, cannot extract targets.")
             return np.array(dataset.y)
        elif isinstance(dataset, ConcatDataset):
            # Handle ConcatDataset used in nested CV FIXED case
            all_targets = []
            for subset in dataset.datasets:
                # Recursively call this function on each subset
                all_targets.append(self._get_targets_from_dataset(subset))
            return np.concatenate(all_targets)
        else:
            # Fallback: Try iterating (slow) - use only if necessary
            logger.warning(f"Attempting slow target extraction by iterating dataset type {type(dataset)}")
            targets = []
            try:
                 # Use a DataLoader to iterate efficiently if possible
                 loader = DataLoader(dataset, batch_size=self.model_adapter.batch_size, shuffle=False, num_workers=0)
                 for _, y_batch in loader:
                     targets.append(y_batch.cpu().numpy())
                 return np.concatenate(targets)
            except Exception as e:
                 logger.error(f"Failed to extract targets by iteration: {e}")
                 raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)}")

    def _load_dataset_to_numpy(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Loads features and labels from a PyTorch Dataset into NumPy arrays."""
        logger.info(f"Loading dataset of type {type(dataset)} (length {len(dataset)}) into NumPy arrays...")
        # Use DataLoader for efficient batch loading
        loader = DataLoader(dataset, batch_size=self.model_adapter.batch_size * 2,  # Maybe larger batch for loading
                            shuffle=False, num_workers=0)  # Use num_workers=0 for stability

        all_features = []
        all_labels = []
        i = 0
        total = len(loader)
        for features, labels in loader:
            i += 1
            logger.debug(f"Loading batch {i}/{total}")
            all_features.append(features.cpu().numpy())  # Move to CPU before numpy
            all_labels.append(labels.cpu().numpy())

        X_np = np.concatenate(all_features, axis=0)
        y_np = np.concatenate(all_labels, axis=0)
        logger.info(f"Finished loading. X shape: {X_np.shape}, y shape: {y_np.shape}")
        return X_np, y_np

    def non_nested_grid_search(self,
                               param_grid: Dict[str, List],
                               cv: int = 5,
                               n_iter: Optional[int] = None,
                               method: str = 'grid',
                               scoring: str = 'accuracy',
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Performs non-nested hyperparameter search (GridSearchCV or RandomizedSearchCV)
        using the training and validation sets defined by the DatasetHandler.

        Tunes hyperparameters on the training set using cross-validation internally
        (or could use the explicit validation set if cv=None, though standard practice uses CV).
        Evaluates the best found model on the separate test set.

        Args:
            param_grid (Dict[str, List]): Dictionary defining the hyperparameter grid or distributions.
            cv (int): Number of cross-validation folds to use within the search on the training data.
            n_iter (Optional[int]): Number of parameter settings to sample for RandomizedSearchCV.
                                    Required if method='random'. Ignored if method='grid'.
            method (str): Search method: 'grid' (GridSearchCV) or 'random' (RandomizedSearchCV).
            scoring (str): Scorer to use for evaluating parameters during search (e.g., 'accuracy', 'f1_macro').
            save_results (bool): Whether to save the results (metrics, best params) to files.

        Returns:
            Dict[str, Any]: Dictionary containing search results, best parameters, best score,
                            CV results details, and evaluation metrics on the test set.

        Raises:
            ValueError: If method is invalid or n_iter is missing for random search.
            RuntimeError: If required dataset splits are missing.
        """
        method_lower = method.lower()
        logger.info(f"Performing non-nested '{method_lower}' search with {cv}-fold CV on training data.")
        logger.info(f"Parameter Grid: {param_grid}")
        logger.info(f"Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None:
            raise ValueError("n_iter must be specified for method='random'.")
        if method_lower not in ['grid', 'random']:
             raise ValueError(f"Unsupported search method: '{method}'. Choose 'grid' or 'random'.")

        # --- Get Data ---
        # Search is performed on the training data, using internal CV for validation.
        train_dataset = self.dataset_handler.get_train_dataset() # PyTorch Dataset
        test_dataset = self.dataset_handler.get_test_dataset()  # PyTorch Dataset
        if train_dataset is None or test_dataset is None:
            raise RuntimeError("Required train or test dataset is missing.")

        logger.info("Loading training data into memory for GridSearchCV...")
        try:
            X_np_train, y_np_train = self._load_dataset_to_numpy(train_dataset)
        except Exception as e:
            logger.error(f"Failed to load training data to NumPy: {e}", exc_info=True)
            raise RuntimeError("Could not load training data to NumPy for CV.") from e

        # --- Setup Search ---
        # Clone the base estimator to avoid modifying the pipeline's main adapter
        estimator = clone(self.model_adapter)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        SearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        search_kwargs = {
            'estimator': estimator,
            'cv': cv_splitter,
            'scoring': scoring,
            'n_jobs': 1, # Often best for GPU-based training
            'verbose': 3, # Show progress
            'return_train_score': True,
            'refit': True # Refit the best estimator on the whole training data
        }
        if method_lower == 'grid':
            search_kwargs['param_grid'] = param_grid
        else:
            search_kwargs['param_distributions'] = param_grid
            search_kwargs['n_iter'] = n_iter
            search_kwargs['random_state'] = RANDOM_SEED

        search = SearchClass(**search_kwargs)

        # --- Run Search ---
        # Skorch/sklearn handle dataset input directly
        logger.info(f"Fitting {SearchClass.__name__} on the training dataset...")

        # Wrap the PyTorch Dataset in SliceDataset before passing to sklearn fit
        # SkorchDataset might also work, but SliceDataset is often recommended for compatibility.
        logger.info(f"Fitting {SearchClass.__name__} on the training dataset...")
        search.fit(X_np_train, y=y_np_train)
        logger.info(f"Search completed.")

        # --- Collect Results ---
        results = {
            'method': f"non_nested_{method_lower}_search",
            'params': {'cv': cv, 'n_iter': n_iter, 'method': method, 'scoring': scoring},
            'best_params': search.best_params_,
            'best_score': search.best_score_, # Score on CV validation folds for best params
            'cv_results': search.cv_results_, # Contains detailed CV performance
        }

        # --- Evaluate Best Model on Test Set ---
        logger.info(f"Evaluating best model (params: {search.best_params_}) on the test set...")
        best_estimator = search.best_estimator_

        logger.info("Loading test data into memory for evaluation...")
        try:
            X_np_test, y_true_test = self._load_dataset_to_numpy(test_dataset)
        except Exception as e:
            logger.error(f"Failed to load test data to NumPy: {e}", exc_info=True)
            raise RuntimeError("Could not load test data to NumPy for evaluation.") from e

        y_pred_test = best_estimator.predict(X_np_test)
        try:
            y_score_test = best_estimator.predict_proba(X_np_test)
        except AttributeError:
             y_score_test = None


        test_metrics = self._compute_metrics(y_true_test, y_pred_test, y_score_test)
        results['test_set_evaluation'] = test_metrics
        logger.info(f"Test Set Evaluation: Accuracy={test_metrics['accuracy']:.4f}, "
                    f"Macro F1={test_metrics['macro_avg']['f1']:.4f}")

        # Add top-level metrics for saving compatibility
        results['accuracy'] = test_metrics.get('accuracy', np.nan)
        results['macro_avg'] = test_metrics.get('macro_avg', {})

        # --- Save Results ---
        if save_results:
            save_params = {'cv': cv, 'method': method_lower, 'scoring': scoring}
            if n_iter: save_params['n_iter'] = n_iter
            self._save_results(
                results,
                f"non_nested_{method_lower}_search",
                params=save_params
            )

        logger.info(f"Non-nested {method_lower} search finished. Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"Best parameters found: {search.best_params_}")

        # Update the main pipeline model adapter with the best found estimator ONLY if desired
        # self.model_adapter = best_estimator
        # logger.info("Pipeline's model adapter updated with the best estimator found during search.")

        return results


    def nested_grid_search(self,
                           param_grid: Dict[str, List],
                           outer_cv: int = 5,
                           inner_cv: int = 3,
                           n_iter: Optional[int] = None,
                           method: str = 'grid',
                           scoring: str = 'accuracy',
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Performs nested cross-validation for unbiased performance estimation of the
        hyperparameter tuning process.

        Uses an outer loop (cross_validate) to split the data. In each outer fold,
        an inner loop (GridSearchCV or RandomizedSearchCV) performs hyperparameter tuning
        on the outer loop's training split. The best model from the inner loop is then
        evaluated on the outer loop's validation/test split.

        - For FLAT datasets: Uses the entire dataset for nested CV.
        - For FIXED datasets: Adapts the process. It performs the inner hyperparameter
          search on the combined 'train' + 'validation' sets (using inner_cv). The single
          best model found is then evaluated *once* on the predefined 'test' set. This
          estimates the performance of the *chosen* model, not the variability of the
          selection process itself, due to the fixed test set.

        Args:
            param_grid (Dict[str, List]): Hyperparameter grid/distributions for the inner search.
            outer_cv (int): Number of folds for the outer cross-validation loop.
            inner_cv (int): Number of folds for the inner cross-validation loop (tuning).
            n_iter (Optional[int]): Number of iterations for RandomizedSearchCV in the inner loop. Required if method='random'.
            method (str): Inner search method: 'grid' or 'random'.
            scoring (str): Scorer used for both inner tuning and outer evaluation.
            save_results (bool): Whether to save the results.

        Returns:
            Dict[str, Any]: Dictionary containing the results. Structure differs slightly
                            based on dataset structure (FLAT vs. FIXED). Includes outer loop scores,
                            (potentially) best parameters per fold, and average performance.

        Raises:
            ValueError: If method is invalid or n_iter is missing for random search.
            RuntimeError: If required dataset splits are missing.
        """
        method_lower = method.lower()
        logger.info(f"Performing nested '{method_lower}' search.")
        logger.info(f"  Outer CV folds: {outer_cv}, Inner CV folds: {inner_cv}")
        logger.info(f"  Parameter Grid: {param_grid}")
        logger.info(f"  Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None:
            raise ValueError("n_iter must be specified for method='random'.")
        if method_lower not in ['grid', 'random']:
             raise ValueError(f"Unsupported search method: '{method}'. Choose 'grid' or 'random'.")

        # --- Setup Inner Search Object ---
        # Base estimator (will be cloned for each outer fold)
        base_estimator = clone(self.model_adapter)
        base_estimator.set_params(train_split=None) # Ensure no internal split

        inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_SEED)

        InnerSearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        inner_search_kwargs = {
            'estimator': base_estimator,
            'cv': inner_cv_splitter,
            'scoring': scoring,
            'n_jobs': 1,
            'verbose': 1, # Verbosity for inner loops
            'refit': True # Refit best inner model on inner training data
        }
        if method_lower == 'grid':
            inner_search_kwargs['param_grid'] = param_grid
        else:
            inner_search_kwargs['param_distributions'] = param_grid
            inner_search_kwargs['n_iter'] = n_iter
            inner_search_kwargs['random_state'] = RANDOM_SEED

        inner_search = InnerSearchClass(**inner_search_kwargs)

        # --- Select Data Based on Structure ---
        if self.dataset_handler.structure == DatasetStructure.FLAT:
            logger.info("Using full dataset for standard nested CV (FLAT structure).")
            dataset_for_cv = self.dataset_handler.get_full_dataset()
            if dataset_for_cv is None:
                raise RuntimeError("Full dataset required for nested CV on FLAT structure, but not available.")
            outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED)
            run_standard_nested_cv = True

        elif self.dataset_handler.structure == DatasetStructure.FIXED:
            logger.warning("Adapting nested CV for FIXED dataset structure:")
            logger.warning("  Inner search (tuning) runs on 'train+validation' sets.")
            logger.warning("  Final evaluation uses the single best model on the fixed 'test' set.")
            train_dataset = self.dataset_handler.get_train_dataset()
            val_dataset = self.dataset_handler.get_val_dataset()
            test_dataset = self.dataset_handler.get_test_dataset()

            if train_dataset is None or test_dataset is None:
                raise RuntimeError("Required train/test datasets missing for FIXED structure nested CV adaptation.")

            # Combine train and validation Datasets for inner search fitting
            # Note: Creating a ConcatDataset is the proper PyTorch way
            datasets_to_combine = [train_dataset]
            if val_dataset:
                datasets_to_combine.append(val_dataset)

            if len(datasets_to_combine) > 1:
                tuning_dataset = torch.utils.data.ConcatDataset(datasets_to_combine)
                logger.info(f"Combined train ({len(train_dataset)}) and validation ({len(val_dataset) if val_dataset else 0}) sets "
                            f"into a single dataset of size {len(tuning_dataset)} for hyperparameter tuning.")
            else:
                 tuning_dataset = train_dataset # Only train set available
                 logger.info("Using only the training set for hyperparameter tuning (no validation set available/used).")

            run_standard_nested_cv = False # Use the adapted approach

        else: # Should not happen
            raise RuntimeError(f"Unknown dataset structure: {self.dataset_handler.structure}")


        # --- Execute Nested CV or Adapted FIXED Workflow ---
        results = {
             'method': f"nested_{method_lower}_search",
             'params': {'outer_cv': outer_cv, 'inner_cv': inner_cv, 'n_iter': n_iter, 'method': method_lower, 'scoring': scoring},
        }

        if run_standard_nested_cv:
            # Standard Nested CV using cross_validate
            logger.info(f"Running standard nested CV using cross_validate with {outer_cv} outer folds...")
            # Define scorers needed for the outer loop evaluation
            scoring_dict = {
                'accuracy': make_scorer(accuracy_score),
                'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
                'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
                'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
                # Note: AUC scorers require predict_proba, which cross_validate doesn't easily support
                # for the final score calculation across folds without custom handling.
            }

            cv_results = cross_validate(
                inner_search, # The inner search object is the estimator for the outer loop
                dataset_for_cv, y=None, # Pass dataset directly
                cv=outer_cv_splitter,
                scoring=scoring_dict,
                return_estimator=True, # Get the fitted inner search object from each fold
                n_jobs=1, # Usually required for nested CV with complex estimators
                verbose=2 # Show progress of outer folds
            )
            logger.info("Nested cross-validation finished.")

            # Process results from cross_validate
            results['outer_cv_scores'] = {k: v.tolist() for k, v in cv_results.items() if k.startswith('test_')}
            results['mean_test_accuracy'] = float(np.mean(cv_results['test_accuracy']))
            results['std_test_accuracy'] = float(np.std(cv_results['test_accuracy']))
            results['mean_test_precision_macro'] = float(np.mean(cv_results['test_precision_macro']))
            results['mean_test_recall_macro'] = float(np.mean(cv_results['test_recall_macro']))
            results['mean_test_f1_macro'] = float(np.mean(cv_results['test_f1_macro']))
            # Extract best params found in each *inner* loop
            try:
                 results['best_params_per_fold'] = [est.best_params_ for est in cv_results['estimator']]
            except Exception as e:
                 logger.warning(f"Could not extract best_params_per_fold: {e}")
                 results['best_params_per_fold'] = "Error extracting"

            # Add top-level metrics for saving compatibility
            results['accuracy'] = results['mean_test_accuracy']
            results['macro_avg'] = {
                'precision': results.get('mean_test_precision_macro', np.nan),
                'recall': results.get('mean_test_recall_macro', np.nan),
                'f1': results.get('mean_test_f1_macro', np.nan),
                'roc_auc': np.nan, # Not computed by cross_validate easily
                'pr_auc': np.nan   # Not computed by cross_validate easily
            }
            logger.info(f"Nested CV Average Performance ({scoring}): "
                        f"{results['mean_test_accuracy']:.4f} +/- {results['std_test_accuracy']:.4f} (Accuracy)")


        else: # Adapted workflow for FIXED dataset
            logger.info(f"Running adapted nested CV for FIXED structure...")
            logger.info(f"  Step 1: Tuning hyperparameters using inner CV on train+validation data ({inner_cv} folds)...")
            inner_search.fit(tuning_dataset, y=None)
            logger.info(f"  Hyperparameter tuning finished. Best params found: {inner_search.best_params_}")
            logger.info(f"  Best score during tuning ({scoring}): {inner_search.best_score_:.4f}")

            results['best_params'] = inner_search.best_params_
            results['best_tuning_score'] = inner_search.best_score_
            results['inner_cv_results'] = inner_search.cv_results_

            # Step 2: Evaluate the single best model on the fixed test set
            logger.info(f"  Step 2: Evaluating the best model on the fixed test set...")
            best_estimator = inner_search.best_estimator_
            best_estimator.set_params(train_split=None) # Ensure no split during prediction

            y_pred_test = best_estimator.predict(test_dataset)
            try:
                y_score_test = best_estimator.predict_proba(test_dataset)
            except AttributeError:
                 y_score_test = None

            # Extract true labels from test dataset
            y_true_test_list = []
            test_loader = DataLoader(test_dataset, batch_size=self.model_adapter.batch_size, shuffle=False)
            for _, y_batch in test_loader:
                 y_true_test_list.append(y_batch.numpy())
            y_true_test = np.concatenate(y_true_test_list)

            test_metrics = self._compute_metrics(y_true_test, y_pred_test, y_score_test)
            results['fixed_test_set_evaluation'] = test_metrics
            logger.info(f"  Fixed Test Set Evaluation: Accuracy={test_metrics['accuracy']:.4f}, "
                        f"Macro F1={test_metrics['macro_avg']['f1']:.4f}")

            # Add top-level metrics for saving compatibility
            results['accuracy'] = test_metrics.get('accuracy', np.nan)
            results['macro_avg'] = test_metrics.get('macro_avg', {})

            # Update the main pipeline model adapter if desired
            # self.model_adapter = best_estimator
            # logger.info("Pipeline's model adapter updated with the best estimator from FIXED nested search.")


        # --- Save Results ---
        if save_results:
            save_params = {'outer_cv': outer_cv if run_standard_nested_cv else 'fixed_test',
                           'inner_cv': inner_cv, 'method': method_lower, 'scoring': scoring}
            if n_iter: save_params['n_iter'] = n_iter
            self._save_results(
                results,
                f"nested_{method_lower}_search",
                params=save_params
            )

        return results


    def cv_model_evaluation(self, cv: int = 5, save_results: bool = True) -> Dict[str, Any]:
        """
        Performs standard cross-validation for model evaluation using the current
        pipeline settings (model type, hyperparameters).

        Trains and evaluates the *same* model configuration on different folds of the data
        to assess robustness and get a performance estimate with confidence intervals.

        This method is only suitable for FLAT dataset structures where the entire dataset
        can be used for cross-validation.

        Args:
            cv (int): Number of cross-validation folds.
            save_results (bool): Whether to save the results.

        Returns:
            Dict[str, Any]: Dictionary containing CV results (scores per fold, averages, std dev).

        Raises:
            ValueError: If called on a dataset with a FIXED structure.
            RuntimeError: If the dataset is not available.
        """
        logger.info(f"Performing {cv}-fold cross-validation for model evaluation.")

        # --- Check Compatibility ---
        if self.dataset_handler.structure == DatasetStructure.FIXED:
            raise ValueError("Standard CV model evaluation is not suitable for FIXED dataset structures "
                             "with a predefined test set. Use 'single_eval' or nested CV adaptation.")

        # --- Get Data ---
        full_dataset = self.dataset_handler.get_full_dataset() # Raises error if not FLAT or not loaded
        logger.info(f"Using full dataset ({len(full_dataset)}) for {cv}-fold CV.")

        # --- Setup CV ---
        # Use the pipeline's current model adapter configuration
        estimator = clone(self.model_adapter)
        estimator.set_params(train_split=None) # Ensure no internal splitting during CV

        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

        # Define scorers - similar to nested CV, AUC is problematic here
        scoring_dict = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        }

        # --- Run Cross-Validation ---
        logger.info("Running cross_validate...")
        cv_results = cross_validate(
            estimator,
            full_dataset, y=None, # Pass dataset directly
            cv=cv_splitter,
            scoring=scoring_dict,
            return_train_score=False, # Typically don't need train scores for final eval
            n_jobs=1,
            verbose=2
        )
        logger.info("Cross-validation finished.")

        # --- Process and Save Results ---
        results = {
             'method': 'cv_model_evaluation',
             'params': {'cv': cv},
             'cv_scores': {k: v.tolist() for k, v in cv_results.items() if k.startswith('test_')},
             'fit_time_mean': float(np.mean(cv_results['fit_time'])),
             'score_time_mean': float(np.mean(cv_results['score_time'])),
             'mean_test_accuracy': float(np.mean(cv_results['test_accuracy'])),
             'std_test_accuracy': float(np.std(cv_results['test_accuracy'])),
             'mean_test_precision_macro': float(np.mean(cv_results['test_precision_macro'])),
             'mean_test_recall_macro': float(np.mean(cv_results['test_recall_macro'])),
             'mean_test_f1_macro': float(np.mean(cv_results['test_f1_macro'])),
        }

        # Add top-level metrics for saving compatibility
        results['accuracy'] = results['mean_test_accuracy']
        results['macro_avg'] = {
            'precision': results.get('mean_test_precision_macro', np.nan),
            'recall': results.get('mean_test_recall_macro', np.nan),
            'f1': results.get('mean_test_f1_macro', np.nan),
            'roc_auc': np.nan, # Not computed
            'pr_auc': np.nan  # Not computed
        }

        if save_results:
            self._save_results(results, "cv_model_evaluation", params={'cv': cv})

        logger.info(f"CV Evaluation Summary (Avg over {cv} folds):")
        logger.info(f"  Accuracy: {results['mean_test_accuracy']:.4f} +/- {results['std_test_accuracy']:.4f}")
        logger.info(f"  Macro F1: {results['mean_test_f1_macro']:.4f} +/- {float(np.std(cv_results['test_f1_macro'])):.4f}")
        logger.info(f"  Note: AUC metrics are not computed by default in cross_validate.")

        return results

    def single_train(self,
                     max_epochs: Optional[int] = None,
                     lr: Optional[float] = None,
                     batch_size: Optional[int] = None,
                     early_stopping_patience: Optional[int] = None,
                     save_model: bool = True) -> Dict[str, Any]:
        """
        Performs a single training run using the training and validation sets
        defined by the DatasetHandler. Skorch's internal train_split is used
        to create the validation set from the combined train+val data.

        Args:
            max_epochs (Optional[int]): Override the default max_epochs for this run.
            lr (Optional[float]): Override the default learning rate for this run.
            batch_size (Optional[int]): Override the default batch size for this run.
            early_stopping_patience (Optional[int]): Override the default early stopping patience.
            save_model (bool): If True, saves the state_dict of the best trained model.

        Returns:
            Dict[str, Any]: Dictionary containing training history, best epoch/loss,
                            and path to the saved model if `save_model` is True.

        Raises:
            RuntimeError: If train dataset is missing or target extraction fails.
        """
        logger.info("Starting single training run...")

        # --- Get Data ---
        train_dataset = self.dataset_handler.get_train_dataset()
        val_dataset = self.dataset_handler.get_val_dataset()  # Can be None if val_split_ratio=0
        if train_dataset is None:
            raise RuntimeError("Training dataset is missing for single_train.")

        # Combine train and validation datasets for skorch internal splitting
        datasets_to_fit = [train_dataset]
        if val_dataset:
            datasets_to_fit.append(val_dataset)
            logger.info(f"Combining train ({len(train_dataset)}) and validation ({len(val_dataset)}) sets "
                        f"for skorch internal split (total: {sum(len(d) for d in datasets_to_fit)} samples).")
            combined_dataset = ConcatDataset(datasets_to_fit)
        else:
            # If no explicit val_dataset, just use train_dataset for internal split
            logger.info(f"Using training set ({len(train_dataset)} samples) for skorch internal split.")
            combined_dataset = train_dataset

        # --- Configure Model Adapter for this Run ---
        # Use method args to override pipeline defaults if provided
        current_params = self.model_adapter.get_params()  # Store original params if needed later
        params_to_set = {}
        if max_epochs is not None: params_to_set['max_epochs'] = max_epochs
        if lr is not None: params_to_set['lr'] = lr
        if batch_size is not None: params_to_set['batch_size'] = batch_size
        # Update callback parameters if needed
        if early_stopping_patience is not None:
            # Check if early stopping callback exists by name
            callback_names = [name for name, _ in self.model_adapter.callbacks]
            if 'early_stopping' in callback_names:
                params_to_set['callbacks__early_stopping__patience'] = early_stopping_patience
            else:
                logger.warning("Tried to set early stopping patience, but 'early_stopping' callback not found.")

        if params_to_set:
            logger.info(f"Overriding model adapter parameters for this run: {params_to_set}")
            self.model_adapter.set_params(**params_to_set)

        # --- Prepare Fit Parameters ---
        # No need to pass X_val/y_val, skorch uses internal train_split
        fit_params = {}

        # --- Train Model ---
        # Extract combined targets for skorch internal dataset handling if y is needed
        # (Skorch might need y depending on how train_split handles the Dataset)
        try:
            y_combined = self._get_targets_from_dataset(combined_dataset)
            logger.debug(f"Extracted combined targets (shape={y_combined.shape}) for skorch fit.")
        except Exception as e:
            logger.error(f"Failed to get targets from combined dataset for single_train: {e}")
            # Fallback or raise error? Raise for now.
            raise RuntimeError("Could not extract targets for single_train fit") from e

        # Pass the combined dataset and extracted targets. Skorch's train_split will handle it.
        self.model_adapter.fit(combined_dataset, y=y_combined, **fit_params)

        # --- Collect Results ---
        history = self.model_adapter.history  # Skorch history object

        best_epoch_info = {}
        valid_loss_key = 'valid_loss'  # Default monitor, should be present now
        es_callback = dict(self.model_adapter.callbacks_).get('early_stopping')
        if es_callback:
            monitor_key = getattr(es_callback, 'monitor', 'valid_loss')
            if history and monitor_key in history[0]:
                valid_loss_key = monitor_key
            elif history:
                logger.warning(
                    f"Early stopping monitor key '{monitor_key}' not found in history[0]. Defaulting check to 'valid_loss'. History keys: {list(history[0].keys())}")

        # Check if validation was run and history is not empty
        validation_was_run = history and valid_loss_key in history[0]

        if validation_was_run:
            try:
                scores = [epoch_hist[valid_loss_key] for epoch_hist in history]
                lower_is_better = valid_loss_key.endswith('_loss')
                best_epoch_idx = np.argmin(scores) if lower_is_better else np.argmax(scores)
                best_epoch_hist = history[int(best_epoch_idx)]
                actual_best_epoch_num = int(best_epoch_idx) + 1

                best_epoch_info = {
                    'best_epoch': actual_best_epoch_num,
                    'best_valid_metric_value': float(best_epoch_hist.get(valid_loss_key, np.nan)),
                    'valid_metric_name': valid_loss_key,
                    'train_loss_at_best': float(best_epoch_hist.get('train_loss', np.nan)),
                }
                logger.info(f"Training finished. Best validation performance at Epoch {best_epoch_info['best_epoch']} "
                            f"({valid_loss_key}={best_epoch_info['best_valid_metric_value']:.4f})")
            except Exception as e:
                logger.error(f"Error processing history to find best epoch: {e}", exc_info=True)
                validation_was_run = False

        if not validation_was_run:
            if history:
                last_epoch_hist = history[-1]
                last_epoch_num = len(history)
            else:
                last_epoch_hist = {}
                last_epoch_num = 0
                logger.error("Training history is empty after fit completed.")

            best_epoch_info = {
                'best_epoch': last_epoch_num,
                'best_valid_metric_value': np.nan,
                'valid_metric_name': valid_loss_key,
                'train_loss_at_best': float(last_epoch_hist.get('train_loss', np.nan)),
            }
            logger.warning(
                f"Could not determine best epoch based on validation metric '{valid_loss_key}'. Reporting last epoch stats.")
            if last_epoch_num > 0:
                logger.info(f"Training finished at Epoch {best_epoch_info['best_epoch']} "
                            f"(Train Loss={best_epoch_info['train_loss_at_best']:.4f})")

        # Prepare results dict
        es_patience = np.nan
        try:
            es_callback_instance = dict(self.model_adapter.callbacks_).get('early_stopping')
            if es_callback_instance:
                es_patience = es_callback_instance.patience
        except Exception:
            pass  # Ignore if callback access fails

        results = {
            'method': 'single_train',
            'params': {
                'lr': self.model_adapter.lr,
                'max_epochs': self.model_adapter.max_epochs,
                'batch_size': self.model_adapter.batch_size,
                'early_stopping_patience': es_patience
            },
            'training_history': history.to_list(),
            **best_epoch_info
        }

        # --- Save Model ---
        if save_model:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            val_metric_val = results.get('best_valid_metric_value', np.nan)
            val_metric_name = results.get('valid_metric_name', 'unknown').replace('_', '-')
            val_metric_str = f"val_{val_metric_name}{val_metric_val:.4f}" if not np.isnan(val_metric_val) else "no_val"
            model_filename = f"{self.model_type}_epoch{results.get('best_epoch', 0)}_{val_metric_str}_{timestamp}.pt"
            model_path = self.results_dir / model_filename
            try:
                # EarlyStopping(load_best=True) should load the best weights, save them
                torch.save(self.model_adapter.module_.state_dict(), model_path)
                logger.info(f"Model state_dict saved to: {model_path}")
                results['saved_model_path'] = str(model_path)
            except Exception as e:
                logger.error(f"Failed to save model to {model_path}: {e}", exc_info=True)
                results['saved_model_path'] = None

        # Add dummy metrics for saving compatibility
        results['accuracy'] = np.nan
        results['macro_avg'] = {}

        # Save run info
        self._save_results(results, "single_train", params=results['params'])

        # Optionally restore original parameters if needed
        # self.model_adapter.set_params(**current_params)

        return results


    def single_eval(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluates the currently loaded/trained model adapter on the test set.

        Computes and returns a comprehensive set of metrics.

        Args:
            save_results (bool): Whether to save the evaluation metrics to files.

        Returns:
            Dict[str, Any]: Dictionary containing the evaluation metrics computed on the test set.

        Raises:
            RuntimeError: If the test dataset is missing or the model hasn't been trained/loaded.
        """
        logger.info("Starting model evaluation on the test set...")

        # --- Check Model State ---
        if not self.model_adapter.initialized_:
             raise RuntimeError("Model adapter has not been initialized. Train or load a model first.")
        if not hasattr(self.model_adapter, 'module_') or not isinstance(self.model_adapter.module_, nn.Module):
             raise RuntimeError("Model module (nn.Module) not found in the adapter. Train or load a model first.")


        # --- Get Data ---
        test_dataset = self.dataset_handler.get_test_dataset()
        if test_dataset is None:
            raise RuntimeError("Test dataset is missing for single_eval.")
        logger.info(f"Evaluating on {len(test_dataset)} test samples.")

        # Ensure model is in evaluation mode (skorch usually handles this, but good practice)
        self.model_adapter.module_.eval()

        # --- Make Predictions ---
        logger.debug("Generating predictions on the test set...")
        # Pass Dataset directly to predict/predict_proba
        y_pred_test = self.model_adapter.predict(test_dataset)
        logger.debug("Generating probabilities on the test set...")
        try:
            y_score_test = self.model_adapter.predict_proba(test_dataset)
        except AttributeError:
             logger.warning("Model adapter does not support predict_proba. AUC metrics will be unavailable.")
             y_score_test = None
        logger.debug("Predictions and probabilities generated.")


        # --- Get True Labels ---
        logger.debug("Extracting true labels from the test set...")
        y_true_test_list = []
        # Use a DataLoader for efficient iteration
        test_loader = DataLoader(test_dataset, batch_size=self.model_adapter.batch_size, shuffle=False, num_workers=NUM_WORKERS)
        for _, y_batch in test_loader:
             y_true_test_list.append(y_batch.cpu().numpy()) # Ensure data is on CPU for numpy conversion
        y_true_test = np.concatenate(y_true_test_list)
        logger.debug("True labels extracted.")

        # --- Compute Metrics ---
        logger.debug("Computing evaluation metrics...")
        metrics = self._compute_metrics(y_true_test, y_pred_test, y_score_test)
        logger.info("Evaluation metrics computed.")

        # Prepare results dict
        results = {
             'method': 'single_eval',
             'params': {}, # No specific params for evaluation itself
             **metrics # Include accuracy, macro_avg, etc. directly
        }

        # --- Save Results ---
        if save_results:
            self._save_results(metrics, "single_eval") # Pass metrics dict directly

        logger.info(f"Evaluation Summary:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Macro Precision: {metrics['macro_avg']['precision']:.4f}")
        logger.info(f"  Macro Recall: {metrics['macro_avg']['recall']:.4f}")
        logger.info(f"  Macro Specificity: {metrics['macro_avg']['specificity']:.4f}")
        logger.info(f"  Macro F1-Score: {metrics['macro_avg']['f1']:.4f}")
        if not np.isnan(metrics['macro_avg']['roc_auc']):
            logger.info(f"  Macro ROC AUC: {metrics['macro_avg']['roc_auc']:.4f}")
        if not np.isnan(metrics['macro_avg']['pr_auc']):
            logger.info(f"  Macro PR AUC: {metrics['macro_avg']['pr_auc']:.4f}")

        return results


    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Loads a trained model's state_dict from a file into the pipeline's model adapter.

        Args:
            model_path (Union[str, Path]): Path to the saved .pt file containing the model state_dict.

        Raises:
            FileNotFoundError: If the model_path does not exist.
            RuntimeError: If the model adapter hasn't been initialized or fails to load weights.
        """
        model_path = Path(model_path)
        logger.info(f"Loading model state_dict from: {model_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Ensure the skorch adapter and its underlying module are initialized
        if not self.model_adapter.initialized_:
            logger.debug("Initializing skorch adapter before loading state_dict...")
            try:
                # Initialize with dummy data or rely on skorch's internal init
                # Passing module params again ensures correct architecture
                module_kwargs = {f"module__{k}": v for k,v in self.model_adapter.module_init_kwargs.items()}
                self.model_adapter.initialize()
                # self.model_adapter.initialize_module(**module_kwargs) # More explicit module init if needed
                # self.model_adapter.initialize_optimizer()
                # self.model_adapter.initialize_criterion()

            except Exception as e:
                 logger.error(f"Failed to initialize model adapter before loading: {e}", exc_info=True)
                 raise RuntimeError("Could not initialize model adapter for loading.") from e

        if not hasattr(self.model_adapter, 'module_') or not isinstance(self.model_adapter.module_, nn.Module):
             raise RuntimeError("Model adapter is initialized but missing the internal nn.Module ('module_'). Cannot load state_dict.")


        # Load the state dict
        try:
            # Load to the correct device
            map_location = self.model_adapter.device
            state_dict = torch.load(model_path, map_location=map_location)
            logger.debug(f"State_dict loaded successfully to device '{map_location}'.")
        except Exception as e:
            logger.error(f"Failed to load state_dict from {model_path}: {e}", exc_info=True)
            raise RuntimeError(f"Error loading state_dict from {model_path}") from e

        # Load the state dict into the module
        try:
            self.model_adapter.module_.load_state_dict(state_dict)
            self.model_adapter.module_.eval() # Set to eval mode after loading
            logger.info("Model state_dict loaded successfully into the model adapter.")
        except Exception as e:
            logger.error(f"Failed to load state_dict into model: {e}", exc_info=True)
            # Provide more info if possible (e.g., mismatched keys)
            if isinstance(e, RuntimeError) and "size mismatch" in str(e):
                 logger.error("Architecture mismatch likely. Ensure loaded weights match the current model.")
            raise RuntimeError("Error loading state_dict into the model adapter module.") from e


# --- Pipeline Executor ---
from unittest.mock import MagicMock # Used for default patience value

class PipelineExecutor:
    """
    Executes a sequence of classification pipeline methods (train, eval, search)
    defined by the user.

    Handles parameter passing for each method and stores results. Checks for
    compatibility between methods and dataset structure before execution.

    Attributes:
        pipeline (ClassificationPipeline): The underlying classification pipeline instance.
        methods_to_run (List[Tuple[str, Dict]]): List of (method_name, parameters) tuples to execute.
        all_results (Dict[str, Any]): Dictionary storing results from executed methods.
    """

    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: str = 'cnn',
                 model_load_path: Optional[Union[str, Path]] = None,
                 results_dir: Union[str, Path] = 'results',
                 methods: List[Tuple[str, Dict[str, Any]]]= None,
                 # Pass pipeline config params here
                 img_size: Tuple[int, int] = (224, 224),
                 val_split_ratio: float = 0.2,
                 data_augmentation: bool = True,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10):
        """
        Initializes the PipelineExecutor.

        Args:
            dataset_path (Union[str, Path]): Path to the dataset root directory.
            model_type (str): Type of model architecture ('cnn', 'vit', 'diffusion').
            model_load_path (Optional[Union[str, Path]]): Optional path to load initial model weights.
            results_dir (Union[str, Path]): Base directory for saving results.
            methods (List[Tuple[str, Dict[str, Any]]]): A list of tuples, where each tuple contains
                                                        the name of the pipeline method to run (str)
                                                        and a dictionary of its parameters (Dict).
                                                        Example: [('single_train', {'max_epochs': 50}), ('single_eval', {})]
            img_size (Tuple[int, int]): Image size for the pipeline.
            val_split_ratio (float): Validation split ratio for the pipeline.
            data_augmentation (bool): Data augmentation flag for the pipeline.
            lr (float): Default learning rate for the pipeline's model adapter.
            max_epochs (int): Default max epochs for the pipeline's model adapter.
            batch_size (int): Default batch size for the pipeline's model adapter.
            patience (int): Default early stopping patience for the pipeline.

        Raises:
            ValueError: If method names are invalid or incompatible with dataset structure.
        """
        logger.info(f"Initializing Pipeline Executor for model '{model_type}' on dataset '{Path(dataset_path).name}'...")

        # Initialize the underlying pipeline
        self.pipeline = ClassificationPipeline(
            dataset_path=dataset_path,
            model_type=model_type,
            model_load_path=model_load_path,
            results_dir=results_dir,
            img_size=img_size,
            val_split_ratio=val_split_ratio,
            data_augmentation=data_augmentation,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience
        )

        self.methods_to_run = methods if methods is not None else []
        self.all_results: Dict[str, Any] = {}

        # Validate methods and parameters
        self._validate_methods()

        method_names = [m[0] for m in self.methods_to_run]
        logger.info(f"Executor configured to run methods: {', '.join(method_names)}")

    def _validate_methods(self) -> None:
        """
        Validates the list of methods to run for name correctness and compatibility
        with the dataset structure.

        Raises:
            ValueError: If an invalid method name is found or a method is incompatible
                        with the dataset structure (e.g., 'cv_model_evaluation' on FIXED).
        """
        valid_method_names = [
            'non_nested_grid_search',
            'nested_grid_search',
            'cv_model_evaluation',
            'single_train',
            'single_eval',
            'load_model' # Allow loading a model as a step
        ]

        dataset_structure = self.pipeline.dataset_handler.structure

        for i, (method_name, params) in enumerate(self.methods_to_run):
            if not isinstance(method_name, str):
                 raise ValueError(f"Method name at index {i} must be a string, got {type(method_name)}.")
            if not isinstance(params, dict):
                 raise ValueError(f"Parameters for method '{method_name}' at index {i} must be a dict, got {type(params)}.")

            if method_name not in valid_method_names:
                raise ValueError(f"Invalid method name '{method_name}' at index {i}. "
                                 f"Valid methods are: {', '.join(valid_method_names)}")

            # Check compatibility with dataset structure
            if method_name == 'cv_model_evaluation' and dataset_structure == DatasetStructure.FIXED:
                raise ValueError(f"Method '{method_name}' at index {i} is incompatible with FIXED dataset structure.")
            if method_name == 'get_full_dataset' and dataset_structure == DatasetStructure.FIXED:
                 # This isn't a runnable method, but illustrates the check
                 raise ValueError("Accessing full dataset is incompatible with FIXED structure.")
             # Add other compatibility checks if needed

            # Check required parameters for specific methods (basic examples)
            if method_name == 'non_nested_grid_search' or method_name == 'nested_grid_search':
                if 'param_grid' not in params:
                     raise ValueError(f"Method '{method_name}' requires 'param_grid' in its parameters.")
                search_type = params.get('method', 'grid').lower()
                if search_type == 'random' and 'n_iter' not in params:
                     raise ValueError(f"Random search ('{method_name}') requires 'n_iter' parameter.")
            if method_name == 'load_model':
                 if 'model_path' not in params:
                      raise ValueError(f"Method 'load_model' requires 'model_path' parameter.")

        logger.debug("Method validation successful.")


    def run(self) -> Dict[str, Any]:
        """
        Executes the configured sequence of pipeline methods.

        Returns:
            Dict[str, Any]: A dictionary containing the results from each executed method,
                            keyed by a unique identifier (e.g., f"{method_name}_{index}").
        """
        self.all_results = {}
        logger.info("Starting execution of pipeline methods...")

        for i, (method_name, params) in enumerate(self.methods_to_run):
            run_id = f"{method_name}_{i}" # Unique ID for storing results
            logger.info(f"--- Running Method {i+1}/{len(self.methods_to_run)}: {method_name} ---")
            logger.debug(f"Parameters for {method_name}: {params}")

            try:
                # Get the corresponding method from the pipeline instance
                pipeline_method = getattr(self.pipeline, method_name)

                # Execute the method with its parameters
                result = pipeline_method(**params)

                # Store the result
                self.all_results[run_id] = result
                logger.info(f"--- Method {method_name} completed successfully ---")

            except Exception as e:
                logger.error(f"!!! Method '{method_name}' failed at step {i+1}: {e}", exc_info=True)
                # Store the error
                self.all_results[run_id] = {"error": str(e), "traceback": logging.traceback.format_exc()}
                logger.error("!!! Pipeline execution stopped due to error.")
                # Decide whether to stop or continue? Stop for now.
                break

        logger.info("Pipeline execution finished.")
        return self.all_results

# --- Example Usage ---
if __name__ == "__main__":
    # Ensure results and logs directories exist relative to the script
    script_dir = Path(__file__).parent
    results_base_dir = script_dir / 'results'
    log_dir = script_dir / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Reconfigure logger to save log file in the script's directory
    logger = setup_logger('image_classification', log_dir / 'classification.log')
    logger.setLevel(logging.DEBUG)

    # --- Configuration ---
    # Use a publicly available dataset for demonstration if needed, e.g., CIFAR10, MNIST
    # Or create a dummy dataset structure:
    # dummy_data_path = script_dir / "data" / "dummy_fixed"
    # dummy_data_path.mkdir(parents=True, exist_ok=True)
    # (dummy_data_path / "train" / "class_a").mkdir(parents=True, exist_ok=True)
    # (dummy_data_path / "train" / "class_b").mkdir(parents=True, exist_ok=True)
    # (dummy_data_path / "test" / "class_a").mkdir(parents=True, exist_ok=True)
    # (dummy_data_path / "test" / "class_b").mkdir(parents=True, exist_ok=True)
    # # Add dummy image files (e.g., copies of a small png/jpg) to these folders
    # logger.info(f"Ensure dummy data exists at {dummy_data_path} or change dataset_path.")

    # dataset_path = dummy_data_path # Example: Use dummy fixed dataset
    dataset_path = script_dir / "../data/mini-GCD-flat" # Use the path from the original example

    # Check if dataset exists
    if not Path(dataset_path).exists():
         logger.error(f"Dataset path not found: {dataset_path}")
         logger.error("Please create the dataset or modify the 'dataset_path' variable.")
         exit()


    model_type = "cnn"  # Options: 'cnn', 'vit', 'diffusion'

    # Define Hyperparameter Search Space (example)
    param_grid_search = {
        'lr': [0.001, 0.0005],
        'optimizer__weight_decay': [0.01, 0.001],
        # 'batch_size': [16, 32], # Batch size changes require careful memory management
        # 'max_epochs': [10, 15] # Can tune epochs too
    }

    # --- Define Method Sequence ---
    # Example 1: Single Train and Eval
    methods_sequence_1 = [
        ('single_train', {'max_epochs': 5, 'save_model': True}), # Short training for demo
        ('single_eval', {'save_results': True}),
    ]

    # Example 2: Non-Nested Grid Search then Eval
    methods_sequence_2 = [
        ('non_nested_grid_search', {
            'param_grid': param_grid_search,
            'cv': 3, # Inner CV folds
            'method': 'grid',
            'scoring': 'accuracy',
            'save_results': True
        }),
         # The search doesn't automatically update the pipeline's main model,
         # so single_eval would run with the *original* hyperparameters unless
         # we explicitly load the best model found or update the adapter.
         # For now, just run the search. Add a load step if needed.
         # ('single_eval', {'save_results': True})
    ]

    # Example 3: Nested Grid Search (FLAT dataset recommended)
    methods_sequence_3 = [
         ('nested_grid_search', {
             'param_grid': param_grid_search,
             'outer_cv': 3, # Outer folds for evaluation
             'inner_cv': 2, # Inner folds for tuning
             'method': 'grid',
             'scoring': 'accuracy',
             'save_results': True
         })
    ]

    # Example 4: Simple CV Evaluation (FLAT dataset required)
    methods_sequence_4 = [
         ('cv_model_evaluation', {
             'cv': 5,
             'save_results': True
         })
    ]

    # Example 5: Load a model and evaluate
    # First, run Example 1 to generate a model file (e.g., results/.../cnn/cnn_epoch...pt)
    # Then run this:
    # model_to_load = "path/to/your/saved/model.pt" # Replace with actual path
    # methods_sequence_5 = [
    #     ('load_model', {'model_path': model_to_load}),
    #     ('single_eval', {'save_results': True})
    # ]


    # --- Choose Sequence and Execute ---
    chosen_sequence = methods_sequence_2 # Select the sequence to run
    # chosen_sequence = methods_sequence_4 # Example: Run CV evaluation if dataset is FLAT

    logger.debug(f"Chosen sequence: {chosen_sequence}")


    # --- Create and Run Executor ---
    try:
        executor = PipelineExecutor(
            dataset_path=dataset_path,
            model_type=model_type,
            results_dir=results_base_dir,
            methods=chosen_sequence,
            # Pipeline default parameters (can be overridden in method sequences)
            img_size=(64, 64), # Smaller size for faster demo
            batch_size=16, # Smaller batch size for potentially lower memory usage
            max_epochs=10,
            patience=5,
            lr=0.001
        )

        final_results = executor.run()

        # Print final results summary
        logger.info("--- Final Execution Results ---")
        for method_id, result_data in final_results.items():
            if isinstance(result_data, dict) and 'error' in result_data:
                 logger.error(f"Method {method_id}: FAILED - {result_data['error']}")
            else:
                 # Print some key metric or info
                 acc = result_data.get('accuracy', result_data.get('mean_test_accuracy', np.nan))
                 best_score = result_data.get('best_score', np.nan) # For search
                 logger.info(f"Method {method_id}: Completed. "
                             f"(Accuracy/MeanAccuracy: {acc:.4f}, BestScore: {best_score:.4f})")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Pipeline initialization or execution failed: {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected errors
         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)


# END OF FILE code_v5_revised.py