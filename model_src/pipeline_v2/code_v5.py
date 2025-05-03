import json
import re
# import matplotlib.pyplot as plt # Matplotlib potentially used for plotting results later, keep import
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Callable, Any, Type

import numpy as np
import pandas as pd
import skorch
import torch
import torch.nn as nn
# from sklearn.preprocessing import label_binarize # Not directly needed, handled by metric functions
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, train_test_split
    # cross_val_score, cross_val_predict # Not used, cross_validate is preferred
)
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint, Callback  # Import base Callback
from skorch.dataset import Dataset as SkorchDataset  # To distinguish from torch.utils.data.Dataset
from skorch.helper import SliceDataset  # Keep SliceDataset import
# import torch.nn.functional as F # F not used directly, can be removed if desired
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, datasets, models

# from torchvision.transforms import v2 # v2 not explicitly used, stick to v1 for now

# --- Global Configurations ---
# TODO: check if reproducibility is guaranteed with this setup (it doesn't seem to be)
# Note: Full reproducibility with CUDA is tricky. deterministic=True helps but isn't always sufficient.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
# Commenting these out might sometimes resolve unrelated CUDA errors, but reduces reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0 # Set to 0 for stability, especially on Windows
# NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 4 # Heuristic for num_workers

import logging
import emoji
import sys
from pathlib import Path
from typing import Optional, Union


# --- ANSI Color Codes ---
class LogColors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

# --- Configure Logging ---
class EnhancedFormatter(logging.Formatter):
    """
    Custom formatter to include timestamp first, emojis, colors (optional),
    level names, and source location (funcName:lineno).
    """

    # Use %(funcName)s instead of %(name)s
    # Adjust padding for funcName (e.g., -25s) and lineno (e.g., -4d)
    level_formats = {
        logging.DEBUG:   f"%(asctime)s | {LogColors.CYAN}%(levelname)-8s{LogColors.RESET} | {emoji.emojize(':magnifying_glass_tilted_left:')} | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.INFO:    f"%(asctime)s | {LogColors.GREEN}%(levelname)-8s{LogColors.RESET} | {emoji.emojize(':information:')} | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.WARNING: f"%(asctime)s | {LogColors.YELLOW}%(levelname)-8s{LogColors.RESET} | {emoji.emojize(':warning:')} | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.ERROR:   f"%(asctime)s | {LogColors.RED}%(levelname)-8s{LogColors.RESET} | {emoji.emojize(':red_exclamation_mark:')} | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.CRITICAL:f"%(asctime)s | {LogColors.BOLD}{LogColors.RED}%(levelname)-8s{LogColors.RESET} | {emoji.emojize(':skull:')} | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
    }
    level_formats_no_color = {
        logging.DEBUG:    "%(asctime)s | %(levelname)-8s | 🐛 | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.INFO:     "%(asctime)s | %(levelname)-8s | ℹ️ | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.WARNING:  "%(asctime)s | %(levelname)-8s | ⚠️ | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.ERROR:    "%(asctime)s | %(levelname)-8s | ❌ | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
        logging.CRITICAL: "%(asctime)s | %(levelname)-8s | 💥 | [%(funcName)-25s:%(lineno)-4d] | %(message)s",
    }
    default_format = f"%(asctime)s | %(levelname)-8s | ? | [%(funcName)-25s:%(lineno)-4d] | %(message)s"

    date_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, use_colors=True):
        super().__init__(fmt="%(levelname)s: %(message)s", datefmt=self.date_format)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        formats = self.level_formats if self.use_colors else self.level_formats_no_color
        log_fmt = formats.get(record.levelno, self.default_format)

        # Add funcName if missing (it usually isn't, but just in case)
        if not hasattr(record, 'funcName'): record.funcName = '?'
        if not hasattr(record, 'lineno'): record.lineno = 0

        formatter = logging.Formatter(log_fmt, datefmt=self.date_format)
        return formatter.format(record)


def get_log_header(use_colors: bool = True) -> str:
    """Generates the log header string."""
    # --- MODIFY HEADER SPACING ---
    # Adjust width for Location column to match format string
    # [%(funcName)-25s:%(lineno)-4d] -> 1 + 25 + 1 + 4 + 1 = 32
    location_width = 32
    header_title = f"{'Timestamp':<21} | {'Level':<8} | {'Emoji':<2} | {'Location':<{location_width}} | {'Message'}"
    separator = f"{'-'*21}-+-{'-'*8}-+-{'-'*2}-+-{'-'*location_width}-+-{'-'*50}"
    # --- END MODIFICATION ---
    if use_colors:
        separator = f"{LogColors.DIM}{separator}{LogColors.RESET}"
    return f"{header_title}\n{separator}"


def write_log_header_if_needed(log_path: Path):
    """Writes a header to the log file only if it's new or empty."""
    # ... (Function body remains the same) ...
    try:
        is_new_or_empty = not log_path.is_file() or log_path.stat().st_size == 0
        if is_new_or_empty:
            header = get_log_header(use_colors=False)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(header + "\n")
            return True
    except Exception as e:
        print(f"Error writing log header to {log_path}: {e}", file=sys.stderr)
    return False


def setup_logger(name: str, log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO, use_colors: bool = True) -> logging.Logger:
    """
    Sets up a logger with enhanced formatting for console and optional file output.
    """
    # ... (Function body remains the same) ...
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    console_formatter = EnhancedFormatter(use_colors=use_colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_log_header_if_needed(log_path)
        try:
            file_formatter = EnhancedFormatter(use_colors=False)
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create file handler for {log_path}: {e}")

    return logger

# --- Example of how to create the logger instance AND PRINT CONSOLE HEADER ---
# (Keep this part outside the functions, where you originally had it)
script_dir = Path(__file__).parent
log_dir = script_dir / 'logs'
log_dir.mkdir(exist_ok=True)
# Create the main logger for your application
logger_name = 'ImgClassPipe' # Or use __name__
log_file_path = log_dir / 'classification.log'
logger = setup_logger(logger_name, log_file_path, level=logging.DEBUG, use_colors=True)
#
# # --- Print Header to Console ---
print(get_log_header(use_colors=True)) # Print header directly to console
logger.info(f"Logger '{logger_name}' initialized. Log file: {log_file_path}")
# # --- End Console Header Print ---


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
        if hasattr(dataset, 'loader') and hasattr(dataset, 'samples'):
             self.loader = dataset.loader
             self.samples = dataset.samples # List of (filepath, class_index) tuples
        # Handle Subset case for FIXED structure where dataset is Subset
        elif isinstance(dataset, Subset) and hasattr(dataset.dataset, 'loader') and hasattr(dataset.dataset, 'samples'):
             logger.debug("TransformedSubset base dataset is a Subset, accessing underlying dataset attributes.")
             self.loader = dataset.dataset.loader
             self.samples = dataset.dataset.samples
             # Adjust indices to be relative to the original dataset if needed?
             # No, Subset __getitem__ handles original indices, so self.indices are correct.
        else:
            # Attempt generic access, might need adjustment for other dataset types
            logger.warning(f"TransformedSubset base dataset is type {type(dataset)}, not ImageFolder or Subset of ImageFolder. "
                           f"Assuming direct __getitem__ returns (path, target) or similar.")
            # Fallback: Try accessing items directly if loader/samples fails
            self.loader = getattr(dataset, 'loader', lambda x: x) # Default loader if not found
            self.samples = getattr(dataset, 'samples', None) # Default samples if not found
            if self.samples is None:
                 logger.warning("Base dataset for TransformedSubset lacks 'samples'. __getitem__ will try direct access.")


    def __getitem__(self, idx: int) -> Tuple[Any, int]: # Return Any for sample before loading
        """
        Retrieves the item at the given index within this subset, applying the specific transform.

        Args:
            idx (int): The index within the subset (not the original dataset).

        Returns:
            Tuple[torch.Tensor, int]: The transformed image tensor and its label.

        Raises:
            IndexError: If the index is out of bounds for the subset.
        """
        if not 0 <= idx < len(self.indices):
             raise IndexError(f"Index {idx} out of bounds for TransformedSubset of length {len(self.indices)}")

        original_idx = self.indices[idx]

        # Use self.samples if available (ImageFolder case)
        if self.samples is not None:
            try:
                path, target = self.samples[original_idx]
                sample = self.loader(path)
            except Exception as e:
                logger.error(f"Error loading sample via self.samples at original index {original_idx} (subset index {idx}): {e}", exc_info=True)
                raise e
        else:
             # Fallback: try direct access on the base dataset (might work for Subset)
             try:
                 # This might return (transformed_image, target) if base dataset has transform
                 sample, target = self.dataset[original_idx]
                 # If sample is already a tensor, fine. If not (e.g. PIL image), need loader?
                 # This path is less reliable if the base dataset isn't ImageFolder.
                 if not isinstance(sample, torch.Tensor):
                      # Attempt to use loader if sample isn't a tensor (e.g., PIL Image)
                      try:
                           sample = self.loader(sample) # This might fail if sample is not a path
                      except Exception as load_err:
                           logger.warning(f"Failed to apply loader to sample of type {type(sample)}: {load_err}. Using sample as is.")

             except Exception as e:
                  logger.error(f"Error loading sample via direct dataset access at original index {original_idx} (subset index {idx}): {e}", exc_info=True)
                  raise e

        # Apply the specific transform of this subset
        if self.transform:
             try:
                 sample = self.transform(sample)
             except Exception as transform_err:
                 logger.error(f"Error applying transform in TransformedSubset for item {idx} (original {original_idx}): {transform_err}", exc_info=True)
                 # What to do here? Raise, return None, return untransformed? Re-raise.
                 raise transform_err


        # Ensure target is an int
        if not isinstance(target, int):
            try:
                target = int(target)
            except Exception:
                 logger.error(f"Could not convert target {target} (type {type(target)}) to int for item {idx}.")
                 # Assign a default or raise? Assign -1 for now.
                 target = -1


        return sample, target

    def __len__(self) -> int:
        """
        Returns the number of samples in this subset.

        Returns:
            int: The size of the subset.
        """
        return len(self.indices)

    # Removed @property targets - rely on _get_targets_from_dataset helper instead
    # as accessing self.dataset.targets directly fails for TransformedSubset of Subset.


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
        class_to_idx (Dict[str, int]): Mapping from class name to index.
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
            ValueError: If val_split_ratio is not between 0 and 1 or dataset issues occur.
        """
        self.root_path = Path(root_path).resolve() # Use resolved path
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
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.num_classes: int = 0

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

        # Get class information from the loaded datasets
        self._extract_class_info()
        if not self.classes:
             raise ValueError(f"Could not determine classes for dataset at {self.root_path}")
        self.num_classes = len(self.classes)
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.classes)}")

        # Assign the correct training dataset based on augmentation flag
        self.train_dataset = self.train_dataset_aug if self.data_augmentation else self.train_dataset_raw
        if self.train_dataset is None:
             raise RuntimeError("Failed to assign a valid training dataset.")

    def _detect_structure(self) -> DatasetStructure:
        """Detects the dataset structure (FLAT or FIXED) based on subdirectories."""
        try:
             root_subdirs = [d.name for d in self.root_path.iterdir() if d.is_dir()]
        except FileNotFoundError:
             logger.error(f"Cannot list directory content, path not found: {self.root_path}")
             raise
        except Exception as e:
             logger.error(f"Error listing directory {self.root_path}: {e}")
             raise

        if 'train' in root_subdirs and 'test' in root_subdirs:
            train_path = self.root_path / 'train'
            test_path = self.root_path / 'test'
            if not train_path.is_dir() or not test_path.is_dir():
                 logger.warning("'train' or 'test' are not directories. Assuming FLAT.")
                 return DatasetStructure.FLAT

            try:
                train_class_dirs = {d.name for d in train_path.iterdir() if d.is_dir()}
                test_class_dirs = {d.name for d in test_path.iterdir() if d.is_dir()}
            except Exception as e:
                 logger.warning(f"Error listing train/test subdirs: {e}. Assuming FLAT.")
                 return DatasetStructure.FLAT

            if train_class_dirs and train_class_dirs == test_class_dirs:
                logger.debug("Found 'train' and 'test' directories with matching class subdirectories. Assuming FIXED structure.")
                return DatasetStructure.FIXED
            elif not train_class_dirs or not test_class_dirs:
                 logger.warning("Found 'train' and 'test' directories, but one or both are empty or contain no class subdirectories. Assuming FLAT.")
                 return DatasetStructure.FLAT
            else:
                 logger.warning("Found 'train' and 'test' directories, but class subdirectories don't match. Assuming FLAT structure.")
                 return DatasetStructure.FLAT

        logger.debug("Did not find standard 'train'/'test' structure. Assuming FLAT structure.")
        return DatasetStructure.FLAT

    def _setup_train_transform(self) -> transforms.Compose:
        """Sets up data augmentation transforms for the training data."""
        # Simple augmentation for stability
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
        try:
            # Load once with eval transform, apply train transform later via wrapper
            full_dataset_obj = datasets.ImageFolder(str(self.root_path), transform=self.eval_transform)
            if len(full_dataset_obj) == 0: raise ValueError("Loaded dataset is empty.")
            self.full_dataset = full_dataset_obj # Store reference
        except Exception as e:
            logger.error(f"Error loading ImageFolder for FLAT dataset: {e}", exc_info=True)
            raise

        logger.info(f"Full dataset loaded: {len(full_dataset_obj)} samples.")
        targets = np.array(full_dataset_obj.targets)
        indices = np.arange(len(targets))

        # Stratified split into train+val / test
        test_split_ratio = 0.20
        if len(indices) < 2 or test_split_ratio <= 0 or test_split_ratio >= 1:
             raise ValueError("Cannot perform train/test split with current settings/data size.")
        try:
            train_val_indices, test_indices = train_test_split(
                indices, test_size=test_split_ratio, stratify=targets, random_state=RANDOM_SEED)
        except ValueError as e:
            logger.warning(f"Stratified train/test split failed ({e}). Using non-stratified split.")
            train_val_indices, test_indices = train_test_split(
                indices, test_size=test_split_ratio, random_state=RANDOM_SEED)
        logger.debug(f"Initial split: {len(train_val_indices)} train+val, {len(test_indices)} test indices.")

        # Stratified split of train+val into actual train / validation
        if self.val_split_ratio > 0 and len(train_val_indices) >= 2:
            train_val_targets = targets[train_val_indices]
            # Adjust validation size calculation relative to the train_val set
            val_size_actual = self.val_split_ratio # This interpretation might need adjustment based on sklearn version
            try:
                 train_indices, val_indices = train_test_split(
                     train_val_indices, test_size=val_size_actual, stratify=train_val_targets, random_state=RANDOM_SEED)
            except ValueError as e:
                 logger.warning(f"Stratified train/val split failed ({e}). Using non-stratified split.")
                 train_indices, val_indices = train_test_split(
                     train_val_indices, test_size=val_size_actual, random_state=RANDOM_SEED)
            logger.debug(f"Train/Val split: {len(train_indices)} train, {len(val_indices)} validation indices.")
        else:
            logger.info("Validation split ratio is 0 or train+val set too small. No validation set created from split.")
            train_indices = train_val_indices
            val_indices = np.array([], dtype=indices.dtype) # Ensure val_indices is an empty array

        # Create Subsets (which inherit the transform from full_dataset_obj)
        self.test_dataset = Subset(full_dataset_obj, test_indices)
        self.val_dataset = Subset(full_dataset_obj, val_indices) if val_indices.size > 0 else None
        self.train_dataset_raw = Subset(full_dataset_obj, train_indices)

        # Create TransformedSubset for training data augmentation
        self.train_dataset_aug = TransformedSubset(
            self.train_dataset_raw, # Base on the raw train Subset
            np.arange(len(train_indices)), # Indices are 0..N-1 within the raw subset
            transform=self.train_transform
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

        try:
            train_val_dataset_obj = datasets.ImageFolder(str(train_path), transform=self.eval_transform)
            if len(train_val_dataset_obj) == 0: raise ValueError("Loaded train dataset is empty.")
            self.test_dataset = datasets.ImageFolder(str(test_path), transform=self.eval_transform)
            if len(self.test_dataset) == 0: logger.warning("Loaded test dataset is empty.")
        except Exception as e:
            logger.error(f"Error loading ImageFolder for FIXED dataset: {e}", exc_info=True)
            raise

        logger.info(f"Train+Validation dataset loaded: {len(train_val_dataset_obj)} samples.")
        logger.info(f"Test dataset loaded: {len(self.test_dataset)} samples.")

        targets = np.array(train_val_dataset_obj.targets)
        indices = np.arange(len(targets))

        # Stratified split of train_val into actual train / validation
        if self.val_split_ratio > 0 and len(indices) >= 2:
            try:
                 train_indices, val_indices = train_test_split(
                     indices, test_size=self.val_split_ratio, stratify=targets, random_state=RANDOM_SEED)
            except ValueError as e:
                 logger.warning(f"Stratified train/val split failed ({e}). Using non-stratified split.")
                 train_indices, val_indices = train_test_split(
                     indices, test_size=self.val_split_ratio, random_state=RANDOM_SEED)
            logger.debug(f"Train/Val split: {len(train_indices)} train, {len(val_indices)} validation indices.")
        else:
            logger.info("Validation split ratio is 0 or train set too small. No validation set created from split.")
            train_indices = indices
            val_indices = np.array([], dtype=indices.dtype)

        # Create Subsets
        self.val_dataset = Subset(train_val_dataset_obj, val_indices) if val_indices.size > 0 else None
        self.train_dataset_raw = Subset(train_val_dataset_obj, train_indices)

        # Create TransformedSubset for training data augmentation
        self.train_dataset_aug = TransformedSubset(
            self.train_dataset_raw, # Base on the raw train Subset
            np.arange(len(train_indices)), # Indices are 0..N-1 within the raw subset
            transform=self.train_transform
        )

        val_count = len(self.val_dataset) if self.val_dataset else 0
        logger.info(f"FIXED Dataset loaded: {len(self.train_dataset_raw)} train, "
                    f"{val_count} validation, "
                    f"{len(self.test_dataset)} test samples.")

    def _extract_class_info(self) -> None:
        """Extracts class names and mapping from the loaded dataset object."""
        dataset_to_inspect = None
        if self.structure == DatasetStructure.FLAT and self.full_dataset:
            dataset_to_inspect = self.full_dataset
        elif self.structure == DatasetStructure.FIXED:
             # Prioritize the train dataset object for class info in FIXED structure
             if self.train_dataset_raw and isinstance(self.train_dataset_raw.dataset, datasets.ImageFolder):
                 dataset_to_inspect = self.train_dataset_raw.dataset
             elif self.test_dataset and isinstance(self.test_dataset, datasets.ImageFolder):
                 logger.warning("Extracting class info from FIXED test dataset.")
                 dataset_to_inspect = self.test_dataset

        if isinstance(dataset_to_inspect, datasets.ImageFolder):
            self.classes = sorted(dataset_to_inspect.classes)
            self.class_to_idx = dataset_to_inspect.class_to_idx
        else:
             logger.warning("Could not find ImageFolder attribute for class info, trying directory scan.")
             source_path = self.root_path / ('train' if self.structure == DatasetStructure.FIXED else '')
             try:
                 found_classes = sorted([d.name for d in source_path.iterdir() if d.is_dir() and not d.name.lower() in ['train', 'test']])
                 if found_classes:
                      self.classes = found_classes
                      self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
                 else:
                      logger.error(f"Directory scan failed to find class folders in {source_path}")
             except Exception as e:
                 logger.error(f"Directory scan failed: {e}")


    def _get_classes(self) -> List[str]:
        """DEPRECATED: Use _extract_class_info instead."""
        # This method is kept for potential backward compatibility but should not be used.
        logger.warning("_get_classes is deprecated. Class info extracted in __init__.")
        return self.classes

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
            num_workers=NUM_WORKERS, # Use global setting
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
            # logger.debug("No validation dataset available, returning None for DataLoader.") # Can be noisy
            return None
        logger.debug(f"Creating validation DataLoader with batch_size={batch_size}")
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
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
            shuffle=False,
            num_workers=NUM_WORKERS,
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
        # Output: 128 * 7 * 7 = 6272 features

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
    Relies on default skorch implementations for fit, step, and data handling,
    but overrides get_dataset to handle data format issues from sklearn CV slicing.
    Uses internal ValidSplit for validation by default.
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
        callbacks: Optional[Union[List[Tuple[str, Callback]], str]] = 'default', # Allow None, 'default', list
        patience: int = 10,
        monitor: str = 'valid_loss',
        lr_scheduler_policy: str = 'ReduceLROnPlateau',
        lr_scheduler_patience: int = 5,
        # --- Use ValidSplit by default ---
        train_split: Optional[Callable] = skorch.dataset.ValidSplit(cv=0.2, stratified=True, random_state=RANDOM_SEED),
        classes = None,
        verbose: int = 1, # Default to 1 for epoch logs
        **kwargs
    ):
        """
        Initializes the SkorchModelAdapter.

        Args:
            module: The PyTorch nn.Module class to wrap.
            module__num_classes: Number of classes, passed to the module's constructor.
            criterion: The loss function class.
            optimizer: The optimizer class.
            lr: Learning rate.
            optimizer__weight_decay: Weight decay.
            max_epochs: Max epochs.
            batch_size: Batch size.
            device: Device ('cuda', 'cpu').
            callbacks: List of skorch callbacks, 'default', or 'disable'.
            patience: Patience for EarlyStopping.
            monitor: Metric to monitor for EarlyStopping/Checkpoint.
            lr_scheduler_policy: Policy for LRScheduler.
            lr_scheduler_patience: Patience for LRScheduler.
            train_split: Skorch data splitting strategy. Defaults to ValidSplit(0.2).
                         Should be set to None only temporarily (e.g., within cross_validate).
            classes: List or array of classes. Required if y is not passed to fit.
            verbose: Skorch verbosity level.
            **kwargs: Additional arguments for NeuralNetClassifier.
        """
        self.module_class = module
        self.module_init_kwargs = {k.split('__', 1)[1]: v for k, v in kwargs.items() if k.startswith('module__')}
        if module__num_classes is not None:
            self.module_init_kwargs['num_classes'] = module__num_classes

        # Handle default/disable callbacks
        final_callbacks_arg = None  # Default to None if 'disable' or invalid
        if callbacks == 'default':
            final_callbacks_arg = [
                ('early_stopping', EarlyStopping(monitor=monitor, patience=patience, load_best=True,
                                                 lower_is_better=monitor.endswith('_loss'))),
                ('checkpoint', Checkpoint(monitor=f'{monitor}_best', f_params='best_model.pt',
                                          dirname=f"skorch_cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                          load_best=False)),
                ('lr_scheduler', LRScheduler(policy=lr_scheduler_policy, monitor=monitor,
                                             mode='min' if monitor.endswith('_loss') else 'max',
                                             patience=lr_scheduler_patience, factor=0.1))
            ]
        elif isinstance(callbacks, list):
            final_callbacks_arg = callbacks  # Pass the list directly
        # If callbacks is None or any other value, final_callbacks_arg remains None

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
            callbacks=final_callbacks_arg,
            train_split=train_split,
            classes=classes,
            verbose=verbose,
            **kwargs
        )

    def get_dataset(self, X, y=None):
        """
        Override to handle the specific case where X is a list of
        (feature, target) tuples (from sklearn CV) and ensure targets
        are LongTensors.
        """
        if isinstance(X, list) and X and isinstance(X[0], (tuple, list)) and len(X[0]) == 2:
            logger.debug("[DEBUG] Adapter.get_dataset: X is list of tuples. Manual unpacking/stacking.")
            try:
                X_features = [item[0] for item in X]
                if not X_features:
                    # Determine shape from first item if possible, else a default placeholder
                    # This assumes all tensors in the list have the same shape, which should be true.
                    # We need C, H, W. Get it from the first tensor.
                    if X: # Ensure X is not empty if X_features is
                         placeholder_shape = X[0][0].shape
                         logger.debug(f"Using placeholder shape {placeholder_shape} for empty X_features_stacked.")
                    else:
                         placeholder_shape = (3, self.module_init_kwargs.get('img_height', 64), self.module_init_kwargs.get('img_width', 64)) # Fallback shape
                         logger.warning(f"Using fallback placeholder shape {placeholder_shape} for empty X_features_stacked.")
                    X_features_stacked = torch.empty((0, *placeholder_shape))
                else:
                    X_features_stacked = torch.stack(X_features, dim=0)

                y_targets = [item[1] for item in X]
                y_targets_tensor = torch.tensor(y_targets, dtype=torch.long)
                logger.debug(f"[DEBUG] Adapter.get_dataset: Instantiating SkorchDataset directly with "
                             f"stacked tensor (shape={X_features_stacked.shape}) and "
                             f"target tensor (shape={y_targets_tensor.shape}, dtype={y_targets_tensor.dtype}).")
                # Instantiate SkorchDataset directly, bypassing parent get_dataset
                return SkorchDataset(X_features_stacked, y_targets_tensor)
            except Exception as e:
                 logger.error(f"Error during list unpacking/SkorchDataset instantiation: {e}", exc_info=True)
                 raise RuntimeError("Failed to process list of tuples input during CV split.") from e

        # --- Handle PyTorch Dataset Input ---
        # If X is a Dataset, skorch should handle it. Pass y=None.
        elif isinstance(X, (Dataset, SkorchDataset)):
            if y is not None:
                logger.debug("[DEBUG] Adapter.get_dataset: X is Dataset and y is not None. Ignoring y.")
            # Use default parent implementation, ensuring y=None is passed
            return super().get_dataset(X, y=None)

        # --- Handle NumPy/Tensor Input ---
        # If X is not a Dataset or list-of-tuples, assume it's features (e.g., np.ndarray)
        # Ensure y is a LongTensor if provided.
        else:
            logger.debug(f"[DEBUG] Adapter.get_dataset: X type {type(X)}. Using default super().get_dataset(X, y).")
            if y is not None:
                 if isinstance(y, torch.Tensor):
                     if y.dtype != torch.long:
                         logger.debug(f"[DEBUG] Adapter.get_dataset: Converting y tensor from {y.dtype} to LongTensor.")
                         y = y.to(dtype=torch.long)
                 else: # Assume numpy array or list
                     logger.debug("[DEBUG] Adapter.get_dataset: Converting y to LongTensor.")
                     y = torch.tensor(np.array(y), dtype=torch.long)
            # Call parent with potentially modified y
            return super().get_dataset(X, y)

    def infer(self, x, **fit_params):
        """
        Perform inference. Assumes 'x' is a Tensor.
        Filters fit_params (like X_val, y_val) before passing to module.
        Moves data to device.
        """
        if not isinstance(x, torch.Tensor):
             logger.error(f"[ERROR] Infer received non-Tensor input: {type(x)}. Upstream issue likely.")
             raise TypeError(f"SkorchModelAdapter.infer received input of type {type(x)}, expected torch.Tensor.")

        # Filter out keys that should not go to the model's forward pass
        module_forward_params = {
            k: v for k, v in fit_params.items()
            if k not in ['X_val', 'y_val'] # Add more if needed
        }

        # Move input tensor to the correct device
        x = x.to(self.device)
        # Call the underlying module with the input and filtered params
        return self.module_(x, **module_forward_params)

    # The following overrides ensure target dtype is correct *just before* loss calculation,
    # acting as a safeguard if data loading/splitting somehow changes it.
    def train_step_single(self, batch, **fit_params):
        """Override train_step_single to ensure yi is LongTensor before loss."""
        self.module_.train()
        Xi, yi = batch
        # Ensure target is LongTensor *before* passing to criterion
        yi = yi.to(dtype=torch.long)
        # Default logic from skorch:
        y_pred = self.infer(Xi, **fit_params) # Assuming default infer is okay
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {'loss': loss, 'y_pred': y_pred}

    def validation_step(self, batch, **fit_params):
        """Override validation_step to ensure yi is LongTensor before loss."""
        self.module_.eval()
        Xi, yi = batch
        # Ensure target is LongTensor *before* passing to criterion
        yi = yi.to(dtype=torch.long)
        # Default logic from skorch:
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params) # Assuming default infer is okay
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {'loss': loss, 'y_pred': y_pred}

    # This is needed to correctly handle X_val/y_val when train_split is None
    def get_split_datasets(self, X, y=None, **fit_params):
        """
        Override to handle explicit validation data passed via fit_params
        OR use the instance's train_split attribute otherwise.
        Handles data type issues from sklearn CV slicing.
        """
        dataset_train = self.get_dataset(X, y) # Our override handles X, y types

        dataset_valid = None
        if 'X_val' in fit_params:
            # Scenario 1: Explicit validation data provided
            logger.debug("[DEBUG] Adapter.get_split_datasets: Found X_val in fit_params. Processing it.")
            dataset_valid = self.get_dataset(fit_params['X_val'], fit_params.get('y_val'))
            # --- REMOVE WARNING ---
            # if self.train_split is not None:
            #     logger.warning("get_split_datasets: X_val was provided, but self.train_split is not None. Check configuration.")
            # --- END REMOVAL ---
        elif self.train_split:
            # Scenario 2: No X_val, use the instance's train_split
            logger.debug("[DEBUG] Adapter.get_split_datasets: No X_val provided, using self.train_split.")
            # Pass original y for stratification by ValidSplit
            dataset_train, dataset_valid = self.train_split(dataset_train, y=y)
        else:
            # Scenario 3: No X_val and no train_split configured
             logger.debug("[DEBUG] Adapter.get_split_datasets: No X_val and self.train_split is None. No validation split.")
             # dataset_valid remains None

        return dataset_train, dataset_valid

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
        self.dataset_path = Path(dataset_path).resolve()
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
        base_results_dir = Path(results_dir).resolve() # Resolve results dir path
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
            classes=np.arange(self.dataset_handler.num_classes), # Pass classes for scoring
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=1 # Set default verbosity for epoch logs
        )
        logger.info(f"  Model Adapter: Initialized with {model_class.__name__}")

        # Load pre-trained weights if specified
        if model_load_path:
            self.load_model(model_load_path)

        logger.info(f"Pipeline initialized successfully for {self.model_type} model and "
                    f"{self.dataset_handler.structure.value} dataset structure.")

    @staticmethod
    def _get_model_class(model_type_str: str) -> Type[nn.Module]:
        """Gets the PyTorch model class based on the model type string."""
        model_mapping = {
            'cnn': SimpleCNN,
            'vit': SimpleViT,
            'diffusion': DiffusionClassifier
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
        Handles multi-class cases using OvR strategy for relevant metrics.
        """
        if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
        if y_score is not None and not isinstance(y_score, np.ndarray): y_score = np.array(y_score)


        metrics: Dict[str, Any] = {'accuracy': accuracy_score(y_true, y_pred)}
        # Use the actual class labels present in the true data for iteration
        present_class_labels = np.unique(y_true)
        # But use the handler's full list for consistent reporting structure
        all_class_names = self.dataset_handler.classes
        if not all_class_names:
             logger.warning("Dataset handler classes list is empty. Cannot compute per-class/macro metrics properly.")
             metrics['macro_avg'] = {'precision': np.nan, 'recall': np.nan, 'specificity': np.nan, 'f1': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan}
             return metrics

        num_classes_total = self.dataset_handler.num_classes # Total expected classes

        per_class_metrics = {name: {} for name in all_class_names}
        all_precisions, all_recalls, all_specificities, all_f1s = [], [], [], []
        all_roc_aucs, all_pr_aucs = [], []

        can_compute_auc = y_score is not None and y_score.shape == (len(y_true), num_classes_total)
        if y_score is not None and not can_compute_auc:
             logger.warning(f"y_score shape {y_score.shape} incompatible with y_true len {len(y_true)} and "
                            f"num_classes {num_classes_total}. Cannot compute AUCs.")

        # Iterate through all expected classes for consistent macro average
        for i, class_name in enumerate(all_class_names):
            class_label = self.dataset_handler.class_to_idx.get(class_name, i) # Get index for this class name

            # Check if this class label actually exists in the current data subset
            if class_label not in present_class_labels:
                 logger.debug(f"Class '{class_name}' (label {class_label}) not present in y_true. Skipping metrics calculation for this class.")
                 # Append NaN or 0? Append NaN for averages that should ignore missing classes.
                 all_precisions.append(np.nan)
                 all_recalls.append(np.nan)
                 all_specificities.append(np.nan)
                 all_f1s.append(np.nan)
                 all_roc_aucs.append(np.nan)
                 all_pr_aucs.append(np.nan)
                 continue # Skip to next class

            # Proceed if class is present
            true_is_class = (y_true == class_label).astype(int)
            pred_is_class = (y_pred == class_label).astype(int)

            precision = precision_score(true_is_class, pred_is_class, zero_division=0)
            recall = recall_score(true_is_class, pred_is_class, zero_division=0) # Sensitivity
            f1 = f1_score(true_is_class, pred_is_class, zero_division=0)
            specificity = recall_score(1 - true_is_class, 1 - pred_is_class, zero_division=0) # Recall of negative class

            all_precisions.append(precision)
            all_recalls.append(recall)
            all_specificities.append(specificity)
            all_f1s.append(f1)

            # --- Calculate AUC Metrics (if scores available) ---
            roc_auc, pr_auc = np.nan, np.nan
            if can_compute_auc:
                try:
                    score_for_class = y_score[:, class_label] # Assumes column index matches class label
                except IndexError:
                    logger.warning(f"Cannot get y_score column index {class_label}. Skipping AUC calculation.")
                    score_for_class = None

                if score_for_class is not None and len(np.unique(true_is_class)) > 1: # AUC requires both classes present
                    try: roc_auc = roc_auc_score(true_is_class, score_for_class)
                    except Exception as e: logger.warning(f"ROC AUC Error (Class {class_name}): {e}")
                    try:
                        prec, rec, _ = precision_recall_curve(true_is_class, score_for_class)
                        if len(rec) > 1 and len(prec) > 1:
                            order = np.argsort(rec)
                            pr_auc = auc(rec[order], prec[order])
                    except Exception as e: logger.warning(f"PR AUC Error (Class {class_name}): {e}")
                elif score_for_class is not None:
                     logger.debug(f"Skipping AUC for class {class_name}: only one class present in y_true.")

            all_roc_aucs.append(roc_auc)
            all_pr_aucs.append(pr_auc)

        # --- Calculate Macro Averages ---
        # Use nanmean to ignore NaN from missing classes or calculation errors
        metrics['macro_avg'] = {
            'precision': float(np.nanmean(all_precisions)),
            'recall': float(np.nanmean(all_recalls)),
            'specificity': float(np.nanmean(all_specificities)),
            'f1': float(np.nanmean(all_f1s)),
            'roc_auc': float(np.nanmean(all_roc_aucs)) if can_compute_auc else np.nan,
            'pr_auc': float(np.nanmean(all_pr_aucs)) if can_compute_auc else np.nan
        }

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

        # Sanitize parameter values before creating the filename string
        def sanitize_value(v):
            if isinstance(v, (str, int, float, bool, type(None))):
                # Replace invalid filename characters (e.g., / \ : * ? " < > |) with underscores
                # Also handle the specific 'N/A' case by replacing '/'
                s_val = str(v).replace('/', '_')  # Replace slashes first
                s_val = re.sub(r'[\\:*?"<>|]', '_', s_val)  # Replace other invalid chars
                return s_val
            return 'complex_param'  # Placeholder for non-simple types

        simple_params = {k: sanitize_value(v) for k, v in params.items() if k != 'self'}
        params_str = '_'.join([f"{k}={v}" for k, v in sorted(simple_params.items())])

        filename_base = f"{method_name}_{params_str}_{timestamp}" if params_str else f"{method_name}_{timestamp}"
        json_filepath = self.results_dir / f"{filename_base}.json"
        csv_filepath = self.results_dir / 'summary_results.csv'

        # --- Save detailed JSON ---
        try:
            # Custom serializer to handle numpy types and NaN
            def json_serializer(obj):
                if isinstance(obj, (np.integer, np.int64)): return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj) if not np.isnan(obj) else None
                elif isinstance(obj, np.ndarray): return obj.tolist()
                elif isinstance(obj, Path): return str(obj)
                elif isinstance(obj, datetime): return obj.isoformat()
                elif isinstance(obj, (slice, type)): return None # Cannot serialize slices or types easily
                try: # Attempt default serialization first
                     return json.JSONEncoder.default(None, obj)
                except TypeError:
                     return str(obj) # Fallback to string representation

            # Clean up non-serializable parts of cv_results if present
            if 'cv_results' in results_data and isinstance(results_data['cv_results'], dict):
                # Example: Remove 'params' key which often contains non-serializable objects
                if 'params' in results_data['cv_results']:
                    del results_data['cv_results']['params']
                # Convert numpy arrays in cv_results
                for key, value in results_data['cv_results'].items():
                     if isinstance(value, np.ndarray):
                          results_data['cv_results'][key] = value.tolist()

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
                'accuracy': results_data.get('accuracy', results_data.get('mean_test_accuracy', np.nan)),
                **{f"macro_{k}": v for k, v in results_data.get('macro_avg', {}).items()},
                **simple_params # Add simple params to summary
            }
            for key, value in summary.items():
                if isinstance(value, float) and np.isnan(value): summary[key] = None

            df_summary = pd.DataFrame([summary])

            if csv_filepath.exists():
                df_summary.to_csv(csv_filepath, mode='a', header=False, index=False, encoding='utf-8')
            else:
                df_summary.to_csv(csv_filepath, mode='w', header=True, index=False, encoding='utf-8')
            logger.info(f"Summary results updated in: {csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to save summary results to CSV {csv_filepath}: {e}", exc_info=True)

    def _get_targets_from_dataset(self, dataset: Dataset) -> np.ndarray:
        """Extracts target labels from various PyTorch Dataset types."""
        # Check for skorch SliceDataset wrapper first
        if isinstance(dataset, SliceDataset):
             # If wrapped around another dataset, recurse
             if hasattr(dataset, 'dataset'):
                  logger.debug("Extracting targets from dataset wrapped by SliceDataset.")
                  return self._get_targets_from_dataset(dataset.dataset)
             # If it wraps data directly, try accessing 'y'
             elif hasattr(dataset, 'y') and dataset.y is not None:
                  logger.debug("Extracting targets directly from SliceDataset.y.")
                  return np.array(dataset.y)

        if isinstance(dataset, Subset):
            if hasattr(dataset.dataset, 'targets'):
                 original_targets = np.array(dataset.dataset.targets)
                 return original_targets[dataset.indices]
            else: # Fallback for Subset of dataset without .targets
                 logger.warning("Base dataset of Subset lacks 'targets'. Iterating subset for targets.")
                 return self._get_targets_by_iteration(dataset)
        elif isinstance(dataset, TransformedSubset):
             # Try accessing targets of the *original* dataset via Subset logic
             if isinstance(dataset.dataset, Subset):
                  if hasattr(dataset.dataset.dataset, 'targets'):
                       original_targets = np.array(dataset.dataset.dataset.targets)
                       # Indices in TransformedSubset are relative to the Subset it wraps
                       # Indices in the Subset are relative to the original dataset
                       subset_indices = dataset.dataset.indices
                       transformed_subset_indices_in_original = [subset_indices[i] for i in dataset.indices]
                       return original_targets[transformed_subset_indices_in_original]
             # Fallback if base isn't Subset or doesn't have targets easily
             logger.warning("Could not get targets efficiently from TransformedSubset base. Iterating.")
             return self._get_targets_by_iteration(dataset)
        elif isinstance(dataset, datasets.ImageFolder):
            return np.array(dataset.targets)
        elif isinstance(dataset, SkorchDataset):
             if dataset.y is None: raise ValueError("SkorchDataset has y=None.")
             return np.array(dataset.y)
        elif isinstance(dataset, ConcatDataset):
            all_targets = [self._get_targets_from_dataset(d) for d in dataset.datasets]
            return np.concatenate(all_targets)
        else: # General fallback
             return self._get_targets_by_iteration(dataset)

    def _get_targets_by_iteration(self, dataset: Dataset) -> np.ndarray:
        """Fallback to extract targets by iterating a dataset."""
        logger.warning(f"Using SLOW target extraction by iteration for dataset type {type(dataset)}")
        targets = []
        try:
             loader = DataLoader(dataset, batch_size=self.model_adapter.batch_size * 4, shuffle=False, num_workers=0)
             for _, y_batch in loader:
                 targets.append(y_batch.cpu().numpy())
             if not targets:
                 logger.warning(f"Iteration yielded no targets for dataset {type(dataset)}.")
                 return np.array([], dtype=int) # Return empty array of appropriate type
             return np.concatenate(targets)
        except Exception as e:
             logger.error(f"Failed to extract targets by iteration: {e}", exc_info=True)
             raise TypeError(f"Unsupported dataset type for target extraction: {type(dataset)}")

    def _load_dataset_to_numpy(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Loads features and labels from a PyTorch Dataset into NumPy arrays."""
        logger.info(f"Loading dataset of type {type(dataset)} (length {len(dataset)}) into NumPy arrays...")
        # Use DataLoader for efficient batch loading
        loader = DataLoader(dataset, batch_size=self.model_adapter.batch_size * 2, # Maybe larger batch for loading
                            shuffle=False, num_workers=0) # Use num_workers=0 for stability

        all_features = []
        all_labels = []
        i = 0
        total = len(loader)
        for features, labels in loader:
            i += 1
            logger.debug(f"Loading batch {i}/{total}")
            all_features.append(features.cpu().numpy()) # Move to CPU before numpy
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
        Performs non-nested hyperparameter search (GridSearchCV or RandomizedSearchCV).
        Uses internal validation split defined in SkorchModelAdapter.
        Loads necessary data into NumPy arrays for compatibility with sklearn CV.
        """
        method_lower = method.lower()
        logger.info(f"Performing non-nested '{method_lower}' search with {cv}-fold CV on training data.")
        logger.info(f"Parameter Grid: {param_grid}")
        logger.info(f"Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None: raise ValueError("n_iter required for random search.")
        if method_lower not in ['grid', 'random']: raise ValueError(f"Unsupported search method: {method}.")

        # --- Get Data as NumPy for sklearn CV ---
        train_dataset_torch = self.dataset_handler.get_train_dataset()
        test_dataset_torch = self.dataset_handler.get_test_dataset()
        if train_dataset_torch is None or test_dataset_torch is None:
             raise RuntimeError("Required train or test PyTorch dataset is missing.")

        logger.info("Loading training data into memory for GridSearchCV...")
        try:
            X_np_train, y_np_train = self._load_dataset_to_numpy(train_dataset_torch)
        except Exception as e:
            logger.error(f"Failed to load training data to NumPy: {e}", exc_info=True)
            raise RuntimeError("Could not load training data to NumPy for CV.") from e

        # --- Setup Search ---
        estimator = clone(self.model_adapter) # Clones adapter WITH ValidSplit enabled
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        SearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        search_kwargs = {
            'estimator': estimator,
            'cv': cv_splitter,
            'scoring': scoring,
            'n_jobs': 1,
            'verbose': 3, # Increased verbosity
            'return_train_score': True,
            'refit': True
        }
        if method_lower == 'grid': search_kwargs['param_grid'] = param_grid
        else:
            search_kwargs['param_distributions'] = param_grid
            search_kwargs['n_iter'] = n_iter
            search_kwargs['random_state'] = RANDOM_SEED
        search = SearchClass(**search_kwargs)

        # --- Run Search ---
        logger.info(f"Fitting {SearchClass.__name__} with NumPy arrays...")
        search.fit(X_np_train, y=y_np_train) # Fit on NumPy data
        logger.info(f"Search completed.")

        # --- Collect Results ---
        results = {
            'method': f"non_nested_{method_lower}_search",
            'params': {'cv': cv, 'n_iter': n_iter if method_lower=='random' else 'N/A', 'method': method_lower, 'scoring': scoring},
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
        }

        # --- Evaluate Best Model on Test Set (using NumPy) ---
        logger.info(f"Evaluating best model (params: {search.best_params_}) on the test set...")
        best_estimator = search.best_estimator_

        logger.info("Loading test data into memory for evaluation...")
        try:
            X_np_test, y_true_test = self._load_dataset_to_numpy(test_dataset_torch)
        except Exception as e:
            logger.error(f"Failed to load test data to NumPy: {e}", exc_info=True)
            raise RuntimeError("Could not load test data to NumPy for evaluation.") from e

        y_pred_test = best_estimator.predict(X_np_test)
        try:
            y_score_test = best_estimator.predict_proba(X_np_test)
        except AttributeError: y_score_test = None

        test_metrics = self._compute_metrics(y_true_test, y_pred_test, y_score_test)
        results['test_set_evaluation'] = test_metrics
        logger.info(f"Test Set Evaluation: Accuracy={test_metrics['accuracy']:.4f}, "
                    f"Macro F1={test_metrics['macro_avg']['f1']:.4f}")

        results['accuracy'] = test_metrics.get('accuracy', np.nan)
        results['macro_avg'] = test_metrics.get('macro_avg', {})

        # --- Save Results ---
        if save_results:
            self._save_results(results, f"non_nested_{method_lower}_search", params=results['params'])

        logger.info(f"Non-nested {method_lower} search finished. Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"Best parameters found: {search.best_params_}")

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
        Performs nested cross-validation for unbiased performance estimation.
        Loads data into NumPy arrays for compatibility.
        """
        method_lower = method.lower()
        logger.info(f"Performing nested '{method_lower}' search.")
        logger.info(f"  Outer CV folds: {outer_cv}, Inner CV folds: {inner_cv}")
        logger.info(f"  Parameter Grid: {param_grid}")
        logger.info(f"  Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None: raise ValueError("n_iter required for random search.")
        if method_lower not in ['grid', 'random']: raise ValueError(f"Unsupported search method: {method}.")

        # --- Setup Inner Search Object ---
        base_estimator = clone(self.model_adapter) # Has ValidSplit enabled
        inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_SEED)
        InnerSearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        inner_search_kwargs = {
            'estimator': base_estimator, 'cv': inner_cv_splitter, 'scoring': scoring,
            'n_jobs': 1, 'verbose': 1, 'refit': True
        }
        if method_lower == 'grid': inner_search_kwargs['param_grid'] = param_grid
        else:
            inner_search_kwargs['param_distributions'] = param_grid
            inner_search_kwargs['n_iter'] = n_iter
            inner_search_kwargs['random_state'] = RANDOM_SEED
        inner_search = InnerSearchClass(**inner_search_kwargs)

        # --- Select Data Based on Structure (Load to NumPy) ---
        X_np, y_np = None, None
        X_np_test, y_np_test = None, None # For fixed test case
        run_standard_nested_cv = False

        if self.dataset_handler.structure == DatasetStructure.FLAT:
            logger.info("Loading full dataset into memory for standard nested CV (FLAT structure).")
            try:
                 full_dataset_torch = self.dataset_handler.get_full_dataset()
                 X_np, y_np = self._load_dataset_to_numpy(full_dataset_torch)
                 run_standard_nested_cv = True
            except Exception as e:
                 raise RuntimeError("Could not load full dataset to NumPy for nested CV.") from e

        elif self.dataset_handler.structure == DatasetStructure.FIXED:
            logger.warning("Adapting nested CV for FIXED dataset structure (using NumPy):")
            logger.warning("  Inner search (tuning) runs on 'train+validation' sets.")
            logger.warning("  Final evaluation uses the single best model on the fixed 'test' set.")
            try:
                 train_dataset = self.dataset_handler.get_train_dataset()
                 val_dataset = self.dataset_handler.get_val_dataset()
                 test_dataset = self.dataset_handler.get_test_dataset()
                 if train_dataset is None or test_dataset is None:
                      raise RuntimeError("Missing train or test dataset for FIXED structure.")

                 datasets_to_combine = [train_dataset]
                 if val_dataset: datasets_to_combine.append(val_dataset)
                 combined_dataset = ConcatDataset(datasets_to_combine) if len(datasets_to_combine) > 1 else train_dataset

                 logger.info("Loading combined train+val data into memory for tuning...")
                 X_np, y_np = self._load_dataset_to_numpy(combined_dataset)
                 logger.info("Loading fixed test data into memory for final evaluation...")
                 X_np_test, y_np_test = self._load_dataset_to_numpy(test_dataset)
                 run_standard_nested_cv = False
            except Exception as e:
                 raise RuntimeError("Could not load data to NumPy for FIXED nested CV adaptation.") from e
        else:
             raise RuntimeError(f"Unknown dataset structure: {self.dataset_handler.structure}")

        # --- Execute Nested CV or Adapted FIXED Workflow ---
        results = {
             'method': f"nested_{method_lower}_search",
             'params': {'outer_cv': outer_cv, 'inner_cv': inner_cv, 'n_iter': n_iter if method_lower=='random' else 'N/A', 'method': method_lower, 'scoring': scoring},
        }

        if run_standard_nested_cv:
            outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED)
            logger.info(f"Running standard nested CV using cross_validate with NumPy data...")
            scoring_dict = { # Use standard sklearn scorers with NumPy data
                'accuracy': 'accuracy',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                'f1_macro': 'f1_macro'
            }
            try:
                 cv_results = cross_validate(
                     inner_search, X_np, y_np, cv=outer_cv_splitter, scoring=scoring_dict,
                     return_estimator=True, n_jobs=1, verbose=2 )
                 logger.info("Nested cross-validation finished.")
                 # Process results (same as before)
                 results['outer_cv_scores'] = {k: v.tolist() for k, v in cv_results.items() if k.startswith('test_')}
                 results['mean_test_accuracy'] = float(np.mean(cv_results['test_accuracy']))
                 # ... (add other mean/std metrics) ...
                 results['best_params_per_fold'] = [est.best_params_ for est in cv_results['estimator']]
                 results['accuracy'] = results['mean_test_accuracy']
                 results['macro_avg'] = {'precision': results.get('mean_test_precision_macro', np.nan)}

            except Exception as e:
                 logger.error(f"Standard nested CV failed: {e}", exc_info=True)
                 results['error'] = str(e) # Store error in results
        else: # Adapted workflow for FIXED dataset (using NumPy data)
             logger.info(f"Running adapted nested CV for FIXED structure with NumPy data...")
             try:
                  logger.info(f"  Step 1: Tuning hyperparameters using inner CV on train+validation data...")
                  inner_search.fit(X_np, y_np)
                  logger.info(f"  Hyperparameter tuning finished. Best params: {inner_search.best_params_}")
                  results['best_params'] = inner_search.best_params_
                  results['best_tuning_score'] = inner_search.best_score_
                  results['inner_cv_results'] = inner_search.cv_results_

                  logger.info(f"  Step 2: Evaluating the best model on the fixed test set...")
                  best_estimator = inner_search.best_estimator_
                  y_pred_test = best_estimator.predict(X_np_test)
                  try: y_score_test = best_estimator.predict_proba(X_np_test)
                  except AttributeError: y_score_test = None
                  test_metrics = self._compute_metrics(y_np_test, y_pred_test, y_score_test)
                  results['fixed_test_set_evaluation'] = test_metrics
                  results['accuracy'] = test_metrics.get('accuracy', np.nan)
                  results['macro_avg'] = test_metrics.get('macro_avg', {})
                  logger.info(f"  Fixed Test Set Eval: Accuracy={test_metrics['accuracy']:.4f}")
             except Exception as e:
                  logger.error(f"Adapted nested CV (FIXED) failed: {e}", exc_info=True)
                  results['error'] = str(e)

        # --- Save Results ---
        if save_results:
            self._save_results(results, f"nested_{method_lower}_search", params=results['params'])

        return results


    def cv_model_evaluation(self, cv: int = 5, save_results: bool = True) -> Dict[str, Any]:
        """
        Performs cross-validation with an inner validation split for monitoring
        and early stopping. Loads data into NumPy arrays.

        Args:
            cv (int): Number of outer cross-validation folds.
            save_results (bool): Whether to save the results.

        Returns:
            Dict[str, Any]: Dictionary containing CV results (scores per fold, averages, std dev).
        """
        logger.info(f"Performing {cv}-fold CV with inner validation split.")

        if self.dataset_handler.structure == DatasetStructure.FIXED:
            raise ValueError("This CV method with inner validation is not designed for FIXED datasets.")

        # --- Load Full Data to NumPy ---
        logger.info("Loading full dataset into memory...")
        try:
            full_dataset_torch = self.dataset_handler.get_full_dataset()
            X_np_full, y_np_full = self._load_dataset_to_numpy(full_dataset_torch)
        except Exception as e:
            raise RuntimeError("Could not load full dataset to NumPy for CV.") from e
        logger.info(f"Using full dataset ({len(X_np_full)} samples) for {cv}-fold CV.")

        # --- Setup Outer CV ---
        outer_cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []
        fold_histories = [] # To store history from each fold

        # --- Manual Outer CV Loop ---
        for fold_idx, (outer_train_indices, outer_test_indices) in enumerate(outer_cv_splitter.split(X_np_full, y_np_full)):
            logger.info(f"--- Starting Outer Fold {fold_idx + 1}/{cv} ---")

            # --- Get Outer Fold Data (NumPy) ---
            X_outer_train = X_np_full[outer_train_indices]
            y_outer_train = y_np_full[outer_train_indices]
            X_test        = X_np_full[outer_test_indices]
            y_test        = y_np_full[outer_test_indices]
            logger.debug(f"Outer split: {len(X_outer_train)} train+val samples, {len(X_test)} test samples.")

            # --- Create Inner Train/Validation Split ---
            # Use a fixed validation split size (e.g., 20% of the outer train data)
            inner_val_size = 0.20
            if len(X_outer_train) < 2 : # Need at least 2 samples to split
                 logger.warning(f"Outer fold {fold_idx+1} training set too small ({len(X_outer_train)}) to create inner validation split. Skipping fold.")
                 # Store NaN results for this fold? Or skip? Skipping for now.
                 continue

            try:
                X_inner_train, X_val, y_inner_train, y_val = train_test_split(
                    X_outer_train, y_outer_train,
                    test_size=inner_val_size,
                    stratify=y_outer_train, # Stratify inner split
                    random_state=RANDOM_SEED
                )
                logger.debug(f"Inner split: {len(X_inner_train)} train samples, {len(X_val)} validation samples.")
            except ValueError as e_inner:
                logger.warning(f"Stratified inner split failed ({e_inner}). Using non-stratified split.")
                X_inner_train, X_val, y_inner_train, y_val = train_test_split(
                    X_outer_train, y_outer_train,
                    test_size=inner_val_size,
                    random_state=RANDOM_SEED
                )

            # --- Setup Estimator for this Fold ---
            # Clone the main adapter (which has train_split=None default, callbacks=default)
            estimator_fold = clone(self.model_adapter)
            # No need to set_params here, defaults are correct

            # --- Fit on Inner Train, Validate on Inner Val ---
            logger.info(f"Fitting model for fold {fold_idx + 1}...")
            try:
                # Pass inner train and explicit validation data
                estimator_fold.fit(X_inner_train, y_inner_train, X_val=X_val, y_val=y_val)
                fold_histories.append(estimator_fold.history) # Store history
            except Exception as fit_err:
                 logger.error(f"Fit failed for fold {fold_idx + 1}: {fit_err}", exc_info=True)
                 # Store NaN results or skip fold? Store NaN for now.
                 fold_results.append({'accuracy': np.nan, 'precision_macro': np.nan,
                                      'recall_macro': np.nan, 'f1_macro': np.nan})
                 continue # Skip scoring for this fold

            # --- Evaluate on Outer Test Set ---
            logger.info(f"Evaluating model on outer test set for fold {fold_idx + 1}...")
            try:
                 y_pred_test = estimator_fold.predict(X_test)
                 # Calculate metrics for this fold
                 fold_acc = accuracy_score(y_test, y_pred_test)
                 fold_prec = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
                 fold_rec = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
                 fold_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
                 fold_results.append({'accuracy': fold_acc, 'precision_macro': fold_prec,
                                       'recall_macro': fold_rec, 'f1_macro': fold_f1})
                 logger.info(f"Fold {fold_idx + 1} Test Scores: Acc={fold_acc:.4f}, F1={fold_f1:.4f}")
            except Exception as score_err:
                 logger.error(f"Scoring failed for fold {fold_idx + 1}: {score_err}", exc_info=True)
                 fold_results.append({'accuracy': np.nan, 'precision_macro': np.nan,
                                      'recall_macro': np.nan, 'f1_macro': np.nan})

        # --- Aggregate Results ---
        if not fold_results: # Handle case where all folds failed
             logger.error("CV evaluation failed for all folds.")
             # Return a structure indicating failure
             return {
                 'method': 'cv_model_evaluation_manual',
                 'params': {'cv': cv},
                 'error': 'All folds failed during execution.',
                 'cv_scores': {},
                 'accuracy': np.nan,
                 'macro_avg': {'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan}
             }

        df_results = pd.DataFrame(fold_results)
        results = {
             'method': 'cv_model_evaluation_manual',
             'params': {'cv': cv},
             'cv_scores': df_results.to_dict(orient='list'), # Store scores per metric
             'mean_test_accuracy': float(df_results['accuracy'].mean()),
             'std_test_accuracy': float(df_results['accuracy'].std()),
             'mean_test_precision_macro': float(df_results['precision_macro'].mean()),
             'mean_test_recall_macro': float(df_results['recall_macro'].mean()),
             'mean_test_f1_macro': float(df_results['f1_macro'].mean()),
             # Add fold histories if needed (can be large)
             # 'fold_histories': fold_histories
        }
        results['accuracy'] = results['mean_test_accuracy']
        results['macro_avg'] = {
            'precision': results['mean_test_precision_macro'],
            'recall': results['mean_test_recall_macro'],
            'f1': results['mean_test_f1_macro'],
            'roc_auc': np.nan, 'pr_auc': np.nan # Not calculated here
        }

        if save_results:
            self._save_results(results, "cv_model_evaluation_manual", params={'cv': cv})

        logger.info(f"Manual CV Evaluation Summary (Avg over {len(fold_results)} folds):")
        logger.info(f"  Accuracy: {results['mean_test_accuracy']:.4f} +/- {results['std_test_accuracy']:.4f}")
        logger.info(f"  Macro F1: {results['mean_test_f1_macro']:.4f} +/- {float(df_results['f1_macro'].std()):.4f}")

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
        """
        logger.info("Starting single training run...")

        # --- Get Data ---
        train_dataset = self.dataset_handler.get_train_dataset()
        val_dataset = self.dataset_handler.get_val_dataset()
        if train_dataset is None: raise RuntimeError("Training dataset missing.")

        # --- Combine Datasets for Skorch's Internal Split ---
        datasets_to_fit = [train_dataset]
        if val_dataset:
            datasets_to_fit.append(val_dataset)
            logger.info(f"Combining train ({len(train_dataset)}) and validation ({len(val_dataset)}) sets.")
            combined_dataset = ConcatDataset(datasets_to_fit)
        else:
            logger.info(f"Using only training set ({len(train_dataset)}) for skorch internal split.")
            combined_dataset = train_dataset

        # --- Configure Model Adapter ---
        # Clone adapter to avoid modifying the pipeline's main one if single_train is intermediate
        adapter_for_train = clone(self.model_adapter)
        # Ensure it uses internal validation split (should be default)
        adapter_for_train.set_params(train_split=skorch.dataset.ValidSplit(cv=0.2, stratified=True, random_state=RANDOM_SEED)) # Explicitly set here too

        params_to_set = {}
        if max_epochs is not None: params_to_set['max_epochs'] = max_epochs
        if lr is not None: params_to_set['lr'] = lr
        if batch_size is not None: params_to_set['batch_size'] = batch_size
        if early_stopping_patience is not None:
             if 'early_stopping' in dict(adapter_for_train.callbacks):
                  params_to_set['callbacks__early_stopping__patience'] = early_stopping_patience
             else: logger.warning("Cannot set patience: 'early_stopping' callback missing.")

        if params_to_set:
            logger.info(f"Overriding adapter parameters for this run: {params_to_set}")
            adapter_for_train.set_params(**params_to_set)

        # --- Train Model ---
        try:
            # Extract targets needed for SkorchDataset creation when split happens
            y_combined = self._get_targets_from_dataset(combined_dataset)
            logger.debug(f"Extracted combined targets (shape={y_combined.shape}) for skorch fit.")
        except Exception as e:
            raise RuntimeError("Could not extract targets for single_train fit") from e

        # Fit using the combined dataset and targets
        adapter_for_train.fit(combined_dataset, y=y_combined) # No fit_params needed

        # --- Collect Results ---
        history = adapter_for_train.history
        best_epoch_info = {}
        valid_loss_key = 'valid_loss'
        # ... (rest of history processing logic from previous version) ...
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
                 logger.error(f"Error processing history: {e}", exc_info=True)
                 validation_was_run = False

        if not validation_was_run:
            # ... (fallback logic using last epoch) ...
            if history:
                 last_epoch_hist = history[-1]
                 last_epoch_num = len(history)
            else: # Should not happen if fit ran, but handle defensively
                 last_epoch_hist = {}
                 last_epoch_num = 0
                 logger.error("Training history is empty after fit completed.")

            best_epoch_info = {
                'best_epoch': last_epoch_num,
                'best_valid_metric_value': np.nan, # Indicate validation metric wasn't used/available
                'valid_metric_name': valid_loss_key, # Still record intended metric
                'train_loss_at_best': float(last_epoch_hist.get('train_loss', np.nan)),
            }
            logger.warning(f"Could not determine best epoch based on validation metric '{valid_loss_key}'. Reporting last epoch stats.")
            if last_epoch_num > 0:
                  logger.info(f"Training finished at Epoch {best_epoch_info['best_epoch']} "
                              f"(Train Loss={best_epoch_info['train_loss_at_best']:.4f})")

        # Prepare results dict
        # ... (get params for results dict) ...
        results = {
            'method': 'single_train',
            'params': { # Store effective parameters used
                'lr': adapter_for_train.lr,
                'max_epochs': adapter_for_train.max_epochs,
                'batch_size': adapter_for_train.batch_size,
                'early_stopping_patience': dict(adapter_for_train.callbacks_).get('early_stopping', MagicMock(patience=None)).patience
            },
            'training_history': history.to_list(), # Convert history to serializable list
            **best_epoch_info # Add best epoch info
        }

        # --- Save Model ---
        if save_model:
            # ... (model saving logic using adapter_for_train.module_.state_dict()) ...
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            val_metric_val = results.get('best_valid_metric_value', np.nan)
            val_metric_name = results.get('valid_metric_name','unknown').replace('_','-')
            val_metric_str = f"val_{val_metric_name}{val_metric_val:.4f}" if not np.isnan(val_metric_val) else "no_val"
            model_filename = f"{self.model_type}_epoch{results.get('best_epoch', 0)}_{val_metric_str}_{timestamp}.pt"
            model_path = self.results_dir / model_filename
            try:
                torch.save(adapter_for_train.module_.state_dict(), model_path)
                logger.info(f"Model state_dict saved to: {model_path}")
                results['saved_model_path'] = str(model_path)
            except Exception as e:
                 logger.error(f"Failed to save model: {e}", exc_info=True)
                 results['saved_model_path'] = None

        # Update the main pipeline adapter with the trained one if desired (or return it)
        self.model_adapter = adapter_for_train
        logger.info("Main pipeline model adapter updated with the model from single_train.")

        # Add dummy metrics for saving compatibility
        results['accuracy'] = np.nan
        results['macro_avg'] = {}
        self._save_results(results, "single_train", params=results['params'])

        return results

    def single_eval(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluates the currently loaded/trained model adapter on the test set.
        Loads test data into NumPy array for prediction.
        """
        logger.info("Starting model evaluation on the test set...")

        if not self.model_adapter.initialized_:
             raise RuntimeError("Model adapter not initialized. Train or load first.")

        # --- Get Data as NumPy ---
        test_dataset_torch = self.dataset_handler.get_test_dataset()
        if test_dataset_torch is None:
            raise RuntimeError("Test dataset missing.")

        logger.info("Loading test data into memory for evaluation...")
        try:
            X_np_test, y_true_test = self._load_dataset_to_numpy(test_dataset_torch)
        except Exception as e:
            raise RuntimeError("Could not load test data to NumPy.") from e
        logger.info(f"Evaluating on {len(X_np_test)} test samples.")

        # --- Make Predictions ---
        y_pred_test = self.model_adapter.predict(X_np_test)
        try:
            y_score_test = self.model_adapter.predict_proba(X_np_test)
        except AttributeError:
             y_score_test = None

        # --- Compute Metrics ---
        metrics = self._compute_metrics(y_true_test, y_pred_test, y_score_test)
        results = {'method': 'single_eval', 'params': {}, **metrics}

        # --- Save Results ---
        if save_results: self._save_results(metrics, "single_eval")

        logger.info(f"Evaluation Summary:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        # ... (log other metrics) ...

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
    logger.setLevel(logging.DEBUG) # Set level to DEBUG for detailed logs

    # --- Configuration ---
    # dataset_path = script_dir / "../data/mini-GCD-flat" # Adjust path as needed
    dataset_path = script_dir / "../data/Swimcat-extend" # Adjust path as needed

    if not Path(dataset_path).exists():
         logger.error(f"Dataset path not found: {dataset_path}")
         logger.error("Please create the dataset or modify the 'dataset_path' variable.")
         exit()

    model_type = "cnn"  # Options: 'cnn', 'vit', 'diffusion'

    param_grid_search = {
        'lr': [0.001, 0.0005],
        'optimizer__weight_decay': [0.01, 0.001],
    }

    # --- Define Method Sequence ---
    # Example 1: Single Train and Eval
    methods_sequence_1 = [
        ('single_train', {'max_epochs': 5, 'save_model': True}),
        ('single_eval', {'save_results': True}),
    ]
    # Example 2: Non-Nested Grid Search
    methods_sequence_2 = [
        ('non_nested_grid_search', {
            'param_grid': param_grid_search, 'cv': 3, 'method': 'grid',
            'scoring': 'accuracy', 'save_results': True
        }),
        # Add ('single_eval', {}) here if you want to eval the refit best model
    ]
    # Example 3: Nested Grid Search (Needs FLAT dataset)
    methods_sequence_3 = [
         ('nested_grid_search', {
             'param_grid': param_grid_search, 'outer_cv': 3, 'inner_cv': 2,
             'method': 'grid', 'scoring': 'accuracy', 'save_results': True
         })
    ]
    # Example 4: Simple CV Evaluation (Needs FLAT dataset)
    methods_sequence_4 = [
         ('cv_model_evaluation', {'cv': 5, 'save_results': True})
    ]

    # --- Choose Sequence and Execute ---
    chosen_sequence = methods_sequence_2 # Select the sequence to run

    logger.debug(f"Chosen sequence: {chosen_sequence}")

    # --- Create and Run Executor ---
    try:
        executor = PipelineExecutor(
            dataset_path=dataset_path,
            model_type=model_type,
            results_dir=results_base_dir,
            methods=chosen_sequence,
            # Pipeline default parameters
            img_size=(64, 64), # Smaller size for faster demo
            batch_size=16,
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
                 acc = result_data.get('accuracy', result_data.get('mean_test_accuracy', np.nan))
                 best_score = result_data.get('best_score', np.nan)
                 logger.info(f"Method {method_id}: Completed. "
                             f"(Accuracy/MeanAccuracy: {acc:.4f}, BestScore: {best_score:.4f})")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Pipeline initialization or execution failed: {e}", exc_info=True)
    except Exception as e: # Catch any other unexpected errors
         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)