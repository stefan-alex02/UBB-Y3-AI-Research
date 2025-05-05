# --- START OF FILE code_v7.py ---
import hashlib
import json
import re
# import matplotlib.pyplot as plt # Keep for potential future use
import time
import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Callable, Any, Type, Optional, Union

import numpy as np
import pandas as pd
import skorch
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.preprocessing import label_binarize # Not needed
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, make_scorer,
    roc_curve
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, train_test_split, PredefinedSplit
    # cross_val_score, cross_val_predict # Not used
)
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint, Callback, EpochScoring
from skorch.dataset import Dataset as SkorchDataset, ValidSplit
from skorch.helper import SliceDataset # Keep? Maybe not needed if always using PathImageDataset
from skorch.utils import to_numpy
# import torch.nn.functional as F # Not used
from torch.utils.data import Dataset, DataLoader, Subset # Subset might still be useful conceptually, PathImageDataset is key
from torchvision import transforms, datasets, models
from PIL import Image # Needed for PathImageDataset

import logging
import emoji
import sys
from pathlib import Path
from unittest.mock import MagicMock # For default patience value

# --- Global Configurations ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED) # Ensure random module is also seeded
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
# Commenting these out might sometimes resolve unrelated CUDA errors, but reduces reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0 # Set to 0 for stability with image loading, especially on Windows

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
# In EnhancedFormatter class:

class EnhancedFormatter(logging.Formatter):
    debug_emoji = '🐞'
    info_emoji = 'ℹ️'
    warning_emoji = '⚠️'
    error_emoji = '❌'
    critical_emoji = '💀'
    default_emoji = '?'

    # Define widths for alignment
    filename_padding = 10
    funcname_padding = 18
    lineno_padding = 4
    # Calculate total width for Location column: Brackets + file + : + func + : + line
    total_location_width = 2 + filename_padding + 1 + funcname_padding + 1 + lineno_padding # = 43

    level_to_color = {
        logging.DEBUG: LogColors.CYAN, logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW, logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.BOLD + LogColors.RED,
    }
    level_to_emoji = {
        logging.DEBUG: debug_emoji, logging.INFO: info_emoji,
        logging.WARNING: warning_emoji, logging.ERROR: error_emoji,
        logging.CRITICAL: critical_emoji,
    }

    date_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, use_colors=True):
        # Basic init, we override format() completely
        super().__init__(fmt=None, datefmt=self.date_format)
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # --- Start Manual Formatting ---
        try:
            # 1. Timestamp
            timestamp = self.formatTime(record, self.date_format)

            # 2. Level Name and Color
            level_name = record.levelname
            level_color = self.level_to_color.get(record.levelno, '') if self.use_colors else ''
            reset_color = LogColors.RESET if self.use_colors and level_color else ''
            formatted_level = f"{level_color}{level_name:<8}{reset_color}" # Pad to 8 chars

            # 3. Location String
            filename = getattr(record, 'filename', '?')
            funcname = getattr(record, 'funcName', '?')
            lineno = getattr(record, 'lineno', 0)

            # Truncate filename (without extension)
            if '.' in filename: filename = filename.rsplit('.', 1)[0]
            max_file_len = self.filename_padding
            truncated_filename = (filename[:max_file_len-3] + '...') if len(filename) > max_file_len else filename

            # Truncate funcName
            max_func_len = self.funcname_padding
            truncated_funcname = (funcname[:max_func_len-3] + '...') if len(funcname) > max_func_len else funcname

            # Format location with padding
            location_str = (
                f"[{truncated_filename:<{self.filename_padding}}:"
                f"{truncated_funcname:<{self.funcname_padding}}:"
                f"{lineno:>{self.lineno_padding}}]"
            )
            # Pad the entire location block
            formatted_location = f"{location_str:<{self.total_location_width}}"

            # 4. Emoji Prefix + Message
            emoji_prefix = self.level_to_emoji.get(record.levelno, self.default_emoji)
            message = record.getMessage() # Get the formatted message
            formatted_message = f"{emoji_prefix} {message}" # Add space after emoji

            # 5. Combine parts
            log_entry = f"{timestamp} | {formatted_level} | {formatted_location} | {formatted_message}"
            return log_entry

        except Exception as e:
            # Fallback formatting on any error during manual formatting
            record.msg = f"!!! LOG FORMATTING ERROR: {e}. Original message: {record.getMessage()}"
            # Use a basic formatter as ultimate fallback
            bf = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt=self.date_format)
            return bf.format(record)

# Update get_log_header to match new padding calculation
def get_log_header(use_colors: bool = True) -> str:
    location_width = EnhancedFormatter.total_location_width # Use width from class
    # Remove Emoji column header
    header_title = f"{'Timestamp':<19} | {'Level':<8} | {'Location':<{location_width}} | {'Message'}"
    separator = f"{'-'*19}-+-{'-'*8}-+-{'-'*location_width}-+-{'-'*50}" # Removed emoji separator part
    if use_colors: separator = f"{LogColors.DIM}{separator}{LogColors.RESET}"
    return f"{header_title}\n{separator}"

def write_log_header_if_needed(log_path: Path):
    try:
        is_new_or_empty = not log_path.is_file() or log_path.stat().st_size == 0
        if is_new_or_empty:
            header = get_log_header(use_colors=False)
            with open(log_path, 'a', encoding='utf-8') as f: f.write(header + "\n")
            return True
    except Exception as e:
        print(f"Error writing log header to {log_path}: {e}", file=sys.stderr)
    return False

def setup_logger(name: str,
                 log_dir: Union[str, Path], # <<< Accept directory instead of full path
                 log_filename: str = 'classification.log', # <<< Default filename
                 level: int = logging.INFO,
                 use_colors: bool = True) -> logging.Logger:
    """Sets up the logger to log to console and a file within the specified directory."""
    logger = logging.getLogger(name)
    # Prevent duplicate handlers if logger already exists (e.g., in interactive sessions)
    if logger.hasHandlers():
        logger.handlers.clear()
        # logger.info("Cleared existing logger handlers.") # Optional debug log

    logger.setLevel(level)
    logger.propagate = False # Prevent propagation to root logger

    # Console Handler (always add)
    console_formatter = EnhancedFormatter(use_colors=use_colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File Handler (if directory provided)
    if log_dir:
        log_path = Path(log_dir) / log_filename
        # Ensure directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Check and write header if needed
        is_new_file = write_log_header_if_needed(log_path)
        try:
            file_formatter = EnhancedFormatter(use_colors=False)
            # Use 'a' mode to append to the log file
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            # Log initialization message only if it's a new file or handlers were just added
            if is_new_file or not logger.handlers: # Simple check if handlers were just added
                 logger.info(f"Logger '{name}' initialized. Log file: {log_path}")

        except Exception as e:
             # Use logger AFTER console handler is added
             logger.error(f"Failed to create file handler for {log_path}: {e}")
    else:
        logger.warning("No log directory provided. Logging to console only.")

    return logger

# --- Setup Logger Instance ---
# This logger will be configured by PipelineExecutor
logger: logging.Logger = logging.getLogger('ImgClassPipe_Default')

# --- Dataset Handling ---

class DatasetStructure(Enum):
    FLAT = "flat"
    FIXED = "fixed"

class PathImageDataset(Dataset):
    """
    Custom PyTorch Dataset that loads images from a list of paths.
    Applies transforms during __getitem__. Includes collate_fn for robustness.
    """
    def __init__(self, paths: List[Union[str, Path]], labels: Optional[List[int]] = None, transform: Optional[Callable] = None):
        """
        Args:
            paths (List[Union[str, Path]]): List of image file paths.
            labels (Optional[List[int]]): Corresponding list of integer labels. If None, labels are ignored.
            transform (Optional[Callable]): Transform to apply to the image.
        """
        if labels is not None and len(paths) != len(labels):
            raise ValueError(f"Paths and labels must have the same length, but got {len(paths)} and {len(labels)}")
        self.paths = [Path(p) for p in paths] # Ensure paths are Path objects
        self.labels = labels
        self.transform = transform
        self.image_size = self._get_image_size_from_transform(transform) # Attempt to get size for collate fallback


    @staticmethod
    def _get_image_size_from_transform(transform) -> Tuple[int, int]:
        """Helper to find image size from Resize transform, defaults to (64, 64)."""
        if isinstance(transform, transforms.Compose):
            for t in transform.transforms:
                if isinstance(t, transforms.Resize):
                    size = t.size
                    if isinstance(size, int): return (size, size)
                    if isinstance(size, (list, tuple)) and len(size) == 2: return tuple(size)
        elif isinstance(transform, transforms.Resize): # Handle direct Resize
             size = transform.size
             if isinstance(size, int): return (size, size)
             if isinstance(size, (list, tuple)) and len(size) == 2: return tuple(size)
        # Fallback size
        logger.debug("Could not determine image size from transform, using default (64, 64) for collate fallback.")
        return (64, 64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label_val = self.labels[idx] if self.labels is not None else -1 # Use -1 if no labels

        try:
            # Use context manager for file opening
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except FileNotFoundError:
             logger.error(f"Image file not found: {img_path}. Returning None.")
             return None, None
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Returning None.")
            return None, None # Return None tuple if loading fails

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 logger.warning(f"Error transforming image {img_path}: {e}. Returning None.")
                 return None, None # Return None tuple if transform fails

        label_tensor = torch.tensor(label_val, dtype=torch.long)
        return image, label_tensor

    @staticmethod
    def collate_fn(batch):
        """Filters None items and stacks the rest."""
        original_batch_size = len(batch)
        # Filter out items where __getitem__ returned (None, None)
        batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]

        if not batch:
            logger.warning(f"Collate_fn received empty batch after filtering {original_batch_size} items.")
            # Determine fallback shape (this is tricky without instance access)
            # Relying on a common size or erroring might be better.
            # Let's try returning truly empty tensors. Skorch might handle this.
            # If not, the dataloader might crash, which is informative.
            return torch.empty((0, 3, 64, 64)), torch.empty((0), dtype=torch.long) # Default 64x64

        try:
            images, labels = zip(*batch)
        except ValueError as e:
            logger.error(f"Error during zip in collate_fn: {e}", exc_info=True)
            raise e

        try:
            # Use default_collate logic if available (handles padding etc.)
            # images = torch.utils.data.dataloader.default_collate(images)
            # labels = torch.utils.data.dataloader.default_collate(labels)
            # Stacking is usually sufficient for classification images/labels
            images = torch.stack(images, 0)
            labels = torch.stack(labels, 0)
        except Exception as e:
            logger.error(f"Error during torch.stack in collate_fn: {e}", exc_info=True)
            # Log shapes for debugging
            if images: logger.error(f"Shapes of images in failed stack: {[img.shape for img in images]}")
            raise e

        return images, labels


class ImageDatasetHandler:
    """
    Handles loading image paths and labels from disk, detecting structure,
    and providing access to paths/labels for different splits. Also manages transforms.
    """
    def __init__(self,
                 root_path: Union[str, Path],
                 img_size: Tuple[int, int] = (224, 224),
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2, # Ratio for test split if structure is FLAT
                 data_augmentation: bool = True,
                 force_flat_for_fixed_cv: bool = False): # New flag
        self.root_path = Path(root_path).resolve()
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"Dataset root path not found: {self.root_path}")
        if not 0.0 <= val_split_ratio < 1.0:
            raise ValueError("val_split_ratio must be between 0.0 and 1.0 (exclusive of 1.0)")
        if not 0.0 <= test_split_ratio_if_flat < 1.0:
             raise ValueError("test_split_ratio_if_flat must be between 0.0 and 1.0 (exclusive of 1.0)")

        self.img_size = img_size
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio_if_flat = test_split_ratio_if_flat
        self.data_augmentation = data_augmentation
        self.force_flat_for_fixed_cv = force_flat_for_fixed_cv

        self.structure = self._detect_structure()
        logger.info(f"Detected dataset structure: {self.structure.value}")
        if self.structure == DatasetStructure.FIXED and self.force_flat_for_fixed_cv:
             logger.warning("Dataset is FIXED, but force_flat_for_fixed_cv=True. "
                            "CV methods will treat train+test as a single dataset. "
                            "Results might not reflect standard fixed-test evaluation.")

        # Transforms
        self.train_transform = self._setup_train_transform() if self.data_augmentation else self._setup_eval_transform()
        self.eval_transform = self._setup_eval_transform()

        # Load paths and labels
        self._all_paths: List[Path] = []
        self._all_labels: List[int] = []
        self._train_val_paths: List[Path] = []
        self._train_val_labels: List[int] = []
        self._test_paths: List[Path] = []
        self._test_labels: List[int] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        self._load_paths_and_labels()
        if not self.classes:
             raise ValueError(f"Could not determine classes for dataset at {self.root_path}")
        self.num_classes = len(self.classes)
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.classes)}")

        # Log dataset sizes
        logger.info(f"Dataset sizes: {len(self._train_val_paths)} train+val, {len(self._test_paths)} test. Total: {len(self._all_paths)}")
        if self.structure == DatasetStructure.FIXED and self.force_flat_for_fixed_cv:
            logger.info(f"Total combined size (for forced CV): {len(self._all_paths)}")


    def _detect_structure(self) -> DatasetStructure:
        # (Logic remains the same as code_v6)
        try: root_subdirs = [d.name for d in self.root_path.iterdir() if d.is_dir()]
        except Exception as e: raise RuntimeError(f"Error listing directory {self.root_path}: {e}") from e

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
                return DatasetStructure.FIXED
            else:
                 logger.warning("Found 'train'/'test' dirs, but class subdirs don't match or are empty. Assuming FLAT.")
                 return DatasetStructure.FLAT
        return DatasetStructure.FLAT


    def _setup_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10), # Reduced rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _setup_eval_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _scan_dir_for_paths_labels(target_dir: Path) -> Tuple[List[Path], List[int], List[str], Dict[str, int]]:
        """Scans a directory (like ImageFolder) for image paths and labels."""
        paths = []
        labels = []
        target_dir = Path(target_dir)
        if not target_dir.is_dir(): return [], [], [], {}

        class_names = sorted([d.name for d in target_dir.iterdir() if d.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        for class_name, class_idx in class_to_idx.items():
            class_dir = target_dir / class_name
            for img_path in class_dir.glob('*.*'):
                # Basic check for common image extensions
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    paths.append(img_path)
                    labels.append(class_idx)
        return paths, labels, class_names, class_to_idx

    def _load_paths_and_labels(self) -> None:
        """Loads paths and labels based on detected structure."""
        if self.structure == DatasetStructure.FLAT:
            logger.info(f"Scanning FLAT dataset from {self.root_path}...")
            all_paths, all_labels, classes, class_to_idx = self._scan_dir_for_paths_labels(self.root_path)
            if not all_paths: raise ValueError(f"No images found in FLAT dataset at {self.root_path}")
            self.classes = classes
            self.class_to_idx = class_to_idx
            self._all_paths = all_paths
            self._all_labels = np.array(all_labels) # Use numpy for easier slicing

            # Stratified split into train+val / test
            if self.test_split_ratio_if_flat > 0 and len(self._all_paths) >= 2:
                 indices = np.arange(len(self._all_paths))
                 try:
                     train_val_indices, test_indices = train_test_split(
                         indices, test_size=self.test_split_ratio_if_flat, stratify=self._all_labels, random_state=RANDOM_SEED)
                 except ValueError as e:
                     logger.warning(f"Stratified train/test split failed ({e}). Using non-stratified split.")
                     train_val_indices, test_indices = train_test_split(
                         indices, test_size=self.test_split_ratio_if_flat, random_state=RANDOM_SEED)

                 self._train_val_paths = [self._all_paths[i] for i in train_val_indices]
                 self._train_val_labels = self._all_labels[train_val_indices].tolist() # Convert back to list
                 self._test_paths = [self._all_paths[i] for i in test_indices]
                 self._test_labels = self._all_labels[test_indices].tolist()
                 self._all_labels = self._all_labels.tolist() # Convert back to list for consistency
            else:
                 logger.info("Test split ratio is 0 or dataset too small. Using all data as train+val.")
                 self._train_val_paths = self._all_paths
                 self._train_val_labels = self._all_labels.tolist()
                 self._test_paths = []
                 self._test_labels = []
                 self._all_labels = self._all_labels.tolist()

        else: # FIXED structure
            train_path = self.root_path / 'train'
            test_path = self.root_path / 'test'
            logger.info(f"Scanning FIXED dataset from {train_path} (train) and {test_path} (test)...")

            train_paths, train_labels, train_classes, train_c2i = self._scan_dir_for_paths_labels(train_path)
            test_paths, test_labels, test_classes, test_c2i = self._scan_dir_for_paths_labels(test_path)

            if not train_paths: logger.warning(f"No images found in train directory: {train_path}")
            if not test_paths: logger.warning(f"No images found in test directory: {test_path}")
            if train_classes != test_classes:
                 logger.warning(f"Class mismatch between train ({train_classes}) and test ({test_classes}) dirs. Using train classes.")
                 # Decide how to handle mismatch - using train's classes might be safer
                 test_labels = [test_c2i.get(test_classes[lbl], -1) for lbl in test_labels] # Remap test labels? Risky. Or error out? Error might be better.
                 # Let's assume they should match for now, use train classes.
                 # If test_c2i is different, labels could be wrong.
                 # Simplest: Use train's mapping for consistency.
                 final_labels_test = []
                 test_idx_to_name = {v:k for k,v in test_c2i.items()}
                 for label_idx in test_labels:
                      class_name = test_idx_to_name.get(label_idx)
                      final_idx = train_c2i.get(class_name, -1) # Map to train index
                      if final_idx == -1: logger.warning(f"Class {class_name} from test set not found in train set.")
                      final_labels_test.append(final_idx)
                 test_labels = final_labels_test


            self.classes = train_classes
            self.class_to_idx = train_c2i
            self._train_val_paths = train_paths
            self._train_val_labels = train_labels
            self._test_paths = test_paths
            self._test_labels = test_labels

            # If forcing flat, combine train and test for _all_paths
            if self.force_flat_for_fixed_cv:
                self._all_paths = self._train_val_paths + self._test_paths
                self._all_labels = self._train_val_labels + self._test_labels

    # --- Public Accessors ---
    def get_train_val_paths_labels(self) -> Tuple[List[Path], List[int]]:
        """Returns paths and labels for the training + validation set."""
        return self._train_val_paths, self._train_val_labels

    def get_test_paths_labels(self) -> Tuple[List[Path], List[int]]:
        """Returns paths and labels for the test set."""
        if self.structure == DatasetStructure.FLAT and not self._test_paths:
             logger.warning("Requesting test paths/labels for FLAT structure, but no test split was created.")
        elif self.structure == DatasetStructure.FIXED and not self._test_paths:
             logger.warning("Requesting test paths/labels for FIXED structure, but test dir was empty.")
        return self._test_paths, self._test_labels

    def get_full_paths_labels_for_cv(self) -> Tuple[List[Path], List[int]]:
        """Returns paths/labels for the entire dataset, respecting force_flat_for_fixed_cv."""
        if self.structure == DatasetStructure.FLAT:
            return self._all_paths, self._all_labels
        elif self.structure == DatasetStructure.FIXED:
            if self.force_flat_for_fixed_cv:
                 logger.debug("Providing combined train+test paths/labels for forced flat CV.")
                 return self._all_paths, self._all_labels # Uses combined lists
            else:
                 raise ValueError("Cannot get 'full' dataset for FIXED structure unless force_flat_for_fixed_cv=True.")
        else: # Should not happen
             raise RuntimeError(f"Unknown dataset structure {self.structure}")

    def get_classes(self) -> List[str]:
        return self.classes

    def get_class_to_idx(self) -> Dict[str, int]:
        return self.class_to_idx

    def get_train_transform(self) -> Callable:
        return self.train_transform

    def get_eval_transform(self) -> Callable:
        return self.eval_transform


# --- Model Definitions (Keep SimpleCNN, SimpleViT, DiffusionClassifier as before) ---
# --- SimpleCNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5): # Add dropout rate
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        # Define layers (using example calculation for IMAGE_SIZE=64 from simplified code)
        # Make this calculation dynamic or ensure image size is passed if needed.
        # For AdaptiveAvgPool, the input size doesn't matter as much.
        img_h_after_pools = 64 // 8 # Assuming 3 max pools and input 64
        img_w_after_pools = 64 // 8

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.MaxPool2d(2), # 64->32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.MaxPool2d(2), # 32->16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(128), nn.MaxPool2d(2) # 16->8
        )
        # Use AdaptiveAvgPool2d to handle variable input sizes gracefully
        self.avgpool = nn.AdaptiveAvgPool2d((max(1, img_h_after_pools//2), max(1, img_w_after_pools//2))) # Pool down further to e.g., 4x4
        pooled_size = 128 * max(1, img_h_after_pools//2) * max(1, img_w_after_pools//2) # Calculate pooled feature size

        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, 512), # Input size depends on avgpool output
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), # Use passed dropout rate
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"SimpleCNN initialized with {num_classes} output classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- SimpleViT (Keep as before, uses models.vit_b_16) ---
class SimpleViT(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        logger.debug("Loading pre-trained vit_b_16 model...")
        vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        logger.debug("Pre-trained model loaded.")
        original_hidden_dim = vit_model.heads.head.in_features
        vit_model.heads.head = nn.Linear(original_hidden_dim, self.num_classes)
        logger.debug(f"Replaced ViT head for {num_classes} classes.")

        num_layers_to_unfreeze = 4
        total_params = len(list(vit_model.parameters()))
        unfrozen_count = 0
        for i, param in enumerate(vit_model.parameters()):
            if i < total_params - num_layers_to_unfreeze:
                 param.requires_grad = False
            else:
                 param.requires_grad = True
                 unfrozen_count += 1
        logger.info(f"SimpleViT: Froze layers, unfroze last {unfrozen_count} parameter groups.")
        self.model = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- DiffusionClassifier (Keep as before, uses models.resnet50) ---
class DiffusionClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.4): # Add dropout rate
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        logger.debug("Loading pre-trained ResNet50 backbone...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        logger.debug("Pre-trained ResNet50 loaded.")
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet

        num_layers_to_unfreeze = 5
        total_params = len(list(self.backbone.parameters()))
        unfrozen_count = 0
        for i, param in enumerate(self.backbone.parameters()):
            if i < total_params - num_layers_to_unfreeze: param.requires_grad = False
            else: param.requires_grad = True; unfrozen_count+=1
        logger.info(f"DiffusionClassifier (ResNet50): Froze layers, unfroze last {unfrozen_count} backbone param groups.")

        self.diffusion_head = nn.Sequential(
            nn.Linear(in_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), # Use passed rate
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), # Use passed rate
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"DiffusionClassifier initialized with ResNet50 backbone and custom head for {num_classes} classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.diffusion_head(features)
        return logits


def _get_default_callbacks(patience: int = 10,
                           monitor: str = 'valid_loss',
                           lr_policy: str = 'ReduceLROnPlateau',
                           lr_patience: int = 5) -> List[Tuple[str, Callback]]:
    """Generates the default list of skorch callbacks."""
    is_loss = monitor.endswith('_loss')
    # Use unique names to avoid potential conflicts if user provides callbacks with same default names
    return [
        ('default_early_stopping', EarlyStopping(
            monitor=monitor, patience=patience, load_best=True, lower_is_better=is_loss)),
        ('default_lr_scheduler', LRScheduler(
            policy=lr_policy, monitor=monitor, mode='min' if is_loss else 'max', patience=lr_patience, factor=0.1)),
        ('default_train_acc', EpochScoring(
            'accuracy', lower_is_better=False, on_train=True, name='train_acc'))
    ]

# --- Skorch Model Adapter (Final Version) ---

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
                X_paths = dataset.X;
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

# --- Classification Pipeline ---

class ClassificationPipeline:
    """
    Manages image classification: data loading (paths), model selection,
    training, tuning, evaluation. Uses SkorchModelAdapter with dynamic transforms.
    """

    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: str = 'cnn',
                 model_load_path: Optional[Union[str, Path]] = None,
                 img_size: Tuple[int, int] = (224, 224),
                 results_dir: Union[str, Path] = 'results',
                 save_detailed_results: bool = False, # Flag for detailed results
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 data_augmentation: bool = True,
                 force_flat_for_fixed_cv: bool = False,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 optimizer__weight_decay: float = 0.01,
                 module__dropout_rate: Optional[float] = None
                 ):
        self.dataset_path = Path(dataset_path).resolve()
        self.model_type = model_type.lower()
        self.force_flat_for_fixed_cv = force_flat_for_fixed_cv
        self.save_detailed_results = save_detailed_results
        # Logger is configured externally by the Executor now
        logger.info(f"Initializing Classification Pipeline:")
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Model Type: {self.model_type}")
        logger.info(f"  Force Flat for Fixed CV: {self.force_flat_for_fixed_cv}")
        logger.info(f"  Save Detailed Results: {self.save_detailed_results}")

        self.dataset_handler = ImageDatasetHandler(
            root_path=self.dataset_path, img_size=img_size,
            val_split_ratio=val_split_ratio, test_split_ratio_if_flat=test_split_ratio_if_flat,
            data_augmentation=data_augmentation, force_flat_for_fixed_cv=self.force_flat_for_fixed_cv
        )

        base_results_dir = Path(results_dir).resolve()
        dataset_name = self.dataset_path.name
        timestamp_init = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = base_results_dir / dataset_name / self.model_type / f"{timestamp_init}_seed{RANDOM_SEED}"
        # Experiment dir created by Executor before logger setup
        # logger.info(f"  Base experiment results dir: {self.experiment_dir}") # Logged by executor

        model_class = self._get_model_class(self.model_type)

        intended_callbacks_setting = 'default'
        initial_callbacks_list = None
        if intended_callbacks_setting == 'default':
            initial_callbacks_list = _get_default_callbacks(
                patience=patience, monitor='valid_loss',
                lr_policy='ReduceLROnPlateau', lr_patience=5
            )
        elif isinstance(intended_callbacks_setting, list):
            initial_callbacks_list = intended_callbacks_setting

        module_params = {}
        if module__dropout_rate is not None: module_params['module__dropout_rate'] = module__dropout_rate

        self.model_adapter_config = {
            'module': model_class, 'module__num_classes': self.dataset_handler.num_classes,
            'criterion': nn.CrossEntropyLoss, 'optimizer': torch.optim.AdamW,
            'lr': lr, 'max_epochs': max_epochs, 'batch_size': batch_size, 'device': DEVICE,
            'callbacks': initial_callbacks_list,
            'patience_cfg': patience, 'monitor_cfg': 'valid_loss',
            'lr_policy_cfg': 'ReduceLROnPlateau', 'lr_patience_cfg': 5,
            'train_transform': self.dataset_handler.get_train_transform(),
            'valid_transform': self.dataset_handler.get_eval_transform(),
            'classes': np.arange(self.dataset_handler.num_classes),
            'verbose': 1, # Default verbosity for adapter itself
            'optimizer__weight_decay': optimizer__weight_decay,
            **module_params
        }

        init_config_for_adapter = self.model_adapter_config.copy()
        init_config_for_adapter.pop('patience_cfg', None); init_config_for_adapter.pop('monitor_cfg', None)
        init_config_for_adapter.pop('lr_policy_cfg', None); init_config_for_adapter.pop('lr_patience_cfg', None)

        # Important: Pass the base config to the adapter, not the processed one
        # Skorch handles parameter setting via set_params
        self.model_adapter = SkorchModelAdapter(**init_config_for_adapter)
        logger.info(f"  Model Adapter: Initialized with {model_class.__name__}")

        if model_load_path:
            self.load_model(model_load_path)
        # logger.info(f"Pipeline initialized successfully.") # Logged by executor

    @staticmethod
    def _get_model_class(model_type_str: str) -> Type[nn.Module]:
        model_mapping = {'cnn': SimpleCNN, 'vit': SimpleViT, 'diffusion': DiffusionClassifier}
        model_class = model_mapping.get(model_type_str.lower())
        if model_class is None:
            raise ValueError(f"Unsupported model type: '{model_type_str}'. Choose from {list(model_mapping.keys())}.")
        return model_class

    # In ClassificationPipeline class:
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_score: Optional[np.ndarray] = None,
                         detailed: bool = False) -> Dict[str, Any]:  # <<< ADDED detailed FLAG
        if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
        if y_score is not None and not isinstance(y_score, np.ndarray): y_score = np.array(y_score)

        metrics: Dict[str, Any] = {}  # Start empty
        class_metrics: Dict[str, Dict[str, float]] = {}  # Store per-class metrics here
        macro_metrics: Dict[str, float] = {}  # Store macro averages here
        detailed_data: Dict[str, Any] = {}  # Store detailed data here

        all_class_names = self.dataset_handler.classes
        num_classes_total = self.dataset_handler.num_classes
        if not all_class_names:
            logger.warning("Cannot compute metrics: class names not available.")
            return {'error': 'Class names missing'}

        # --- Overall Accuracy ---
        metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)

        # --- Per-Class Metrics ---
        present_class_labels = np.unique(np.concatenate((y_true, y_pred)))  # Consider labels in both true and pred
        all_precisions, all_recalls, all_specificities, all_f1s = [], [], [], []
        all_roc_aucs, all_pr_aucs = [], []
        # For detailed results
        all_roc_curves = {}  # Store {class_name: {'fpr': list, 'tpr': list, 'thresholds': list}}
        all_pr_curves = {}  # Store {class_name: {'precision': list, 'recall': list, 'thresholds': list}}

        can_compute_auc = y_score is not None and len(y_score.shape) == 2 and y_score.shape[
            1] == num_classes_total and len(y_score) == len(y_true)
        if y_score is not None and not can_compute_auc:
            logger.warning(f"y_score shape incompatible. Cannot compute AUCs.")

        for i, class_name in enumerate(all_class_names):
            class_label = self.dataset_handler.class_to_idx.get(class_name, i)
            # Check if class actually present in y_true for some metrics
            is_present = class_label in np.unique(y_true)
            # Check if class present in y_true OR y_pred for basic metrics
            is_present_or_predicted = class_label in present_class_labels

            if not is_present_or_predicted:
                # Class completely absent, record NaNs for basic metrics
                class_metrics[class_name] = {'precision': np.nan, 'recall': np.nan, 'specificity': np.nan,
                                             'f1': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan}
                all_precisions.append(np.nan);
                all_recalls.append(np.nan);
                all_specificities.append(np.nan);
                all_f1s.append(np.nan);
                all_roc_aucs.append(np.nan);
                all_pr_aucs.append(np.nan)
                if detailed:  # Add empty curve data if detailed
                    all_roc_curves[class_name] = {'fpr': [], 'tpr': [], 'thresholds': []}
                    all_pr_curves[class_name] = {'precision': [], 'recall': [], 'thresholds': []}
                continue

            true_is_class = (y_true == class_label)
            pred_is_class = (y_pred == class_label)

            precision = precision_score(true_is_class, pred_is_class, zero_division=0)
            recall = recall_score(true_is_class, pred_is_class, zero_division=0)  # Sensitivity
            f1 = f1_score(true_is_class, pred_is_class, zero_division=0)
            # Specificity = TN / (TN + FP) = Recall of negative class
            specificity = recall_score(~true_is_class, ~pred_is_class, zero_division=0)

            roc_auc, pr_auc = np.nan, np.nan
            roc_curve_data = {'fpr': [], 'tpr': [], 'thresholds': []}
            pr_curve_data = {'precision': [], 'recall': [], 'thresholds': []}

            if can_compute_auc and is_present:  # Need true class present for meaningful AUC/curves
                score_for_class = y_score[:, class_label]
                if len(np.unique(true_is_class)) > 1:  # AUC/curves require both +ve/-ve samples
                    try:
                        roc_auc = roc_auc_score(true_is_class, score_for_class)
                    except ValueError:
                        pass  # Ignore if only one class present after all
                    except Exception as e:
                        logger.warning(f"ROC AUC Error (Class {class_name}): {e}")

                    try:
                        prec, rec, pr_thresh = precision_recall_curve(true_is_class, score_for_class)
                        order = np.argsort(rec)  # Sort by recall for AUC calc
                        pr_auc = auc(rec[order], prec[order])
                        if detailed:
                            pr_curve_data['precision'] = prec.tolist()
                            pr_curve_data['recall'] = rec.tolist()
                            # Thresholds might be one less
                            pr_curve_data['thresholds'] = pr_thresh.tolist() if pr_thresh is not None else []
                    except ValueError:
                        pass
                    except Exception as e:
                        logger.warning(f"PR AUC Error (Class {class_name}): {e}")

                    if detailed:
                        try:
                            fpr, tpr, roc_thresh = roc_curve(true_is_class, score_for_class)
                            roc_curve_data['fpr'] = fpr.tolist()
                            roc_curve_data['tpr'] = tpr.tolist()
                            roc_curve_data['thresholds'] = roc_thresh.tolist()
                        except ValueError:
                            pass
                        except Exception as e:
                            logger.warning(f"ROC Curve Error (Class {class_name}): {e}")

            # Store per-class results
            class_metrics[class_name] = {
                'precision': precision, 'recall': recall, 'specificity': specificity, 'f1': f1,
                'roc_auc': roc_auc, 'pr_auc': pr_auc
            }
            # Append for macro calculation
            all_precisions.append(precision);
            all_recalls.append(recall);
            all_specificities.append(specificity);
            all_f1s.append(f1)
            all_roc_aucs.append(roc_auc);
            all_pr_aucs.append(pr_auc)
            # Store detailed curve data if requested
            if detailed:
                all_roc_curves[class_name] = roc_curve_data
                all_pr_curves[class_name] = pr_curve_data

        # --- Macro Averages ---
        macro_metrics['precision'] = float(np.nanmean(all_precisions))
        macro_metrics['recall'] = float(np.nanmean(all_recalls))
        macro_metrics['specificity'] = float(np.nanmean(all_specificities))
        macro_metrics['f1'] = float(np.nanmean(all_f1s))
        macro_metrics['roc_auc'] = float(np.nanmean(all_roc_aucs)) if can_compute_auc else np.nan
        macro_metrics['pr_auc'] = float(np.nanmean(all_pr_aucs)) if can_compute_auc else np.nan

        metrics['per_class'] = class_metrics
        metrics['macro_avg'] = macro_metrics

        # --- Add Detailed Data if Requested ---
        if detailed:
            detailed_data['y_true'] = y_true.tolist()  # Convert to list for JSON
            detailed_data['y_pred'] = y_pred.tolist()
            if y_score is not None:
                detailed_data['y_score'] = y_score.tolist()
            detailed_data['roc_curve_points'] = all_roc_curves
            detailed_data['pr_curve_points'] = all_pr_curves
            metrics['detailed_data'] = detailed_data

        logger.debug(
            f"Computed Metrics: Acc={metrics['overall_accuracy']:.4f}, Macro F1={metrics['macro_avg']['f1']:.4f}")
        return metrics

    def _save_results(self,
                      results_data: Dict[str, Any],
                      method_name: str,
                      run_id: str,  # Unique ID for this specific method run (e.g., "single_train_0")
                      method_params: Optional[Dict[str, Any]] = None
                      ) -> None:
        """
        Saves results to JSON (potentially cleaned based on self.save_detailed_results)
        in a method-specific subdirectory and updates a summary CSV in the main
        experiment directory.

        Args:
            results_data: The dictionary containing results from the method.
            method_name: The name of the pipeline method (e.g., "single_train").
            run_id: A unique identifier for this specific run (e.g., "single_train_0"),
                    used for the subdirectory name.
            method_params: Dictionary of key parameters used for this method run,
                           primarily for the summary CSV.
        """
        method_params = method_params or {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- Create Method-Specific Subdirectory ---
        # Structure: [base]/[dataset]/[model]/[timestamp_seed]/[run_id]/
        method_dir = self.experiment_dir / run_id  # self.experiment_dir is set in __init__
        method_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Saving results for {run_id} to: {method_dir}")

        # --- Define keys to remove if not detailed ---
        keys_to_remove_if_not_detailed = [
            'training_history',  # Can be large (single_train)
            'cv_results',  # Raw sklearn CV results dict (grid searches)
            'detailed_data',  # Raw preds/scores/curves (single_eval, cv_eval fold)
            'fold_detailed_results',  # List of detailed fold metrics (cv_eval)
            'full_params_used',  # Full config dict used (can be large/complex)
            'inner_cv_results',  # Raw inner CV results (nested_fixed_adapted - if re-enabled)
            # Add others here if needed
        ]
        # Store original cv_results before potential cleaning
        cv_results_full = results_data.get('cv_results')

        # --- Prepare data to save based on detail flag ---
        if not self.save_detailed_results:
            logger.debug(f"Cleaning results for run '{run_id}' (save_detailed_results=False).")
            results_to_save = {}
            for key, value in results_data.items():
                if key not in keys_to_remove_if_not_detailed:
                    results_to_save[key] = value
            # Special handling for cv_results: keep only mean/std scores if available
            if cv_results_full and isinstance(cv_results_full, dict):
                summary_cv = {}
                for k, v_list in cv_results_full.items():
                    # Check if key indicates a score list and is numeric-like after conversion
                    if k.startswith(
                            ('mean_test_', 'std_test_', 'mean_train_', 'std_train_', 'rank_test_')) and isinstance(
                            v_list, (list, np.ndarray)) and len(v_list) > 0:
                        try:
                            # Attempt to convert first element to float to check type
                            _ = float(v_list[0])
                            summary_cv[k] = v_list  # Keep the whole list/array for summary
                        except (ValueError, TypeError):
                            pass  # Skip non-numeric lists
                if summary_cv:
                    results_to_save['cv_results_summary'] = summary_cv
            logger.debug(f"Cleaned results keys: {list(results_to_save.keys())}")
        else:
            logger.debug(f"Saving full detailed results for run '{run_id}'.")
            results_to_save = results_data.copy()  # Save everything

        # Add method parameters used to the saved dict for context
        results_to_save['method_params_used'] = method_params

        # --- Save results JSON ---
        json_filename = f"{method_name}_results_{timestamp}.json"
        json_filepath = method_dir / json_filename

        try:
            def json_serializer(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj) if not np.isnan(obj) else None
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, (slice, type, Callable)):
                    return None  # Don't serialize certain types
                elif isinstance(obj, (torch.optim.Optimizer, nn.Module, EarlyStopping, LRScheduler, ValidSplit)):
                    return str(type(obj).__name__)
                try:
                    return json.JSONEncoder.default(None, obj)
                except TypeError:
                    return str(obj)  # Fallback

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=4, default=json_serializer)
            logger.info(f"Results JSON saved to: {json_filepath}")
        except OSError as oe:
            logger.error(f"OS Error saving results JSON {json_filepath}: {oe}", exc_info=True)
            try:  # Attempt fallback
                fallback_path = method_dir / f"{method_name}_fallback_{timestamp}.json"
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(results_to_save, f, indent=4, default=json_serializer)
                logger.warning(f"Saved results to fallback file: {fallback_path}")
            except Exception as fallback_e:
                logger.error(f"Fallback JSON save also failed: {fallback_e}")
        except Exception as e:
            logger.error(f"Failed to save results JSON {json_filepath}: {e}", exc_info=True)

        # --- Prepare and save summary CSV ---
        # Save summary CSV in the main experiment directory (one level up)
        csv_filepath = self.experiment_dir / f"summary_results_seed{RANDOM_SEED}.csv"
        try:
            # Extract primary metrics from the original results_data for summary
            macro_avg = results_data.get('macro_avg', {})
            # Use overall_accuracy if present (from single_eval/cv_eval), else mean test score
            overall_acc = results_data.get('overall_accuracy', results_data.get('mean_test_accuracy', np.nan))

            # --- Filter method_params for summary ---
            # Include only simple types and potentially key config like 'cv'
            summary_params = {}
            allowed_types = (str, int, float, bool)
            key_cv_params = ['cv', 'outer_cv', 'inner_cv', 'n_iter', 'internal_val_split_ratio']
            if method_params:
                for k, v in method_params.items():
                    if isinstance(v, allowed_types) or k in key_cv_params:
                        summary_params[k] = v
            # Add best params specifically if available (from grid search)
            if 'best_params' in results_data and isinstance(results_data['best_params'], dict):
                for k, v in results_data['best_params'].items():
                    if isinstance(v, allowed_types):
                        summary_params[f'best_{k}'] = v

            summary = {
                'method_run_id': run_id,
                'timestamp': timestamp,
                'accuracy': overall_acc,
                'macro_f1': macro_avg.get('f1', results_data.get('mean_test_f1_macro', np.nan)),
                'macro_roc_auc': macro_avg.get('roc_auc', results_data.get('mean_test_roc_auc_macro', np.nan)),
                'macro_pr_auc': macro_avg.get('pr_auc', results_data.get('mean_test_pr_auc_macro', np.nan)),
                'best_cv_score': results_data.get('best_score', np.nan),
                'best_epoch': results_data.get('best_epoch', np.nan),
                **summary_params  # Add filtered/key parameters
            }
            summary = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in summary.items()}

            df_summary = pd.DataFrame([summary])
            file_exists = csv_filepath.exists()
            df_summary.to_csv(csv_filepath, mode='a', header=not file_exists, index=False, encoding='utf-8')
            logger.info(f"Summary results updated in: {csv_filepath}")
        except Exception as e:
            logger.error(f"Failed to save summary results to CSV {csv_filepath}: {e}", exc_info=True)


    # --- Pipeline Methods ---

    def non_nested_grid_search(self,
                               param_grid: Dict[str, List],
                               cv: int = 5,
                               internal_val_split_ratio: Optional[float] = None,
                               n_iter: Optional[int] = None,  # For RandomizedSearch
                               method: str = 'grid',  # 'grid' or 'random'
                               scoring: str = 'accuracy',  # Sklearn scorer string or callable
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Performs non-nested hyperparameter search (Grid/RandomizedSearchCV)
        using the train+validation data. Refits the best model on the train+val
        data and updates the pipeline's main adapter. Does NOT evaluate on the
        test set itself. Works by passing paths directly.
        """
        method_lower = method.lower()
        search_type = "GridSearchCV" if method_lower == 'grid' else "RandomizedSearchCV"
        logger.info(f"Performing non-nested {search_type} with {cv}-fold CV.")
        logger.info(f"Parameter Grid/Dist: {param_grid}")
        logger.info(f"Scoring Metric: {scoring}")

        if method_lower == 'random' and n_iter is None: raise ValueError("n_iter required for random search.")
        if method_lower not in ['grid', 'random']: raise ValueError(f"Unsupported search method: {method}.")

        # --- Get Data (Paths/Labels) ---
        # Only need trainval data for fitting the search
        X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels()
        if not X_trainval: raise RuntimeError("Train+validation data is empty.")
        logger.info(f"Using {len(X_trainval)} samples for Train+Validation in GridSearchCV.")

        # --- Determine & Validate Internal Validation Split ---
        default_internal_val_fallback = 0.15
        val_frac_to_use = internal_val_split_ratio if internal_val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. Using default fallback: {default_internal_val_fallback:.3f}")
             val_frac_to_use = default_internal_val_fallback
        logger.info(f"Skorch internal validation split configured: {val_frac_to_use * 100:.1f}% of each CV fold's training data.")
        train_split_config = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED)
        # --- End Determine & Validate ---

        # --- Setup Skorch Estimator for Search ---
        adapter_config = self.model_adapter_config.copy()
        adapter_config['train_split'] = train_split_config # Always set a valid split
        adapter_config['verbose'] = 3 # Show epoch table

        # Remove config keys not needed by SkorchModelAdapter init
        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        estimator = SkorchModelAdapter(**adapter_config)

        # --- Setup Search ---
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        SearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        search_kwargs = {
            'estimator': estimator, 'cv': cv_splitter, 'scoring': scoring,
            'n_jobs': 1, 'verbose': 3, 'refit': True,  # Keep refit=True
            'return_train_score': True, 'error_score': 'raise'
        }
        if method_lower == 'grid':
            search_kwargs['param_grid'] = param_grid
        else:
            search_kwargs['param_distributions'] = param_grid
            search_kwargs['n_iter'] = n_iter
            search_kwargs['random_state'] = RANDOM_SEED
        search = SearchClass(**search_kwargs)

        # --- Run Search ---
        logger.info(f"Fitting {SearchClass.__name__} on train+validation data...")
        search.fit(X_trainval, y=np.array(y_trainval))  # Fit on trainval only
        logger.info(f"Search completed.")

        # --- Collect Results (Search Results Only) ---
        results = {
            'method': f"non_nested_{method_lower}_search",
            'params': {'cv': cv, 'n_iter': n_iter if method_lower == 'random' else 'N/A', 'method': method_lower,
                       'scoring': scoring},
            'best_params': search.best_params_,
            'best_score': search.best_score_,  # This is the CV score on trainval
            'cv_results': search.cv_results_,
            # --- Test set evaluation REMOVED from this method ---
            'test_set_evaluation': {'message': 'Test set evaluation not performed in this method.'},
            'accuracy': np.nan,  # Indicate no test accuracy from this step
            'macro_avg': {}  # Indicate no test metrics from this step
        }

        # --- Update the pipeline's main adapter with the refitted best model ---
        if hasattr(search, 'best_estimator_'):
            logger.info("Updating main pipeline adapter with the best model found and refit by GridSearchCV.")
            self.model_adapter = search.best_estimator_
            if not self.model_adapter.initialized_:
                logger.warning("Refit best estimator seems not initialized, attempting initialize.")
                try:
                    self.model_adapter.initialize()
                except Exception as init_err:
                    logger.error(f"Failed to initialize refit estimator: {init_err}", exc_info=True)
        else:
            logger.warning("GridSearchCV did not produce a 'best_estimator_'. Pipeline adapter not updated.")

        # --- Save Results (Search results + Best Params) ---
        if save_results:
             # Prepare params for saving (include best params found)
             save_params = results['params'].copy()
             save_params.update({f"best_{k}": v for k, v in results.get('best_params', {}).items()})
             # Pass only simple params relevant to the *search itself* for filename/summary
             filename_params = {k: v for k, v in save_params.items() if k in ['cv', 'n_iter', 'method', 'scoring']}
             run_id_for_save = f"{method_lower}_{datetime.now().strftime('%H%M%S')}" # Generate ID
             # --- FIX: Use method_params= ---
             self._save_results(results, f"non_nested_{method_lower}_search",
                                run_id=run_id_for_save,
                                method_params=filename_params) # Use correct keyword

        logger.info(f"Non-nested {method_lower} search finished. Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"Best parameters found: {search.best_params_}")
        logger.info(f"Pipeline adapter has been updated with the best model refit on train+validation data.")
        return results


    def nested_grid_search(self,
                           param_grid: Dict[str, List],
                           outer_cv: int = 5,
                           inner_cv: int = 3,
                           internal_val_split_ratio: Optional[float] = None,
                           n_iter: Optional[int] = None, # For RandomizedSearch
                           method: str = 'grid', # 'grid' or 'random'
                           scoring: str = 'accuracy', # Sklearn scorer string or callable
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Performs nested cross-validation for unbiased performance estimation.
        Uses the full dataset (respecting force_flat_for_fixed_cv).
        Passes paths to sklearn cross_validate. Skorch adapter handles transforms.
        """
        method_lower = method.lower()
        search_type = "GridSearchCV" if method_lower == 'grid' else "RandomizedSearchCV"
        logger.info(f"Performing nested {search_type} search.")
        logger.info(f"  Outer CV folds: {outer_cv}, Inner CV folds: {inner_cv}")
        logger.info(f"  Parameter Grid/Dist: {param_grid}")
        logger.info(f"  Scoring Metric: {scoring}")

        # --- Check Compatibility ---
        if self.dataset_handler.structure == DatasetStructure.FIXED and not self.force_flat_for_fixed_cv:
             # Provide a specific path for FIXED datasets without the flag
             raise ValueError(f"nested_grid_search requires a FLAT dataset structure "
                              f"or a FIXED structure with force_flat_for_fixed_cv=True.")

        # --- Standard Nested CV (FLAT or FIXED with force_flat_for_fixed_cv=True) ---
        logger.info("Proceeding with standard nested CV using the full dataset.")
        try:
            X_full, y_full = self.dataset_handler.get_full_paths_labels_for_cv()
            if not X_full: raise RuntimeError("Full dataset for CV is empty.")
            logger.info(f"Using {len(X_full)} samples for outer cross-validation.")
            y_full_np = np.array(y_full) # Needed for stratification
        except Exception as e:
            logger.error(f"Failed to get full dataset paths/labels for nested CV: {e}", exc_info=True)
            raise

        # --- Determine & Validate Internal Validation Split ---
        default_internal_val_fallback = 0.15
        val_frac_to_use = internal_val_split_ratio if internal_val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. "
                            f"Using default fallback: {default_internal_val_fallback:.3f} for inner loop fits.")
             val_frac_to_use = default_internal_val_fallback
        # --- End Determine & Validate ---

        logger.info(f"Inner loop Skorch validation split configured: {val_frac_to_use * 100:.1f}% of inner CV fold's training data.")
        train_split_config = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED)

        # --- Setup Inner Search Object ---
        adapter_config = self.model_adapter_config.copy()
        adapter_config['train_split'] = train_split_config # Always set a valid split
        adapter_config['verbose'] = 3

        # Remove config keys not needed by SkorchModelAdapter init
        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        base_estimator = SkorchModelAdapter(**adapter_config)

        inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_SEED)
        InnerSearchClass = GridSearchCV if method_lower == 'grid' else RandomizedSearchCV
        inner_search_kwargs = {
            'estimator': base_estimator, 'cv': inner_cv_splitter, 'scoring': scoring,
            'n_jobs': 1, 'verbose': 3, 'refit': True, 'error_score': 'raise'
        }
        if method_lower == 'grid': inner_search_kwargs['param_grid'] = param_grid
        else:
            inner_search_kwargs['param_distributions'] = param_grid
            inner_search_kwargs['n_iter'] = n_iter
            inner_search_kwargs['random_state'] = RANDOM_SEED
        inner_search = InnerSearchClass(**inner_search_kwargs)

        # --- Setup Outer CV ---
        outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED + 1) # Different seed

        # Define multiple scorers for cross_validate
        scoring_dict = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            # Add others if needed (e.g., roc_auc requires predict_proba)
            # 'roc_auc_ovr': make_scorer(roc_auc_score, average='macro', multi_class='ovr', needs_proba=True)
        }

        # --- Run Nested CV using cross_validate ---
        logger.info(f"Running standard nested CV using cross_validate...")
        try:
             cv_results = cross_validate(
                 inner_search, X_full, y_full_np, cv=outer_cv_splitter, scoring=scoring_dict,
                 return_estimator=False, # Don't return estimators by default (memory)
                 n_jobs=1, verbose=3, error_score='raise'
             )
             logger.info("Nested cross-validation finished.")

             # --- Process and Save Results ---
             results = {
                 'method': f"nested_{method_lower}_search",
                 'params': {'outer_cv': outer_cv, 'inner_cv': inner_cv, 'n_iter': n_iter if method_lower=='random' else 'N/A', 'method': method_lower, 'scoring': scoring, 'forced_flat': self.force_flat_for_fixed_cv},
                 'outer_cv_scores': {k: v.tolist() for k, v in cv_results.items() if k.startswith('test_')},
                 'mean_test_accuracy': float(np.mean(cv_results['test_accuracy'])),
                 'std_test_accuracy': float(np.std(cv_results['test_accuracy'])),
                 'mean_test_f1_macro': float(np.mean(cv_results['test_f1_macro'])),
                 'std_test_f1_macro': float(np.std(cv_results['test_f1_macro'])),
                 # 'best_params_per_fold': "Estimators not returned" # Or return them if needed
             }
             # For summary file
             results['accuracy'] = results['mean_test_accuracy']
             results['macro_avg'] = {'f1': results['mean_test_f1_macro']}

             if save_results:
                 # Clean params for saving filename/summary
                 # Pass the results['params'] which contains outer_cv, inner_cv etc.
                 run_id_for_save = f"{method_lower}_{datetime.now().strftime('%H%M%S')}"  # Generate ID
                 # --- FIX: Use method_params= ---
                 self._save_results(results, f"nested_{method_lower}_search",
                                    run_id=run_id_for_save,
                                    method_params=results['params'])  # Use correct keyword

             logger.info(f"Nested CV Results (avg over {outer_cv} outer folds):")
             logger.info(f"  Mean Test Accuracy: {results['mean_test_accuracy']:.4f} +/- {results['std_test_accuracy']:.4f}")
             logger.info(f"  Mean Test Macro F1: {results['mean_test_f1_macro']:.4f} +/- {results['std_test_f1_macro']:.4f}")
             return results

        except Exception as e:
             logger.error(f"Standard nested CV failed: {e}", exc_info=True)
             # Return error information
             return {
                'method': f"nested_{method_lower}_search",
                'params': {'outer_cv': outer_cv, 'inner_cv': inner_cv, 'n_iter': n_iter if method_lower=='random' else 'N/A', 'method': method_lower, 'scoring': scoring},
                'error': str(e)
             }

    def cv_model_evaluation(self,
                            cv: int = 5,
                            internal_val_split_ratio: Optional[float] = None,
                            params: Optional[Dict] = None,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Performs K-Fold CV for evaluation using fixed hyperparameters.
        Uses full dataset (respecting force_flat_for_fixed_cv). Skorch adapter's
        internal ValidSplit is used for monitoring within each fold's training.
        """
        logger.info(f"Performing {cv}-fold CV for evaluation with fixed parameters.")

        # --- Check Compatibility ---
        if self.dataset_handler.structure == DatasetStructure.FIXED and not self.force_flat_for_fixed_cv:
            raise ValueError("cv_model_evaluation requires a FLAT dataset structure or a FIXED structure with force_flat_for_fixed_cv=True.")

        # --- Get Full Data ---
        try:
            X_full, y_full = self.dataset_handler.get_full_paths_labels_for_cv()
            if not X_full: raise RuntimeError("Full dataset for CV is empty.")
            y_full_np = np.array(y_full) # Needed for stratification
        except Exception as e:
            logger.error(f"Failed to get full dataset paths/labels for CV evaluation: {e}", exc_info=True)
            raise

        # --- Hyperparameters for this evaluation ---
        # Use provided params or fall back to pipeline defaults
        eval_params = self.model_adapter_config.copy() # Start with base config
        if params: # Override with explicitly passed params for this run
             logger.info(f"Using provided parameters for CV evaluation: {params}")
             eval_params.update(params)
        else:
             logger.info(f"Using pipeline default parameters for CV evaluation.")
        # Extract non-skorch params if necessary
        module_dropout_rate = eval_params.pop('module__dropout_rate', None)
        # Ensure essential adapter params are present
        eval_params.setdefault('module', self._get_model_class(self.model_type))
        eval_params.setdefault('module__num_classes', self.dataset_handler.num_classes)
        if module_dropout_rate is not None: eval_params['module__dropout_rate'] = module_dropout_rate

        # --- Determine & Validate Internal Validation Split ---
        default_internal_val_fallback = 0.15
        val_frac_to_use = internal_val_split_ratio if internal_val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if not 0.0 < val_frac_to_use < 1.0:
             logger.warning(f"Provided internal validation split ratio ({val_frac_to_use:.3f}) is invalid. "
                            f"Using default fallback: {default_internal_val_fallback:.3f} for fold fits.")
             val_frac_to_use = default_internal_val_fallback
        # --- End Determine & Validate ---

        logger.info(f"Skorch internal validation split configured: {val_frac_to_use * 100:.1f}% of each CV fold's training data.")
        # Note: We create the actual ValidSplit inside the loop with fold-specific seed


        # --- Setup CV Strategy ---
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []
        fold_histories = [] # Store history from each fold
        fold_detailed_results = []  # <<< LIST FOR DETAILED FOLD RESULTS

        # --- Manual Outer CV Loop ---
        for fold_idx, (outer_train_indices, outer_test_indices) in enumerate(cv_splitter.split(X_full, y_full_np)):
            logger.info(f"--- Starting CV Evaluation Fold {fold_idx + 1}/{cv} ---")

            # --- Get Outer Fold Data (Paths/Labels) ---
            X_outer_train = [X_full[i] for i in outer_train_indices]
            y_outer_train = y_full_np[outer_train_indices]
            X_test        = [X_full[i] for i in outer_test_indices]
            y_test        = y_full_np[outer_test_indices]
            logger.debug(f"Outer split: {len(X_outer_train)} train samples, {len(X_test)} test samples.")

            if not X_outer_train or not X_test:
                 logger.warning(f"Fold {fold_idx+1} resulted in empty train or test set. Skipping.")
                 fold_results.append({'accuracy': np.nan, 'f1_macro': np.nan}) # Append NaNs
                 continue

            # --- Setup Estimator for this Fold ---
            fold_adapter_config = eval_params.copy()
            # Enable internal validation split for monitoring (e.g., EarlyStopping)
            fold_adapter_config['train_split'] = ValidSplit(cv=val_frac_to_use, stratified=True, random_state=RANDOM_SEED + fold_idx) # Fold specific seed
            fold_adapter_config['verbose'] = 3 # Show progress per fold

            n_outer_train = len(X_outer_train)
            n_inner_val = int(n_outer_train * val_frac_to_use) # Use the validated fraction
            n_inner_train = n_outer_train - n_inner_val
            logger.debug(
                f"Fold {fold_idx + 1}: Internal split for monitoring: ~{n_inner_train} train / ~{n_inner_val} valid.")

            # Remove config keys not needed by SkorchModelAdapter init
            fold_adapter_config.pop('patience_cfg', None)
            fold_adapter_config.pop('monitor_cfg', None)
            fold_adapter_config.pop('lr_policy_cfg', None)
            fold_adapter_config.pop('lr_patience_cfg', None)

            estimator_fold = SkorchModelAdapter(**fold_adapter_config)

            # --- Fit on Outer Train (Skorch uses internal split for validation) ---
            logger.info(f"Fitting model for fold {fold_idx + 1}...")
            try:
                estimator_fold.fit(X_outer_train, y=y_outer_train)
                # Check if history exists and has validation scores
                if hasattr(estimator_fold, 'history_') and estimator_fold.history_:
                     fold_histories.append(estimator_fold.history)
                else: fold_histories.append(None) # Append None if no history

            except Exception as fit_err:
                 logger.error(f"Fit failed for fold {fold_idx + 1}: {fit_err}", exc_info=True)
                 fold_results.append({'accuracy': np.nan, 'f1_macro': np.nan})
                 fold_histories.append(None)
                 continue # Skip scoring for this fold

            # --- Evaluate on Outer Test Set ---
            logger.info(f"Evaluating model on outer test set for fold {fold_idx + 1}...")
            try:
                 y_pred_test = estimator_fold.predict(X_test)
                 y_score_test = estimator_fold.predict_proba(X_test)
                 # --- Compute detailed metrics per fold ---
                 fold_metrics = self._compute_metrics(y_test, y_pred_test, y_score_test,
                                                      detailed=self.save_detailed_results) # <<< PASS FLAG
                 fold_results.append({ # Append summary stats for aggregation
                     'accuracy': fold_metrics.get('overall_accuracy', np.nan),
                     'f1_macro': fold_metrics.get('macro_avg', {}).get('f1', np.nan)
                 })
                 if self.save_detailed_results: # Store full metrics dict if detailed
                      fold_detailed_results.append(fold_metrics)
                 # --- End Compute ---
                 logger.info(f"Fold {fold_idx + 1} Test Scores: Acc={fold_metrics.get('overall_accuracy', np.nan):.4f}, "
                             f"F1={fold_metrics.get('macro_avg', {}).get('f1', np.nan):.4f}")
            except Exception as score_err:
                 logger.error(f"Scoring failed for fold {fold_idx + 1}: {score_err}", exc_info=True)
                 fold_results.append({'accuracy': np.nan, 'f1_macro': np.nan})

        # --- Aggregate Results ---
        if not fold_results:
             logger.error("CV evaluation failed for all folds.")
             return {'method': 'cv_model_evaluation', 'params': params or {}, 'error': 'All folds failed.'}

        df_results = pd.DataFrame(fold_results)
        results = {
             'method': 'cv_model_evaluation',
             'params': eval_params, # Store effective params used
             'cv_scores_summary': df_results.to_dict(orient='list'), # Summary scores
             'mean_test_accuracy': float(df_results['accuracy'].mean()),
             'std_test_accuracy': float(df_results['accuracy'].std()),
             'mean_test_f1_macro': float(df_results['f1_macro'].mean()),
             'std_test_f1_macro': float(df_results['f1_macro'].std()),
            'fold_detailed_results': fold_detailed_results,  # Add detailed list
             # 'fold_histories': fold_histories # Optional: Can be very large
        }
        results['accuracy'] = results['mean_test_accuracy'] # For summary
        results['macro_avg'] = {'f1': results['mean_test_f1_macro']} # For summary

        if save_results:
            # Pass detailed flag and run_id
            simple_params = {k: v for k, v in eval_params.items() if isinstance(v, (str, int, float, bool))}
            simple_params['cv'] = cv  # Add CV folds number
            simple_params['internal_val_split_ratio'] = val_frac_to_use  # Add frac used
            run_id_for_save = f"cv_model_evaluation_{datetime.now().strftime('%H%M%S')}"  # Generate ID
            # --- FIX: Use method_params= ---
            self._save_results(results, "cv_model_evaluation",
                               run_id=run_id_for_save,
                               method_params=simple_params)  # Use correct keyword

        logger.info(f"CV Evaluation Summary (Avg over {len(fold_results)} folds):")
        logger.info(f"  Accuracy: {results['mean_test_accuracy']:.4f} +/- {results['std_test_accuracy']:.4f}")
        logger.info(f"  Macro F1: {results['mean_test_f1_macro']:.4f} +/- {results['std_test_f1_macro']:.4f}")

        return results


    def single_train(self,
                     max_epochs: Optional[int] = None,
                     lr: Optional[float] = None,
                     batch_size: Optional[int] = None,
                     # Add other tunable params like weight_decay, dropout_rate here if needed
                     optimizer__weight_decay: Optional[float] = None,
                     module__dropout_rate: Optional[float] = None,
                     val_split_ratio: Optional[float] = None, # Override handler's default split
                     save_model: bool = True,
                     save_results: bool = True) -> Dict[str, Any]:
        """
        Performs a single training run using a train/validation split.
        Manually creates the split and uses Skorch with PredefinedSplit.
        """
        logger.info("Starting single training run...")

        # --- Get Train+Validation Data ---
        X_trainval, y_trainval = self.dataset_handler.get_train_val_paths_labels()
        if not X_trainval: raise RuntimeError("Train+validation data is empty.")
        y_trainval_np = np.array(y_trainval)

        # --- Determine Validation Split ---
        current_val_split_ratio = val_split_ratio if val_split_ratio is not None else self.dataset_handler.val_split_ratio
        if current_val_split_ratio <= 0 or current_val_split_ratio >= 1.0:
            logger.warning("Validation split ratio is <= 0 or >= 1. Training on full trainval set without validation.")
            X_train, y_train = X_trainval, y_trainval_np
            X_val, y_val = [], np.array([])
            X_fit, y_fit = X_train, y_train
            train_split_config = None
            n_train, n_val = len(y_train), 0
        elif len(np.unique(y_trainval_np)) < 2:
             logger.warning("Only one class present in trainval data. Cannot stratify split. Training without validation.")
             X_train, y_train = X_trainval, y_trainval_np
             X_val, y_val = [], np.array([])
             X_fit, y_fit = X_train, y_train
             train_split_config = None
             n_train, n_val = len(y_train), 0
        else:
            try:
                 train_indices, val_indices = train_test_split(
                     np.arange(len(X_trainval)), test_size=current_val_split_ratio,
                     stratify=y_trainval_np, random_state=RANDOM_SEED)
            except ValueError as e:
                 logger.warning(f"Stratified train/val split failed ({e}). Using non-stratified split.")
                 train_indices, val_indices = train_test_split(
                     np.arange(len(X_trainval)), test_size=current_val_split_ratio,
                     random_state=RANDOM_SEED)

            X_train = [X_trainval[i] for i in train_indices]
            y_train = y_trainval_np[train_indices]
            X_val = [X_trainval[i] for i in val_indices]
            y_val = y_trainval_np[val_indices]
            n_train, n_val = len(y_train), len(y_val)

            # Combine for skorch fit and create PredefinedSplit
            X_fit = X_train + X_val # Combine lists of paths
            y_fit = np.concatenate((y_train, y_val))
            test_fold = np.full(len(X_fit), -1, dtype=int) # -1 indicates train
            test_fold[n_train:] = 0 # 0 indicates validation fold
            ps = PredefinedSplit(test_fold=test_fold)
            train_split_config = ValidSplit(cv=ps, stratified=False) # Wrap ps

        logger.info(f"Using split: {n_train} train / {n_val} validation samples.")

        # --- Configure Model Adapter ---
        adapter_config = self.model_adapter_config.copy()
        # Override params for this run
        if max_epochs is not None: adapter_config['max_epochs'] = max_epochs
        if lr is not None: adapter_config['lr'] = lr
        if batch_size is not None: adapter_config['batch_size'] = batch_size
        if optimizer__weight_decay is not None: adapter_config['optimizer__weight_decay'] = optimizer__weight_decay
        if module__dropout_rate is not None: adapter_config['module__dropout_rate'] = module__dropout_rate
        # Set the train split strategy (None or PredefinedSplit via ValidSplit)
        adapter_config['train_split'] = train_split_config
        # Ensure callbacks use 'valid_loss' if validation split exists
        if train_split_config is None:
             # No validation: monitor train_loss? Or remove callbacks? Remove EarlyStopping/LRScheduler.
             logger.warning("No validation set, removing EarlyStopping and LRScheduler callbacks.")

        adapter_config['verbose'] = 3 # Show epoch progress

        adapter_config.pop('patience_cfg', None)
        adapter_config.pop('monitor_cfg', None)
        adapter_config.pop('lr_policy_cfg', None)
        adapter_config.pop('lr_patience_cfg', None)

        adapter_for_train = SkorchModelAdapter(**adapter_config)

        # --- Train Model ---
        logger.info("Fitting model...")
        adapter_for_train.fit(X_fit, y=y_fit) # Pass combined data

        # --- Collect Results ---
        history = adapter_for_train.history
        results = {'method': 'single_train', 'params': adapter_config} # Store effective config
        best_epoch_info = {}
        valid_loss_key = 'valid_loss' # Metric monitored by callbacks
        validation_was_run = train_split_config is not None and history and valid_loss_key in history[-1]  # Access last epoch dict directly
        if validation_was_run:
            try:
                # Find the index of the epoch with the best validation score
                scores = [epoch.get(valid_loss_key, np.inf if valid_loss_key.endswith('_loss') else -np.inf) for epoch
                          in history]
                if valid_loss_key.endswith('_loss'):  # Lower is better for loss
                    best_idx = np.argmin(scores)
                else:  # Higher is better for accuracy etc.
                    best_idx = np.argmax(scores)

                # Convert best_idx to standard Python int before using it to index history
                best_idx_int = int(best_idx)

                # Handle case where history might be empty or scores invalid
                if best_idx_int < len(history):
                    best_epoch_hist = history[best_idx_int]  # Use python int index
                    actual_best_epoch_num = best_epoch_hist.get('epoch')  # Get actual epoch number

                    best_epoch_info = {
                        'best_epoch': actual_best_epoch_num,
                        'best_valid_metric_value': float(best_epoch_hist.get(valid_loss_key, np.nan)),
                        'valid_metric_name': valid_loss_key,
                        'train_loss_at_best': float(best_epoch_hist.get('train_loss', np.nan)),
                        'train_acc_at_best': float(best_epoch_hist.get('train_acc', np.nan)),
                        'valid_acc_at_best': float(best_epoch_hist.get('valid_acc', np.nan)),
                    }
                    logger.info(
                        f"Training finished. Best validation performance found at Epoch {best_epoch_info['best_epoch']} "
                        f"({valid_loss_key}={best_epoch_info['best_valid_metric_value']:.4f})")
                else:
                    logger.error("Could not determine best epoch index from history scores.")
                    validation_was_run = False  # Fallback to last epoch logic
            except Exception as e:
                logger.error(f"Error processing history for best epoch: {e}", exc_info=True)
                validation_was_run = False  # Fallback to last epoch

        if not validation_was_run: # No validation or error processing history
             if history:
                  last_epoch_hist = history[-1]
                  last_epoch_num = last_epoch_hist.get('epoch', len(history))
             else: last_epoch_hist = {}; last_epoch_num = 0; logger.error("History empty after fit.")

             best_epoch_info = {
                 'best_epoch': last_epoch_num,
                 'best_valid_metric_value': np.nan, 'valid_metric_name': valid_loss_key,
                 'train_loss_at_best': float(last_epoch_hist.get('train_loss', np.nan)),
                 'train_acc_at_best': float(last_epoch_hist.get('train_acc', np.nan)),
                 'valid_acc_at_best': np.nan, # No valid acc
             }
             logger.warning(f"No validation performed or error finding best epoch. Reporting last epoch ({last_epoch_num}) stats.")

        results.update(best_epoch_info)
        results['training_history'] = history.to_list() if history else []

        # --- Save Model ---
        model_path = None
        if save_model:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                val_metric_val = results.get('best_valid_metric_value', np.nan)
                val_metric_str = f"val_{valid_loss_key.replace('_','-')}{val_metric_val:.4f}" if not np.isnan(val_metric_val) else "no_val"
                model_filename = f"{self.model_type}_epoch{results.get('best_epoch', 0)}_{val_metric_str}_{timestamp}.pt"
                model_path = self.results_dir / model_filename
                # Save using skorch helper or directly
                # adapter_for_train.save_params(f_params=model_path)
                torch.save(adapter_for_train.module_.state_dict(), model_path)
                logger.info(f"Model state_dict saved to: {model_path}")
                results['saved_model_path'] = str(model_path)
            except Exception as e:
                 logger.error(f"Failed to save model: {e}", exc_info=True)
                 results['saved_model_path'] = None

        # Update the main pipeline adapter with the trained one
        self.model_adapter = adapter_for_train
        logger.info("Main pipeline model adapter updated with the model from single_train.")

        # Add dummy metrics for saving compatibility if needed
        results['accuracy'] = results.get('valid_acc_at_best', np.nan) # Use valid acc if available
        results['macro_avg'] = {}

        # Add effective adapter config to results
        results['full_params_used'] = adapter_config  # Store the config used

        if save_results:
            # Prepare simple params for summary/filename base
            simple_params = {k: v for k, v in adapter_config.items() if isinstance(v, (str, int, float, bool))}
            run_id_for_save = f"single_train_{datetime.now().strftime('%H%M%S')}"  # Generate ID here
            self._save_results(results, "single_train",
                               run_id=run_id_for_save,
                               method_params=simple_params  # Use correct keyword
                               )
        return results


    def single_eval(self, save_results: bool = True) -> Dict[str, Any]:
        """ Evaluates the current model adapter on the test set. """
        logger.info("Starting model evaluation on the test set...")

        if not self.model_adapter.initialized_:
             raise RuntimeError("Model adapter not initialized. Train or load first.")

        # --- Get Test Data ---
        X_test, y_test = self.dataset_handler.get_test_paths_labels()
        if not X_test:
             logger.warning("Test set is empty. Skipping evaluation.")
             return {'method': 'single_eval', 'message': 'Test set empty, evaluation skipped.'}
        y_test_np = np.array(y_test)

        # --- Make Predictions ---
        logger.info(f"Evaluating on {len(X_test)} test samples...")
        try:
             y_pred_test = self.model_adapter.predict(X_test)
             y_score_test = self.model_adapter.predict_proba(X_test)
        except Exception as e:
             logger.error(f"Prediction failed during single_eval: {e}", exc_info=True)
             raise RuntimeError("Failed to get predictions from model adapter.") from e

        # --- Compute Metrics (pass detailed flag) ---
        metrics = self._compute_metrics(y_test_np, y_pred_test, y_score_test,
                                     detailed=self.save_detailed_results)  # <<< PASS FLAG
        results = {'method': 'single_eval', 'params': {}, **metrics}

        if save_results:
            run_id_for_save = f"single_eval_{datetime.now().strftime('%H%M%S')}"
            self._save_results(results, "single_eval",
                               method_params=results['params'],
                               run_id=run_id_for_save
                               )

        return results


    def load_model(self, model_path: Union[str, Path]) -> None:
        """ Loads a state_dict into the pipeline's model adapter. """
        # (Keep logic from code_v6, it should work with the initialized adapter)
        model_path = Path(model_path)
        logger.info(f"Loading model state_dict from: {model_path}")
        if not model_path.is_file(): raise FileNotFoundError(f"Model file not found at {model_path}")

        if not self.model_adapter.initialized_:
            logger.debug("Initializing skorch adapter before loading state_dict...")
            try: self.model_adapter.initialize()
            except Exception as e: raise RuntimeError("Could not initialize model adapter for loading.") from e

        if not hasattr(self.model_adapter, 'module_') or not isinstance(self.model_adapter.module_, nn.Module):
             raise RuntimeError("Adapter missing internal nn.Module ('module_'). Cannot load state_dict.")

        try:
            map_location = self.model_adapter.device
            state_dict = torch.load(model_path, map_location=map_location)
            logger.debug(f"State_dict loaded successfully to device '{map_location}'.")
            self.model_adapter.module_.load_state_dict(state_dict)
            self.model_adapter.module_.eval() # Set to eval mode
            logger.info("Model state_dict loaded successfully into the model adapter.")
        except Exception as e:
            logger.error(f"Failed to load state_dict into model: {e}", exc_info=True)
            if isinstance(e, RuntimeError) and "size mismatch" in str(e):
                 logger.error("Architecture mismatch likely. Ensure loaded weights match the current model config.")
            raise RuntimeError("Error loading state_dict into the model adapter module.") from e


# --- Pipeline Executor ---

class PipelineExecutor:
    """
    Executes a sequence of classification pipeline methods.
    Handles parameter passing and compatibility checks.
    """
    def __init__(self,
                 dataset_path: Union[str, Path],
                 model_type: str = 'cnn',
                 model_load_path: Optional[Union[str, Path]] = None,
                 results_dir: Union[str, Path] = 'results',
                 save_detailed_results: bool = False, # <<< ADDED here
                 methods: List[Tuple[str, Dict[str, Any]]]= None,
                 # Pipeline config params passed down
                 img_size: Tuple[int, int] = (224, 224),
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 data_augmentation: bool = True,
                 force_flat_for_fixed_cv: bool = False,
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 optimizer__weight_decay: float = 0.01,
                 module__dropout_rate: Optional[float] = None
                 ):
        global logger # <<< Access the global logger instance

        # --- Initialize Pipeline FIRST to get experiment_dir ---
        self.pipeline = ClassificationPipeline(
            dataset_path=dataset_path, model_type=model_type, model_load_path=model_load_path,
            results_dir=results_dir,
            save_detailed_results=save_detailed_results, # <<< Pass flag
            img_size=img_size, val_split_ratio=val_split_ratio,
            test_split_ratio_if_flat=test_split_ratio_if_flat, data_augmentation=data_augmentation,
            force_flat_for_fixed_cv=force_flat_for_fixed_cv, lr=lr, max_epochs=max_epochs,
            batch_size=batch_size, patience=patience,
            optimizer__weight_decay=optimizer__weight_decay, module__dropout_rate=module__dropout_rate
        )
        # --- End Pipeline Init ---

        # --- Configure Logger AFTER pipeline init ---
        logger_name = 'ImgClassPipe'
        experiment_log_dir = self.pipeline.experiment_dir # Get dir from pipeline
        logger = setup_logger( # Reconfigure the global logger
             name=logger_name,
             log_dir=experiment_log_dir,
             log_filename=f"experiment_{Path(experiment_log_dir).name}.log", # Unique log filename per experiment
             level=logging.DEBUG,
             use_colors=True
         )
        print(get_log_header(use_colors=True)) # Print header after setup
        logger.info(f"--- Starting Experiment Run ---")
        logger.info(f"Pipeline Executor initialized for model '{model_type}' on dataset '{Path(dataset_path).name}'")
        logger.info(f"Results base directory: {self.pipeline.experiment_dir}")
        # --- End Logger Config ---

        self.methods_to_run = methods if methods is not None else []
        self.all_results: Dict[str, Any] = {}
        try: # Validate methods after logger is fully set up
             self._validate_methods()
        except ValueError as e:
             logger.error(f"Method validation failed: {e}")
             raise # Re-raise after logging
        method_names = [m[0] for m in self.methods_to_run]
        logger.info(f"Executor configured to run methods: {', '.join(method_names)}")

    def _validate_methods(self) -> None:
        """ Basic validation of method names and parameter types. """
        valid_method_names = [
            'non_nested_grid_search', 'nested_grid_search', 'cv_model_evaluation',
            'single_train', 'single_eval', 'load_model'
        ]
        for i, (method_name, params) in enumerate(self.methods_to_run):
            if not isinstance(method_name, str) or method_name not in valid_method_names:
                 raise ValueError(f"Invalid method name '{method_name}' at index {i}. Valid: {valid_method_names}")
            if not isinstance(params, dict):
                 raise ValueError(f"Parameters for method '{method_name}' at index {i} must be a dict.")
            # Specific parameter checks (examples)
            if 'search' in method_name and 'param_grid' not in params:
                 raise ValueError(f"Method '{method_name}' requires 'param_grid'.")
            if method_name == 'load_model' and 'model_path' not in params:
                 raise ValueError(f"Method 'load_model' requires 'model_path'.")
            # Compatibility checks are now mostly done *inside* the methods themselves.
        logger.debug("Basic method validation successful.")

    # Add helper method to get previous results
    def _get_previous_result(self, step_index: int) -> Optional[Dict[str, Any]]:
        """Gets the results dict from a previous step if available."""
        if step_index < 0 or step_index >= len(self.methods_to_run):
            return None
        prev_method_name, _ = self.methods_to_run[step_index]
        run_id = f"{prev_method_name}_{step_index}"
        return self.all_results.get(run_id)

    def run(self) -> Dict[str, Any]:
        """ Executes the configured sequence of pipeline methods. """
        self.all_results = {}
        logger.info("Starting execution of pipeline methods...")
        start_time_total = time.time()

        for i, (method_name, params) in enumerate(self.methods_to_run):
            run_id = f"{method_name}_{i}"  # <<< USE THIS AS THE UNIQUE ID
            logger.info(f"--- Running Method {i + 1}/{len(self.methods_to_run)}: {method_name} ({run_id}) ---")

            # --- Parameter Injection Logic ---
            current_params = params.copy()  # Work with a copy
            use_best_params_key = 'use_best_params_from_step'
            if use_best_params_key in current_params:
                prev_step_index = current_params.pop(use_best_params_key)
                if not isinstance(prev_step_index, int) or prev_step_index >= i:
                    logger.error(f"Invalid previous step index '{prev_step_index}' for '{method_name}'.")
                    self.all_results[run_id] = {"error": f"Invalid '{use_best_params_key}' value."}
                    break
                logger.info(
                    f"Injecting 'best_params' from step {prev_step_index} ({self.methods_to_run[prev_step_index][0]}) into params for '{method_name}'.")
                prev_result = self._get_previous_result(prev_step_index)

                if prev_result and isinstance(prev_result, dict) and 'best_params' in prev_result and isinstance(
                        prev_result['best_params'], dict):
                    best_params = prev_result['best_params']
                    logger.info(f"  Injecting best params: {best_params}")

                    # --- MODIFIED MERGING ---
                    # Ensure 'params' dict exists in current_params for methods like cv_model_evaluation
                    if 'params' not in current_params:
                        current_params['params'] = {}  # Create if missing

                    if isinstance(current_params['params'], dict):
                        # Create final nested params: Start with best_params, then overwrite
                        # with any specific 'params' the user provided for this step.
                        final_nested_params = best_params.copy()
                        final_nested_params.update(current_params['params'])  # Apply user overrides
                        current_params['params'] = final_nested_params
                    else:
                        logger.error(f"'params' key in config for step {i} is not a dict. Cannot inject best_params.")
                        self.all_results[run_id] = {"error": "'params' key is not a dictionary."}
                        break
                    # --- END MODIFIED MERGING ---

                else:
                    logger.error(f"Could not find 'best_params' dictionary in results of step {prev_step_index}.")
                    self.all_results[run_id] = {"error": f"Missing 'best_params' in step {prev_step_index} results."}
                    break
            # --- End Parameter Injection Logic ---

            logger.debug(f"Running with effective parameters: {current_params}")
            start_time_method = time.time()

            try:
                pipeline_method = getattr(self.pipeline, method_name)
                # --- Pass run_id to save_results via the method if necessary ---
                # This requires modifying the signature of _save_results and how it's called
                # OR modifying methods to accept run_id if they call save_results internally
                # Let's modify _save_results to accept run_id instead.
                result = pipeline_method(**current_params)  # Call method
                # If the method didn't call _save_results itself (e.g., load_model)
                # we might want to save something here, but usually results are generated
                # by the methods that do computations.

                # Ensure the result is stored correctly
                self.all_results[run_id] = result
                method_duration = time.time() - start_time_method
                logger.info(f"--- Method {method_name} ({run_id}) completed successfully in {method_duration:.2f}s ---")

            except ValueError as ve:  # Catch specific config errors
                # Change exc_info to True here to log the stack trace for ValueError
                logger.error(f"!!! Configuration error in '{method_name}': {ve}", exc_info=True)
                logger.error(
                    f"!!! Check method compatibility with dataset structure (FIXED requires force_flat_for_fixed_cv=True for some methods) or parameters.")
                self.all_results[run_id] = {"error": str(ve)}
                break  # Stop execution on config errors
            except FileNotFoundError as fnf:
                # This already logs the stack trace
                logger.error(f"!!! File not found during '{method_name}': {fnf}", exc_info=True)
                self.all_results[run_id] = {"error": str(fnf)}
                break
            except RuntimeError as rte:  # Catch runtime errors (e.g., CUDA, data loading)
                # This already logs the stack trace
                logger.error(f"!!! Runtime error during '{method_name}': {rte}", exc_info=True)
                self.all_results[run_id] = {"error": str(rte)}
                break
            except Exception as e:  # Catch any other unexpected errors
                # This already logs the stack trace
                logger.critical(f"!!! An unexpected critical error occurred during '{method_name}': {e}", exc_info=True)
                self.all_results[run_id] = {"error": str(e), "traceback": logging.traceback.format_exc()}
                break  # Stop on critical errors

        total_duration = time.time() - start_time_total
        logger.info(f"Pipeline execution finished in {total_duration:.2f}s.")
        return self.all_results

# --- Example Usage ---
if __name__ == "__main__":
    script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    results_base_dir = script_dir / 'results'

    # --- Configuration ---
    # Select Dataset:
    dataset_path = script_dir / "../data/mini-GCD-flat" # FLAT example
    # dataset_path = script_dir / "../data/Swimcat-extend" # FIXED example
    # dataset_path = Path("PATH_TO_YOUR_DATASET") # Use your actual path

    if not Path(dataset_path).exists():
         logger.error(f"Dataset path not found: {dataset_path}")
         logger.error("Please create the dataset or modify the 'dataset_path' variable.")
         exit()

    # Select Model:
    model_type = "cnn"  # 'cnn', 'vit', 'diffusion'

    # Flag for CV methods on FIXED datasets:
    # Set to True to allow nested_grid_search and cv_model_evaluation on FIXED datasets
    # by treating train+test as one pool (USE WITH CAUTION - not standard evaluation).
    force_flat = False

    param_grid_cnn = {
        # Skorch parameters
        'lr': [0.005, 0.001, 0.0005],
        'batch_size': [16, 32],  # Note: Changing batch size can affect memory and convergence

        # Optimizer (AdamW) parameters
        'optimizer__weight_decay': [0.01, 0.001, 0.0001],
        # 'optimizer__betas': [(0.9, 0.999), (0.85, 0.99)], # Less common to tune

        # Module (SimpleCNN) parameters
        'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],  # Tune dropout in the classifier head

        # Maybe max_epochs if not using EarlyStopping effectively? Usually fixed or high w/ early stopping.
        # 'max_epochs': [15, 25],
    }

    param_grid_vit = {
        # Skorch parameters (especially LR for fine-tuning)
        'lr': [0.001, 0.0005, 0.0001, 0.00005],  # Often lower LRs for fine-tuning
        'batch_size': [16, 32],  # Memory constraints often tighter with ViT

        # Optimizer (AdamW) parameters
        'optimizer__weight_decay': [0.01, 0.001, 0.0],  # Weight decay is important

        # Module (SimpleViT) parameters
        # Since we only replaced the head and froze most layers, there are fewer
        # *direct* module hyperparameters to tune via __init__.
        # If you added dropout to the new head, you could tune 'module__dropout_rate'.
        # You *could* potentially tune which layers are frozen, but that's complex via grid search.

        # Training duration / EarlyStopping focus
        # 'max_epochs': [5, 10, 15], # If fine-tuning quickly
    }

    param_grid_diffusion = {
        # Skorch parameters
        'lr': [0.001, 0.0005, 0.0001],  # Fine-tuning learning rate
        'batch_size': [16, 32, 64],  # ResNet might be less memory-intensive than ViT

        # Optimizer (AdamW) parameters
        'optimizer__weight_decay': [0.01, 0.001, 0.0001],

        # Module (DiffusionClassifier) parameters
        'module__dropout_rate': [0.3, 0.4, 0.5, 0.6],  # Tune dropout in the custom head

        # Training duration
        # 'max_epochs': [10, 20, 30],
    }

    # --- Define Hyperparameter Grid / Fixed Params ---
    if model_type == 'cnn':
        chosen_param_grid = param_grid_cnn
    elif model_type == 'vit':
        chosen_param_grid = param_grid_vit
    elif model_type == 'diffusion':
        chosen_param_grid = param_grid_diffusion
    else:
        logger.error(f"Model type '{model_type}' not recognized. Supported: 'cnn', 'vit', 'diffusion'.")
        exit()

    # Temporarily set param grid
    chosen_param_grid = {
        # Skorch parameters
        'lr': [0.001],

        # Module (SimpleCNN) parameters
        'module__dropout_rate': [0.3, 0.6],  # Tune dropout in the classifier head
    }

    fixed_params_for_eval = {
        'lr': 0.001,
        'optimizer__weight_decay': 0.01,
        # 'module__dropout_rate': 0.4
    }


    # --- Define Method Sequence ---
    # Example 1: Single Train (using val split) and Eval
    methods_seq_1 = [
        ('single_train', {'max_epochs': 5, 'save_model': True, 'save_results': True, 'val_split_ratio': 0.2}), # Explicit val split
        ('single_eval', {'save_results': True}),
    ]
    # Example 2: Non-Nested Grid Search + Eval best model
    methods_seq_2 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid, 'cv': 3, 'method': 'grid',
            'scoring': 'accuracy', 'save_results': True
        }),
        # The best model is refit and stored in pipeline.model_adapter after search
        ('single_eval', {'save_results': True}), # Evaluate the refit best model
    ]
    # Example 3: Nested Grid Search (Requires FLAT or FIXED with force_flat=True)
    methods_seq_3 = [
         ('nested_grid_search', {
             'param_grid': chosen_param_grid, 'outer_cv': 3, 'inner_cv': 2,
             'method': 'grid', 'scoring': 'accuracy', 'save_results': True
         })
    ]
    # Example 4: Simple CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_4 = [
         ('cv_model_evaluation', {
             'cv': 3,
             'params': fixed_params_for_eval, # Pass fixed hyperparams
             'save_results': True
        })
    ]

    # Example 5: Non-Nested Grid Search + CV Evaluation (Requires FLAT or FIXED with force_flat=True)
    methods_seq_5 = [
        ('non_nested_grid_search', {
            'param_grid': chosen_param_grid,
            'cv': 3,
            'method': 'grid',
            'scoring': 'accuracy',
            'save_results': True
        }),
         ('cv_model_evaluation', {
             'cv': 3,
             # Special key indicates using best_params from previous step (index 0)
             'use_best_params_from_step': 0,
             # Optionally provide specific params for cv_eval to override defaults if needed
             # 'params': {'max_epochs': 15}, # e.g., override max_epochs just for CV eval
             'save_results': True
        })
    ]

    # Example 6: Load Pre-trained and Evaluate
    pretrained_model_path = "results/SOME_DATASET_cnn_TIMESTAMP/cnn_epochX_val....pt" # Replace with actual path
    methods_seq_6 = [
        ('load_model', {'model_path': pretrained_model_path}),
        ('single_eval', {'save_results': True}),
    ]


    # --- Choose Sequence and Execute ---
    chosen_sequence = methods_seq_1 # <--- SELECT SEQUENCE TO RUN

    logger.info(f"Executing sequence: {[m[0] for m in chosen_sequence]}")

    # --- Create and Run Executor ---
    try:
        executor = PipelineExecutor(
            dataset_path=dataset_path,
            model_type=model_type,
            results_dir=results_base_dir,
            methods=chosen_sequence,
            force_flat_for_fixed_cv=force_flat, # Pass the flag
            # Pipeline default parameters (can be overridden by methods)
            img_size=(64, 64), # Smaller size for faster demo
            batch_size=16,     # Smaller batch size for demo
            max_epochs=10,     # Fewer epochs for demo
            patience=3,        # Reduced patience for demo
            lr=0.001,
            optimizer__weight_decay=0.01,
            # module__dropout_rate=0.5 # If applicable to model
            save_detailed_results=True,  # Or False, depending on desired output level
        )
        final_results = executor.run()

        # Print final results summary
        logger.info("--- Final Execution Results Summary ---")
        for method_id, result_data in final_results.items():
            if isinstance(result_data, dict) and 'error' in result_data:
                 logger.error(f"Method {method_id}: FAILED - {result_data['error']}")
            elif isinstance(result_data, dict):
                 # Try to extract relevant metrics for summary log
                 acc = result_data.get('accuracy', result_data.get('mean_test_accuracy', np.nan))
                 f1 = result_data.get('macro_avg',{}).get('f1', result_data.get('mean_test_f1_macro', np.nan))
                 best_s = result_data.get('best_score', result_data.get('best_tuning_score', np.nan))
                 logger.info(f"Method {method_id}: Completed. "
                             f"(Acc: {acc:.4f}, F1: {f1:.4f}, BestScore: {best_s:.4f})")
            else:
                 logger.info(f"Method {method_id}: Completed. Result: {result_data}")


    except (FileNotFoundError, ValueError, RuntimeError) as e:
         logger.error(f"Pipeline initialization or execution failed: {e}", exc_info=True)
    except Exception as e:
         logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
