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
    roc_auc_score, precision_recall_curve, auc, make_scorer
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, train_test_split, PredefinedSplit
    # cross_val_score, cross_val_predict # Not used
)
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint, Callback
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
class EnhancedFormatter(logging.Formatter):
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
        if not hasattr(record, 'funcName'): record.funcName = '?'
        if not hasattr(record, 'lineno'): record.lineno = 0
        formatter = logging.Formatter(log_fmt, datefmt=self.date_format)
        return formatter.format(record)

def get_log_header(use_colors: bool = True) -> str:
    location_width = 32
    header_title = f"{'Timestamp':<21} | {'Level':<8} | {'Emoji':<2} | {'Location':<{location_width}} | {'Message'}"
    separator = f"{'-'*21}-+-{'-'*8}-+-{'-'*2}-+-{'-'*location_width}-+-{'-'*50}"
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

def setup_logger(name: str, log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO, use_colors: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.hasHandlers(): logger.handlers.clear()

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
        except Exception as e: logger.error(f"Failed to create file handler for {log_path}: {e}")
    return logger

# --- Setup Logger Instance ---
script_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd() # Handle interactive use
log_dir = script_dir / 'logs'
log_dir.mkdir(exist_ok=True)
logger_name = 'ImgClassPipe'
log_file_path = log_dir / 'classification.log'
logger = setup_logger(logger_name, log_file_path, level=logging.DEBUG, use_colors=True)
print(get_log_header(use_colors=True)) # Print header directly to console
logger.info(f"Logger '{logger_name}' initialized. Log file: {log_file_path}")

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
        logger.info(f"Dataset sizes: {len(self._train_val_paths)} train+val, {len(self._test_paths)} test.")
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

    def _scan_dir_for_paths_labels(self, target_dir: Path) -> Tuple[List[Path], List[int], List[str], Dict[str, int]]:
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
            policy=lr_policy, monitor=monitor, mode='min' if is_loss else 'max', patience=lr_patience, factor=0.1))
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
            f"Creating DataLoader: batch_size={batch_size}, shuffle={shuffle}, collate_fn={'Assigned' if collate_fn else 'None'}, other_kwargs={loader_kwargs}")

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
        Xi, yi = batch # Already transformed tensors from PathImageDataset
        yi = yi.to(dtype=torch.long)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        try:
            y_true_batch = yi.cpu().numpy()
            y_pred_batch = y_pred.argmax(dim=1).cpu().numpy()
            # Handle case where batch might be empty after collate filtering
            batch_acc = accuracy_score(y_true_batch, y_pred_batch) if len(y_true_batch) > 0 else 0.0
        except Exception as e:
            logger.warning(f"Acc calc failed: {e}")
            batch_acc = 0.0 # Default value on error

        # --- Return train_acc so skorch logs it ---
        return {'loss': loss, 'train_acc': batch_acc, 'y_pred': y_pred}

    def validation_step(self, batch, **fit_params):
        self.module_.eval()
        Xi, yi = batch # Already transformed tensors from PathImageDataset
        yi = yi.to(dtype=torch.long)
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        # Return y_pred so skorch can calculate valid_acc etc.
        return {'loss': loss, 'y_pred': y_pred}

# --- REMOVE SetTransformMode callback class ---

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
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 data_augmentation: bool = True,
                 force_flat_for_fixed_cv: bool = False, # New flag
                 # Skorch adapter params
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10, # For default EarlyStopping
                 optimizer__weight_decay: float = 0.01, # Example tunable param
                 module__dropout_rate: Optional[float] = None # Example model param
                 ):
        self.dataset_path = Path(dataset_path).resolve()
        self.model_type = model_type.lower()
        self.force_flat_for_fixed_cv = force_flat_for_fixed_cv
        logger.info(f"Initializing Classification Pipeline:")
        logger.info(f"  Dataset Path: {self.dataset_path}")
        logger.info(f"  Model Type: {self.model_type}")
        logger.info(f"  Force Flat for Fixed CV: {self.force_flat_for_fixed_cv}")

        # Initialize dataset handler (gets paths, labels, transforms)
        self.dataset_handler = ImageDatasetHandler(
            root_path=self.dataset_path,
            img_size=img_size,
            val_split_ratio=val_split_ratio,
            test_split_ratio_if_flat=test_split_ratio_if_flat,
            data_augmentation=data_augmentation,
            force_flat_for_fixed_cv=self.force_flat_for_fixed_cv
        )

        # Results directory
        base_results_dir = Path(results_dir).resolve()
        dataset_name = self.dataset_path.name
        timestamp_init = datetime.now().strftime('%Y%m%d_%H%M%S') # Add timestamp to avoid conflicts
        self.results_dir = base_results_dir / f"{dataset_name}_{self.model_type}_{timestamp_init}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Results will be saved to: {self.results_dir}")

        # Select model class
        model_class = self._get_model_class(self.model_type)

        # --- Prepare Skorch Adapter configuration ---
        # Store the INTENDED callback setting and config params
        module_params = {}
        if module__dropout_rate is not None: module_params['module__dropout_rate'] = module__dropout_rate

        # --- Determine initial callbacks list ---
        intended_callbacks_setting = 'default'  # Or could be passed as argument
        initial_callbacks_list = None
        if intended_callbacks_setting == 'default':
            initial_callbacks_list = _get_default_callbacks(
                patience=patience, monitor='valid_loss',  # Use init args for defaults
                lr_policy='ReduceLROnPlateau', lr_patience=5
            )
        elif isinstance(intended_callbacks_setting, list):
            initial_callbacks_list = intended_callbacks_setting
        # --- End Determine ---

        self.model_adapter_config = {
            'module': model_class,
            'module__num_classes': self.dataset_handler.num_classes,
            'criterion': nn.CrossEntropyLoss,
            'optimizer': torch.optim.AdamW,
            'lr': lr,
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'device': DEVICE,
            # --- Store the ACTUAL list/None ---
            'callbacks': initial_callbacks_list,
            # --- Store config params separately IF NEEDED for modification later ---
            'patience_cfg': patience,  # Use different keys to avoid clashes if needed
            'monitor_cfg': 'valid_loss',
            'lr_policy_cfg': 'ReduceLROnPlateau',
            'lr_patience_cfg': 5,
            # --- End Store Config ---
            'train_transform': self.dataset_handler.get_train_transform(),
            'valid_transform': self.dataset_handler.get_eval_transform(),
            'classes': np.arange(self.dataset_handler.num_classes),
            'verbose': 3,  # Set default verbose level here
            'optimizer__weight_decay': optimizer__weight_decay,
            **module_params
        }

        # --- Instantiate the INITIAL adapter ---
        # The config now already contains the correct callback list/None
        init_config_for_adapter = self.model_adapter_config.copy()
        # Remove config params not needed by SkorchModelAdapter init directly
        init_config_for_adapter.pop('patience_cfg', None)
        init_config_for_adapter.pop('monitor_cfg', None)
        init_config_for_adapter.pop('lr_policy_cfg', None)
        init_config_for_adapter.pop('lr_patience_cfg', None)

        # Initialize the adapter instance (can be cloned later)
        self.model_adapter = SkorchModelAdapter(**self.model_adapter_config)
        logger.info(f"  Model Adapter: Initialized with {model_class.__name__}")

        if model_load_path:
            self.load_model(model_load_path)

        logger.info(f"Pipeline initialized successfully.")

    @staticmethod
    def _get_model_class(model_type_str: str) -> Type[nn.Module]:
        model_mapping = {'cnn': SimpleCNN, 'vit': SimpleViT, 'diffusion': DiffusionClassifier}
        model_class = model_mapping.get(model_type_str.lower())
        if model_class is None:
            raise ValueError(f"Unsupported model type: '{model_type_str}'. Choose from {list(model_mapping.keys())}.")
        return model_class

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
        # (Keep logic from code_v6, it works on numpy arrays)
        if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
        if not isinstance(y_pred, np.ndarray): y_pred = np.array(y_pred)
        if y_score is not None and not isinstance(y_score, np.ndarray): y_score = np.array(y_score)

        metrics: Dict[str, Any] = {'accuracy': accuracy_score(y_true, y_pred)}
        present_class_labels = np.unique(y_true)
        all_class_names = self.dataset_handler.classes
        if not all_class_names:
             logger.warning("Dataset handler classes list empty. Cannot compute per-class/macro metrics.")
             metrics['macro_avg'] = {'precision': np.nan, 'recall': np.nan, 'specificity': np.nan, 'f1': np.nan, 'roc_auc': np.nan, 'pr_auc': np.nan}
             return metrics

        num_classes_total = self.dataset_handler.num_classes
        all_precisions, all_recalls, all_specificities, all_f1s = [], [], [], []
        all_roc_aucs, all_pr_aucs = [], []
        can_compute_auc = y_score is not None and len(y_score.shape) == 2 and y_score.shape[1] == num_classes_total and len(y_score) == len(y_true)

        if y_score is not None and not can_compute_auc:
             logger.warning(f"y_score shape {y_score.shape if y_score is not None else 'None'} incompatible with y_true len {len(y_true)} and num_classes {num_classes_total}. Cannot compute AUCs.")

        for i, class_name in enumerate(all_class_names):
            class_label = self.dataset_handler.class_to_idx.get(class_name, i)
            if class_label not in present_class_labels:
                 # Append NaN for missing classes in this subset
                 all_precisions.append(np.nan); all_recalls.append(np.nan); all_specificities.append(np.nan); all_f1s.append(np.nan); all_roc_aucs.append(np.nan); all_pr_aucs.append(np.nan)
                 continue

            true_is_class = (y_true == class_label)
            pred_is_class = (y_pred == class_label)

            precision = precision_score(true_is_class, pred_is_class, zero_division=0)
            recall = recall_score(true_is_class, pred_is_class, zero_division=0) # Sensitivity
            f1 = f1_score(true_is_class, pred_is_class, zero_division=0)
            # Specificity = TN / (TN + FP) = Recall of negative class
            specificity = recall_score(~true_is_class, ~pred_is_class, zero_division=0)

            all_precisions.append(precision); all_recalls.append(recall); all_specificities.append(specificity); all_f1s.append(f1)

            roc_auc, pr_auc = np.nan, np.nan
            if can_compute_auc:
                try:
                    score_for_class = y_score[:, class_label]
                    if len(np.unique(true_is_class)) > 1: # AUC requires both classes present
                        try: roc_auc = roc_auc_score(true_is_class, score_for_class)
                        except Exception as e: logger.warning(f"ROC AUC Error (Class {class_name}): {e}")
                        try:
                            prec, rec, _ = precision_recall_curve(true_is_class, score_for_class)
                            # Ensure recall is sorted for AUC calculation
                            order = np.argsort(rec)
                            pr_auc = auc(rec[order], prec[order])
                        except Exception as e: logger.warning(f"PR AUC Error (Class {class_name}): {e}")
                    # else: logger.debug(f"Skipping AUC for class {class_name}: only one class present.") # Too verbose
                except IndexError:
                    logger.warning(f"IndexError getting y_score column {class_label}. Skipping AUCs for class {class_name}.")
                except Exception as e_outer:
                     logger.warning(f"Error calculating AUCs for class {class_name}: {e_outer}")


            all_roc_aucs.append(roc_auc)
            all_pr_aucs.append(pr_auc)

        metrics['macro_avg'] = {
            'precision': float(np.nanmean(all_precisions)), 'recall': float(np.nanmean(all_recalls)),
            'specificity': float(np.nanmean(all_specificities)), 'f1': float(np.nanmean(all_f1s)),
            'roc_auc': float(np.nanmean(all_roc_aucs)) if can_compute_auc else np.nan,
            'pr_auc': float(np.nanmean(all_pr_aucs)) if can_compute_auc else np.nan
        }
        logger.debug(f"Computed Metrics: Acc={metrics['accuracy']:.4f}, Macro F1={metrics['macro_avg']['f1']:.4f}")
        return metrics

    def _save_results(self, results_data: Dict[str, Any], method_name: str, params: Optional[Dict[str, Any]] = None) -> None:
        params = params or {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- Define Keys for Filename Parameters ---
        # Select only simple, key hyperparameters relevant for filenames
        filename_param_keys = [
            'lr', 'batch_size', 'max_epochs', 'patience', # Common Skorch params
            'optimizer__weight_decay', # Example optimizer param
            'module__dropout_rate', # Example module param
            # CV specific params (if applicable)
            'cv', 'outer_cv', 'inner_cv', 'n_iter'
        ]

        # --- Filter and Sanitize Filename Parameters ---
        filename_params = {}
        for key in filename_param_keys:
            if key in params:
                value = params[key]
                # Check if value is a simple type suitable for filename
                if isinstance(value, (str, int, float, bool, type(None))):
                     # Sanitize the value
                     s_val = str(value).replace('/', '_').replace('\\', '_')
                     s_val = re.sub(r'[<>:"|?*]', '_', s_val) # More restricted set
                     filename_params[key] = s_val[:30] # Limit length per value too
                # else: skip complex types for filename

        params_str = '_'.join([f"{k}={v}" for k, v in sorted(filename_params.items())])

        filename_base = f"{method_name}_{params_str}_{timestamp}" if params_str else f"{method_name}_{timestamp}"
        json_filepath = self.results_dir / f"{filename_base}.json"
        # Ensure filename isn't exceeding limits (simple check)
        max_len = 240 # Leave some room for directory path
        if len(json_filepath.name) > max_len:
             logger.warning(f"Generated filename exceeds {max_len} chars, truncating: {json_filepath.name}")
             # Truncate the params part intelligently if possible, or just hash it
             hash_part = hashlib.md5(params_str.encode()).hexdigest()[:8]
             filename_base = f"{method_name}_params_{hash_part}_{timestamp}"
             json_filepath = self.results_dir / f"{filename_base}.json"
             logger.warning(f"Using truncated filename: {json_filepath.name}")


        csv_filepath = self.results_dir.parent / f"{self.dataset_handler.root_path.name}_summary_results.csv" # Summary in parent dir

        # --- Save detailed JSON ---
        try:
            def json_serializer(obj):
                 # (Keep existing serializer logic)
                 if isinstance(obj, (np.integer, np.int64)): return int(obj)
                 elif isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj) if not np.isnan(obj) else None
                 elif isinstance(obj, np.ndarray): return obj.tolist()
                 elif isinstance(obj, Path): return str(obj)
                 elif isinstance(obj, datetime): return obj.isoformat()
                 elif isinstance(obj, (slice, type, Callable)): return None
                 # --- ADDED: Handle specific non-serializable skorch/torch items in config ---
                 elif isinstance(obj, (torch.optim.Optimizer, nn.Module, EarlyStopping, LRScheduler, ValidSplit)):
                      return str(type(obj).__name__) # Just store the type name
                 # --- END ADDED ---
                 try: return json.JSONEncoder.default(None, obj)
                 except TypeError: return str(obj)


            # Clean results *before* saving
            clean_results = results_data.copy()
            # Save the *original* full params dict inside the JSON (after serialization)
            clean_results['full_params_used'] = params

            # Clean cv_results if present
            if 'cv_results' in clean_results and isinstance(clean_results['cv_results'], dict):
                 clean_results['cv_results'].pop('params', None)
                 clean_results['cv_results'].pop('estimator', None)
                 for key, value in clean_results['cv_results'].items():
                     if isinstance(value, np.ndarray):
                          clean_results['cv_results'][key] = value.tolist()
            if 'fold_histories' in clean_results: del clean_results['fold_histories']

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, indent=4, default=json_serializer)
            logger.info(f"Detailed results saved to: {json_filepath}")
        except OSError as oe: # Catch specific OS errors like filename too long
             logger.error(f"OS Error saving detailed results to JSON {json_filepath}: {oe}", exc_info=True)
             # Maybe try saving with a simpler filename as fallback?
             try:
                  fallback_path = self.results_dir / f"{method_name}_fallback_{timestamp}.json"
                  with open(fallback_path, 'w', encoding='utf-8') as f:
                      json.dump(clean_results, f, indent=4, default=json_serializer)
                  logger.warning(f"Saved results to fallback file: {fallback_path}")
             except Exception as fallback_e:
                  logger.error(f"Fallback JSON save also failed: {fallback_e}")
        except Exception as e:
            logger.error(f"Failed to save detailed results to JSON {json_filepath}: {e}", exc_info=True)

        # --- Prepare and save summary CSV ---
        try:
            # (Keep existing summary logic)
            macro_avg = results_data.get('macro_avg', {})
            test_eval = results_data.get('test_set_evaluation', results_data.get('fixed_test_set_evaluation', {}))

            summary = {
                'method': method_name,
                'timestamp': timestamp,
                'model_type': self.model_type,
                'dataset_name': self.dataset_handler.root_path.name,
                'dataset_structure': self.dataset_handler.structure.value,
                'forced_flat': self.force_flat_for_fixed_cv if self.dataset_handler.structure == DatasetStructure.FIXED else 'N/A',
                'accuracy': results_data.get('accuracy', results_data.get('mean_test_accuracy', test_eval.get('accuracy', np.nan))),
                'macro_f1': macro_avg.get('f1', results_data.get('mean_test_f1_macro', test_eval.get('macro_avg', {}).get('f1', np.nan))),
                'macro_roc_auc': macro_avg.get('roc_auc', results_data.get('mean_test_roc_auc_macro', test_eval.get('macro_avg', {}).get('roc_auc', np.nan))),
                'macro_pr_auc': macro_avg.get('pr_auc', results_data.get('mean_test_pr_auc_macro', test_eval.get('macro_avg', {}).get('pr_auc', np.nan))),
                'best_cv_score': results_data.get('best_score', results_data.get('best_tuning_score', np.nan)),
                **filename_params # Add simple FILENAME params to summary
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

        # --- Setup Skorch Estimator for Search ---
        adapter_config = self.model_adapter_config.copy()
        internal_val_fraction = 0.15  # Define fraction
        adapter_config['train_split'] = ValidSplit(cv=internal_val_fraction, stratified=True,
                                                   random_state=RANDOM_SEED)  # Use internal validation split
        logger.info(
            f"Skorch internal validation split configured: {internal_val_fraction * 100:.1f}% of each CV fold's training data.")
        adapter_config['verbose'] = 3  # Less verbose fits during search

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
            self._save_results(results, f"non_nested_{method_lower}_search", params=filename_params)

        logger.info(f"Non-nested {method_lower} search finished. Best CV score ({scoring}): {search.best_score_:.4f}")
        logger.info(f"Best parameters found: {search.best_params_}")
        logger.info(f"Pipeline adapter has been updated with the best model refit on train+validation data.")
        return results


    def nested_grid_search(self,
                           param_grid: Dict[str, List],
                           outer_cv: int = 5,
                           inner_cv: int = 3,
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

        # --- Setup Inner Search Object ---
        adapter_config = self.model_adapter_config.copy()
        # Use ValidSplit for internal validation during each inner GridSearch fit
        internal_val_fraction = 0.15  # Define fraction
        adapter_config['train_split'] = ValidSplit(cv=internal_val_fraction, stratified=True, random_state=RANDOM_SEED)
        logger.info(
            f"Inner loop Skorch validation split configured: {internal_val_fraction * 100:.1f}% of inner CV fold's training data.")
        adapter_config['verbose'] = 3 # More verbose inner loops

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
                 self._save_results(results, f"nested_{method_lower}_search", params=results['params'])

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

    def cv_model_evaluation(self, cv: int = 5, params: Optional[Dict] = None, save_results: bool = True) -> Dict[str, Any]:
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


        # --- Setup CV Strategy ---
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        fold_results = []
        fold_histories = [] # Store history from each fold

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
            internal_val_fraction = 0.15  # Define the fraction used
            fold_adapter_config['train_split'] = ValidSplit(cv=internal_val_fraction, stratified=True, random_state=RANDOM_SEED)
            fold_adapter_config['verbose'] = 3 # Show progress per fold
            fold_adapter_config['callbacks'] = [ # Ensure callbacks are set for this fold
                ('early_stopping', EarlyStopping(monitor='valid_loss', patience=eval_params.get('patience', 10), load_best=True)),
                ('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', monitor='valid_loss', patience=5))
             ]

            n_outer_train = len(X_outer_train)
            n_inner_val = int(n_outer_train * internal_val_fraction)
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
                 # Use score method for accuracy
                 fold_acc = estimator_fold.score(X_test, y=y_test)
                 # Predict for other metrics
                 y_pred_test = estimator_fold.predict(X_test)
                 fold_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
                 # y_score_test = estimator_fold.predict_proba(X_test) # For AUC if needed

                 fold_results.append({'accuracy': fold_acc, 'f1_macro': fold_f1})
                 logger.info(f"Fold {fold_idx + 1} Test Scores: Acc={fold_acc:.4f}, F1={fold_f1:.4f}")
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
             'params': eval_params, # Save the actual parameters used
             'cv_scores': df_results.to_dict(orient='list'),
             'mean_test_accuracy': float(df_results['accuracy'].mean()),
             'std_test_accuracy': float(df_results['accuracy'].std()),
             'mean_test_f1_macro': float(df_results['f1_macro'].mean()),
             'std_test_f1_macro': float(df_results['f1_macro'].std()),
             # 'fold_histories': fold_histories # Optional: Can be very large
        }
        results['accuracy'] = results['mean_test_accuracy'] # For summary
        results['macro_avg'] = {'f1': results['mean_test_f1_macro']} # For summary

        if save_results:
            self._save_results(results, "cv_model_evaluation", params=eval_params)

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
             adapter_config['callbacks'] = [] # Or filter existing list
        else:
             # Ensure callbacks monitor validation metrics
             adapter_config['callbacks'] = [
                 ('early_stopping', EarlyStopping(monitor='valid_loss', patience=adapter_config.get('patience', 10), load_best=True)),
                 ('lr_scheduler', LRScheduler(policy='ReduceLROnPlateau', monitor='valid_loss', patience=5))
             ]
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

        if save_results:
            self._save_results(results, "single_train", params=adapter_config)

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

        # --- Compute Metrics ---
        metrics = self._compute_metrics(y_test_np, y_pred_test, y_score_test)
        results = {'method': 'single_eval', 'params': {}, **metrics}

        if save_results:
             self._save_results(results, "single_eval") # Use metrics dict directly

        logger.info(f"Evaluation Summary:")
        logger.info(f"  Accuracy: {metrics.get('accuracy', np.nan):.4f}")
        logger.info(f"  Macro F1: {metrics.get('macro_avg', {}).get('f1', np.nan):.4f}")
        logger.info(f"  Macro ROC AUC: {metrics.get('macro_avg', {}).get('roc_auc', np.nan):.4f}")
        logger.info(f"  Macro PR AUC: {metrics.get('macro_avg', {}).get('pr_auc', np.nan):.4f}")

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
                 methods: List[Tuple[str, Dict[str, Any]]]= None,
                 # Pipeline config params passed down
                 img_size: Tuple[int, int] = (224, 224),
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 data_augmentation: bool = True,
                 force_flat_for_fixed_cv: bool = False, # New flag
                 lr: float = 0.001,
                 max_epochs: int = 20,
                 batch_size: int = 32,
                 patience: int = 10,
                 optimizer__weight_decay: float = 0.01,
                 module__dropout_rate: Optional[float] = None
                 ):
        logger.info(f"Initializing Pipeline Executor for model '{model_type}' on dataset '{Path(dataset_path).name}'...")

        self.pipeline = ClassificationPipeline(
            dataset_path=dataset_path, model_type=model_type, model_load_path=model_load_path,
            results_dir=results_dir, img_size=img_size, val_split_ratio=val_split_ratio,
            test_split_ratio_if_flat=test_split_ratio_if_flat, data_augmentation=data_augmentation,
            force_flat_for_fixed_cv=force_flat_for_fixed_cv, lr=lr, max_epochs=max_epochs,
            batch_size=batch_size, patience=patience,
            optimizer__weight_decay=optimizer__weight_decay, module__dropout_rate=module__dropout_rate
        )

        self.methods_to_run = methods if methods is not None else []
        self.all_results: Dict[str, Any] = {}
        self._validate_methods() # Basic validation
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

    def run(self) -> Dict[str, Any]:
        """ Executes the configured sequence of pipeline methods. """
        self.all_results = {}
        logger.info("Starting execution of pipeline methods...")
        start_time_total = time.time()

        for i, (method_name, params) in enumerate(self.methods_to_run):
            run_id = f"{method_name}_{i}"
            logger.info(f"--- Running Method {i+1}/{len(self.methods_to_run)}: {method_name} ---")
            logger.debug(f"Parameters: {params}")
            start_time_method = time.time()

            try:
                pipeline_method = getattr(self.pipeline, method_name)
                result = pipeline_method(**params)
                self.all_results[run_id] = result
                method_duration = time.time() - start_time_method
                logger.info(f"--- Method {method_name} completed successfully in {method_duration:.2f}s ---")

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

    # --- Define Hyperparameter Grid / Fixed Params ---
    param_grid_search = {
        'lr': [0.001],
        'optimizer__weight_decay': [0.01, 0.005],
        # 'module__dropout_rate': [0.3, 0.5] # Example if model has dropout_rate
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
            'param_grid': param_grid_search, 'cv': 3, 'method': 'grid',
            'scoring': 'accuracy', 'save_results': True
        }),
        # The best model is refit and stored in pipeline.model_adapter after search
        ('single_eval', {'save_results': True}), # Evaluate the refit best model
    ]
    # Example 3: Nested Grid Search (Requires FLAT or FIXED with force_flat=True)
    methods_seq_3 = [
         ('nested_grid_search', {
             'param_grid': param_grid_search, 'outer_cv': 3, 'inner_cv': 2,
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
    # Example 5: Load Pre-trained and Evaluate
    # pretrained_model_path = "results/SOME_DATASET_cnn_TIMESTAMP/cnn_epochX_val....pt" # Replace with actual path
    # methods_seq_5 = [
    #     ('load_model', {'model_path': pretrained_model_path}),
    #     ('single_eval', {'save_results': True}),
    # ]


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
