import os
import sys
import logging
import random
import json
import time
import datetime
from typing import List, Tuple, Dict, Optional, Any, Generator, Union
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, WeightedRandomSampler
from torchvision import transforms # Use torchvision transforms for convenience
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 1. Logger Setup ---
def setup_logger(log_file_path: str = 'old_pipeline.log') -> logging.Logger:
    """Sets up a logger that outputs to console and a file."""
    logger = logging.getLogger('ClassificationPipelineLogger')
    logger.setLevel(logging.DEBUG) # Capture all levels of logs

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO) # Show INFO level and above on console
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File Handler
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG) # Log everything to the file
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
                                           datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# --- Emojis for Logging ---
LOG_EMOJIS = {
    "start": "🚀",
    "data": "💾",
    "split": "🔀",
    "model": "🤖",
    "train": "🚂",
    "val": " V ", # Using V as validation emoji proxy
    "test": "🧪",
    "hyper": "⚙️",
    "fold": "📁",
    "result": "📊",
    "save": "📄",
    "stop": "🛑",
    "best": "⭐",
    "error": "❌",
    "info": "ℹ️",
    "done": "✅"
}

# Global logger instance (initialize once)
logger = setup_logger()


# --- 2. Custom Dataset ---
class CustomImageDataset(Dataset):
    """A custom dataset to load images from paths."""
    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[Any] = None):
        """
        Args:
            image_paths (List[str]): List of full paths to images.
            labels (List[int]): List of corresponding integer labels.
            transform (Optional[Any]): PyTorch transforms to apply to images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        if len(image_paths) != len(labels):
             raise ValueError("image_paths and labels must have the same length.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Open image using PIL and convert to RGB (safer)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"{LOG_EMOJIS['error']} Error loading image {img_path}: {e}")
            # Let's raise error, as it indicates data issue
            raise IOError(f"Could not load image: {img_path}") from e

        if self.transform:
            image = self.transform(image)

        return image, label


# --- 3. Dataset Handler ---
class DatasetHandler:
    """Handles dataset loading, structure detection, and splitting."""

    STRUCTURE_FLAT = "flat"
    STRUCTURE_TRAIN_TEST = "train_test"

    def __init__(self, root_dir: str, transform: Any):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            transform (Any): Transformations to apply to images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.structure_type: Optional[str] = None
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

        # Store paths and labels for different splits
        self.all_image_paths: List[str] = []
        self.all_labels: List[int] = []
        self.train_image_paths: List[str] = []
        self.train_labels: List[int] = []
        self.test_image_paths: List[str] = [] # Only used if structure is train_test
        self.test_labels: List[int] = []      # Only used if structure is train_test

        self._detect_structure_and_load()

    def _detect_structure_and_load(self):
        """Detects dataset structure and loads image paths and labels."""
        logger.info(f"{LOG_EMOJIS['data']} Detecting dataset structure in: {self.root_dir}")
        has_train = os.path.isdir(os.path.join(self.root_dir, 'train'))
        has_test = os.path.isdir(os.path.join(self.root_dir, 'test'))

        if has_train and has_test:
            self.structure_type = self.STRUCTURE_TRAIN_TEST
            logger.info(f"{LOG_EMOJIS['data']} Detected 'train'/'test' structure.")
            train_dir = os.path.join(self.root_dir, 'train')
            test_dir = os.path.join(self.root_dir, 'test')

            self.train_image_paths, self.train_labels, train_classes = self._scan_folder(train_dir)
            self.test_image_paths, self.test_labels, test_classes = self._scan_folder(test_dir)

            if set(train_classes) != set(test_classes):
                 logger.warning(f"{LOG_EMOJIS['error']} Train and Test folders have different class names!")
                 # TODO: Decide how to handle: error out, use union, use train classes? Using train for now.
            self.class_names = sorted(list(train_classes))

            # Combine train paths/labels into 'all' for potential full dataset operations
            # if needed, but primarily keep them separate.
            self.all_image_paths = self.train_image_paths # In this structure, 'all' non-test data is train data
            self.all_labels = self.train_labels

        else:
            # Assume flat structure: root_dir contains class folders
            self.structure_type = self.STRUCTURE_FLAT
            logger.info(f"{LOG_EMOJIS['data']} Detected 'flat' structure (class folders in root).")
            self.all_image_paths, self.all_labels, self.class_names = self._scan_folder(self.root_dir)
            self.class_names = sorted(self.class_names)
            # In this structure, train/test paths are initially empty
            self.train_image_paths = self.all_image_paths # TODO: Should be considered as 'train' data or all data?
            self.train_labels = self.all_labels


        if not self.class_names:
             raise ValueError(f"No classes found in {self.root_dir}. Check directory structure.")

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}

        # Update labels to be integer indices based on final class_to_idx
        self.all_labels = [self.class_to_idx[self.idx_to_class[lbl]] for lbl in self.all_labels] # Re-map based on final sorted class_names
        self.train_labels = [self.class_to_idx[self.idx_to_class[lbl]] for lbl in self.train_labels]
        if self.structure_type == self.STRUCTURE_TRAIN_TEST:
            self.test_labels = [self.class_to_idx[self.idx_to_class[lbl]] for lbl in self.test_labels]

        logger.info(f"{LOG_EMOJIS['data']} Found {self.num_classes} classes: {', '.join(self.class_names)}")
        logger.info(f"{LOG_EMOJIS['data']} Total images found: {len(self.all_image_paths) + len(self.test_image_paths)}")
        if self.structure_type == self.STRUCTURE_TRAIN_TEST:
             logger.info(f"{LOG_EMOJIS['data']} Train images: {len(self.train_image_paths)}, Test images: {len(self.test_image_paths)}")


    @staticmethod
    def _scan_folder(folder_path: str) -> Tuple[List[str], List[int], List[str]]:
        """Scans a folder containing class subdirectories."""
        image_paths = []
        labels = []
        class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
        temp_class_to_idx = {name: i for i, name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(folder_path, class_name)
            if not os.path.isdir(class_dir): continue # Skip files # TODO: is this line necessary? (as we check above)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    image_paths.append(os.path.join(class_dir, file_name))
                    labels.append(temp_class_to_idx[class_name]) # Use temporary index

        return image_paths, labels, class_names

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def has_fixed_test_set(self) -> bool:
        return self.structure_type == self.STRUCTURE_TRAIN_TEST

    def get_fixed_test_dataset(self) -> Optional[CustomImageDataset]:
        """Returns the dataset for the fixed test set, if available."""
        if self.has_fixed_test_set:
            return CustomImageDataset(self.test_image_paths, self.test_labels, self.transform)
        return None

    def get_full_dataset(self, use_train_only: bool = False) -> CustomImageDataset:
        """
        Returns a dataset object representing either all data (if flat)
        or only the training data (if train_test structure).
        """
        if use_train_only and self.has_fixed_test_set:
             # Return dataset based only on the 'train' folder data
             return CustomImageDataset(self.train_image_paths, self.train_labels, self.transform)
        else:
             # Return dataset based on 'all' loaded data (which is train data if train_test, or everything if flat)
             return CustomImageDataset(self.all_image_paths, self.all_labels, self.transform)

    # --- Splitting Methods ---

    def get_train_val_test_split(self,
                                 test_size: float = 0.2,
                                 val_size: float = 0.2, # Proportion of the *original* dataset
                                 random_state: Optional[int] = None
                                 ) -> Tuple[Subset, Subset, Optional[Subset]]:
        """
        Performs a single stratified train-val-test split.

        - If has_fixed_test_set is True, it splits only the 'train' data into train/val,
          and the 'test' split returned is None (use get_fixed_test_dataset).
        - If has_fixed_test_set is False, it splits all data into train/val/test.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the dataset to include in the validation split.
            random_state (int, optional): Random seed for reproducibility.

        Returns:
            Tuple[Subset, Subset, Optional[Subset]]: Train, validation, and test subsets.
        """
        if self.has_fixed_test_set:
            logger.info(f"{LOG_EMOJIS['split']} Performing train/val split on the predefined 'train' data.")
            base_dataset = self.get_full_dataset(use_train_only=True)
            indices = list(range(len(base_dataset)))
            labels = base_dataset.labels # Use labels directly from the dataset

            # Split train data into train and validation
            # Adjust val_size relative to the training set size
            relative_val_size = val_size / (1.0 - test_size) # Incorrect assumption here - val_size should be proportion of TRAIN set
            # Let's redefine val_size = 0.2 to mean 20% of the *training* data goes to validation
            if len(indices) == 0:
                 raise ValueError("No training data available for splitting.")
            if len(set(labels)) < 2 : # Need at least 2 classes for stratification
                logger.warning(f"{LOG_EMOJIS['split']} Not enough classes in training data for stratification. Performing random split.")
                train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=random_state)
            else:
                try:
                    train_idx, val_idx = train_test_split(
                        indices, test_size=val_size, random_state=random_state, stratify=labels
                    )
                except ValueError as e:
                    logger.warning(f"{LOG_EMOJIS['split']} Stratification failed (perhaps too few samples per class?): {e}. Falling back to random split.")
                    train_idx, val_idx = train_test_split(indices, test_size=val_size, random_state=random_state)


            train_subset = Subset(base_dataset, train_idx)
            val_subset = Subset(base_dataset, val_idx)
            logger.info(f"{LOG_EMOJIS['split']} Split sizes: Train={len(train_subset)}, Val={len(val_subset)}. Test set is fixed.")
            return train_subset, val_subset, None # No test subset derived from this split

        else:
            # Split all data into train, validation, and test
            logger.info(f"{LOG_EMOJIS['split']} Performing train/val/test split on the full dataset.")
            base_dataset = self.get_full_dataset()
            indices = list(range(len(base_dataset)))
            labels = base_dataset.labels

            if len(indices) == 0:
                 raise ValueError("No data available for splitting.")
            if len(set(labels)) < 2:
                logger.warning(f"{LOG_EMOJIS['split']} Not enough classes for stratification. Performing random split.")
                # First split into train+val and test
                train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
                # Then split train+val into train and val
                relative_val_size = val_size / (1.0 - test_size)
                train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val_size, random_state=random_state)

            else:
                 try:
                     # First split into train+val and test
                     train_val_idx, test_idx = train_test_split(
                         indices, test_size=test_size, random_state=random_state, stratify=labels
                     )
                     # Need labels corresponding to train_val_idx for the second split
                     train_val_labels = [labels[i] for i in train_val_idx]

                     if len(set(train_val_labels)) < 2:
                         logger.warning(f"{LOG_EMOJIS['split']} Not enough classes in train+val for stratification. Performing random split for train/val.")
                         relative_val_size = val_size / (1.0 - test_size)
                         train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val_size, random_state=random_state)
                     else:
                        # Calculate relative validation size
                        relative_val_size = val_size / (1.0 - test_size)
                        if relative_val_size >= 1.0:
                             raise ValueError(f"test_size ({test_size}) + val_size ({val_size}) must be less than 1.0")
                        try:
                            train_idx, val_idx = train_test_split(
                                train_val_idx, test_size=relative_val_size, random_state=random_state, stratify=train_val_labels
                            )
                        except ValueError as e_inner:
                            logger.warning(f"{LOG_EMOJIS['split']} Inner stratification failed: {e_inner}. Falling back to random split for train/val.")
                            train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val_size, random_state=random_state)


                 except ValueError as e_outer:
                    logger.warning(f"{LOG_EMOJIS['split']} Outer stratification failed: {e_outer}. Falling back to random split for train+val/test.")
                    # First split into train+val and test (random)
                    train_val_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
                    # Then split train+val into train and val (random)
                    relative_val_size = val_size / (1.0 - test_size)
                    train_idx, val_idx = train_test_split(train_val_idx, test_size=relative_val_size, random_state=random_state)

            train_subset = Subset(base_dataset, train_idx)
            val_subset = Subset(base_dataset, val_idx)
            test_subset = Subset(base_dataset, test_idx)
            logger.info(f"{LOG_EMOJIS['split']} Split sizes: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
            return train_subset, val_subset, test_subset


    def get_cv_train_val_folds(self, n_splits: int = 5, random_state: Optional[int] = None
                               ) -> Generator[Tuple[Subset, Subset], None, None]:
        """
        Yields (train_subset, val_subset) for K-Fold CV on the 'training' data.
        'Training' data is all data if flat structure, or data in 'train' folder if train_test structure.
        Used for hyperparameter tuning (Step 1).
        """
        if self.has_fixed_test_set:
            logger.info(f"{LOG_EMOJIS['fold']} Generating {n_splits} train/val CV folds from the predefined 'train' data.")
            base_dataset = self.get_full_dataset(use_train_only=True)
            indices = np.arange(len(base_dataset))
            labels = np.array(base_dataset.labels)
        else:
            logger.info(f"{LOG_EMOJIS['fold']} Generating {n_splits} train/val CV folds from the full dataset.")
            base_dataset = self.get_full_dataset()
            indices = np.arange(len(base_dataset))
            labels = np.array(base_dataset.labels)

        if len(indices) == 0:
            raise ValueError("No data available for K-Fold splitting.")
        if len(set(labels)) < n_splits:
             logger.warning(f"{LOG_EMOJIS['split']} Number of classes ({len(set(labels))}) is less than n_splits ({n_splits}). Stratification might behave unexpectedly or fail if classes have < n_splits samples.")
        # Check if any class has fewer samples than n_splits
        unique_labels, counts = np.unique(labels, return_counts=True)
        if any(count < n_splits for count in counts) and len(set(labels)) > 1 : # Avoid warning if only 1 class
             logger.warning(f"{LOG_EMOJIS['split']} Some classes have fewer than {n_splits} samples. StratifiedKFold requires at least n_splits samples per class. Consider reducing n_splits or using non-stratified KFold.")
             # For now, we proceed, but scikit-learn might raise an error.

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
                logger.debug(f"Yielding CV Fold {fold+1}/{n_splits} (Train/Val for HP Tuning)")
                train_subset = Subset(base_dataset, train_idx)
                val_subset = Subset(base_dataset, val_idx)
                yield train_subset, val_subset
        except ValueError as e:
            logger.error(f"{LOG_EMOJIS['error']} StratifiedKFold failed: {e}. This often happens if a class has fewer members than n_splits.")
            raise e


    def get_cv_eval_folds(self, n_splits: int = 5, val_size_ratio: float = 0.25, random_state: Optional[int] = None
                         ) -> Generator[Tuple[Subset, Subset, Subset], None, None]:
        """
        Yields (train_subset, val_subset, test_subset) for K-Fold CV on the *entire* dataset.
        The 'test' set is one fold, and the remaining data is split into train/val.
        Used for evaluating hyperparameter robustness (Step 2).
        **Only valid if has_fixed_test_set is False.**
        """
        if self.has_fixed_test_set:
             raise ValueError("Cannot perform K-Fold evaluation splitting (Step 2) when a fixed test set is provided.")

        logger.info(f"{LOG_EMOJIS['fold']} Generating {n_splits} train/val/test CV folds from the full dataset for evaluation.")
        base_dataset = self.get_full_dataset()
        indices = np.arange(len(base_dataset))
        labels = np.array(base_dataset.labels)

        if len(indices) == 0:
            raise ValueError("No data available for K-Fold splitting.")
        if len(set(labels)) < n_splits:
             logger.warning(f"{LOG_EMOJIS['split']} Number of classes ({len(set(labels))}) is less than n_splits ({n_splits}). Stratification might behave unexpectedly or fail.")
        # Check if any class has fewer samples than n_splits
        unique_labels, counts = np.unique(labels, return_counts=True)
        if any(count < n_splits for count in counts) and len(set(labels)) > 1:
             logger.warning(f"{LOG_EMOJIS['split']} Some classes have fewer than {n_splits} samples. StratifiedKFold requires at least n_splits samples per class. Cannot perform robust evaluation CV.")
             raise ValueError(f"Cannot perform StratifiedKFold for evaluation: classes {[unique_labels[i] for i, count in enumerate(counts) if count < n_splits]} have fewer than {n_splits} samples.")


        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels)):
            logger.debug(f"Processing CV Fold {fold+1}/{n_splits} for Robustness Evaluation")
            test_subset = Subset(base_dataset, test_idx)

            # Split the remaining train_val_idx into train and validation
            current_train_val_labels = labels[train_val_idx]

            if len(train_val_idx) == 0:
                logger.warning(f"Fold {fold+1}: No data left for training/validation after selecting test set.")
                # Yield empty subsets? Or skip? Let's skip this fold with a warning.
                continue

            if len(set(current_train_val_labels)) < 2:
                 logger.warning(f"{LOG_EMOJIS['split']} Fold {fold+1}: Not enough classes in train+val data for stratification. Performing random split.")
                 train_idx_inner, val_idx_inner = train_test_split(train_val_idx, test_size=val_size_ratio, random_state=random_state)
            else:
                try:
                    train_idx_inner, val_idx_inner = train_test_split(
                        train_val_idx,
                        test_size=val_size_ratio, # val_size_ratio is proportion of the train_val data
                        random_state=random_state,
                        stratify=current_train_val_labels
                    )
                except ValueError as e:
                    logger.warning(f"{LOG_EMOJIS['split']} Fold {fold+1}: Inner stratification failed: {e}. Falling back to random split.")
                    train_idx_inner, val_idx_inner = train_test_split(train_val_idx, test_size=val_size_ratio, random_state=random_state)


            train_subset = Subset(base_dataset, train_idx_inner)
            val_subset = Subset(base_dataset, val_idx_inner)
            logger.debug(f"  Fold {fold+1} Split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")

            yield train_subset, val_subset, test_subset

# --- 4. Dummy Model ---
class SimpleCNN(nn.Module):
    """A simple CNN model for demonstration."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # Reduces H, W by factor of 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # Reduces H, W by factor of 2
        # Example: If input is 64x64 -> pool1 -> 32x32 -> pool2 -> 16x16
        # Adjust the input features to the linear layer accordingly
        # Let's assume input image size is 64x64 for calculation
        # self.fc_input_features = 32 * (input_size // 4) * (input_size // 4)
        self.fc_input_features = 32 * 16 * 16 # Hardcoding for 64x64 example
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

        # Placeholder for dynamic calculation (requires input tensor shape)
        self._feature_size_calculated = False

    def _calculate_feature_size(self, input_shape: Tuple[int, int, int, int]):
        # Input shape: (N, C, H, W)
        if not self._feature_size_calculated:
             with torch.no_grad():
                 dummy_input = torch.zeros(input_shape)
                 x = self.pool1(self.relu1(self.conv1(dummy_input)))
                 x = self.pool2(self.relu2(self.conv2(x)))
                 self.fc_input_features = x.numel() // x.shape[0] # numel = N * C * H * W
                 self.fc1 = nn.Linear(self.fc_input_features, 128).to(dummy_input.device) # Recreate layer
                 self.fc2 = nn.Linear(128, self.fc2.out_features).to(dummy_input.device) # Recreate layer
                 self._feature_size_calculated = True
                 logger.info(f"{LOG_EMOJIS['model']} Calculated FC input features: {self.fc_input_features}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
         # Dynamically calculate FC layer size on first forward pass if needed
         if not self._feature_size_calculated:
             self._calculate_feature_size(x.shape)

         x = self.pool1(self.relu1(self.conv1(x)))
         x = self.pool2(self.relu2(self.conv2(x)))
         x = torch.flatten(x, 1) # Flatten all dimensions except batch
         x = self.relu3(self.fc1(x))
         x = self.fc2(x)
         return x

# --- 5. Early Stopping ---
class EarlyStopping:
    """Stops training when validation loss doesn't improve."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, verbose: bool = True,
                 path: str = 'best_model.pth', trace_func=logger.info):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss # We want to maximize this (minimize loss)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            self.trace_func(f"⏳ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f"{LOG_EMOJIS['save']} Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        self.best_model_state = model.state_dict()
        # Optional: Save directly to file here if needed persistently across runs
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- 6. The Common Train/Evaluate Function ---
def train_and_evaluate(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    early_stopping_patience: Optional[int] = 7,
    fold_num: Optional[int] = None # For logging purposes
    ) -> Dict[str, Any]:
    """
    Trains and/or evaluates a model on given data loaders.

    Args:
        model: The PyTorch model.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on (CPU/GPU).
        num_epochs: Max number of epochs to train.
        train_loader: DataLoader for training data. If None, training is skipped.
        val_loader: DataLoader for validation data. Used for early stopping if provided.
        test_loader: DataLoader for test data. If None, final testing is skipped.
        early_stopping_patience: Patience for early stopping based on validation loss. If None, disabled.
        fold_num: Optional identifier for the fold (for logging).

    Returns:
        A dictionary containing collected metrics like losses and accuracies.
    """
    fold_prefix = f"[Fold {fold_num}] " if fold_num else ""
    logger.info(f"{LOG_EMOJIS['train']} {fold_prefix}Starting Training & Evaluation Cycle...")

    model.to(device)
    results = defaultdict(list)
    best_val_loss = np.inf
    best_epoch = -1

    early_stopper = None
    if early_stopping_patience is not None and val_loader is not None:
        early_stopper = EarlyStopping(patience=early_stopping_patience, verbose=True,
                                     trace_func=lambda msg: logger.info(f"{fold_prefix}{msg}"))

    if train_loader is None and val_loader is None and test_loader is None:
        logger.warning(f"{LOG_EMOJIS['error']} {fold_prefix}No data loaders provided. Skipping training and evaluation.")
        return dict(results)

    if train_loader is None:
        logger.info(f"{LOG_EMOJIS['info']} {fold_prefix}No train_loader provided. Skipping training phase.")
        num_epochs = 1 # Only need one "epoch" for evaluation if no training

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_log_prefix = f"{fold_prefix}Epoch {epoch+1}/{num_epochs}"

        # --- Training Phase ---
        if train_loader:
            model.train()
            running_loss = 0.0
            train_preds, train_targets = [], []
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_targets.extend(labels.cpu().numpy())

            epoch_train_loss = running_loss / len(train_loader.dataset) # Use dataset length for avg
            epoch_train_acc = accuracy_score(train_targets, train_preds)
            results['train_loss'].append(epoch_train_loss)
            results['train_accuracy'].append(epoch_train_acc)
            train_log = f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
        else:
            train_log = "Training skipped"
            results['train_loss'].append(None)
            results['train_accuracy'].append(None)


        # --- Validation Phase ---
        val_log = ""
        epoch_val_loss = None
        epoch_val_acc = None
        if val_loader:
            model.eval()
            running_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            epoch_val_loss = running_loss / len(val_loader.dataset)
            epoch_val_acc = accuracy_score(val_targets, val_preds)
            results['val_loss'].append(epoch_val_loss)
            results['val_accuracy'].append(epoch_val_acc)
            val_log = f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"

            # Early Stopping Check
            if early_stopper:
                early_stopper(epoch_val_loss, model)
                if early_stopper.early_stop:
                    logger.info(f"{LOG_EMOJIS['stop']} {epoch_log_prefix} Early stopping triggered!")
                    # Store the best epoch number based on validation loss
                    best_epoch = epoch + 1 - early_stopping_patience # Epoch where loss started increasing
                    break # Stop training loop
        else:
            val_log = "Validation skipped"
            results['val_loss'].append(None)
            results['val_accuracy'].append(None)

        epoch_duration = time.time() - epoch_start_time
        logger.info(f"{LOG_EMOJIS['train']} {epoch_log_prefix} - {train_log} | {val_log} ({epoch_duration:.2f}s)")

        # Track overall best val loss (if not early stopping or if it finishes)
        if epoch_val_loss is not None and epoch_val_loss < best_val_loss:
             best_val_loss = epoch_val_loss
             best_epoch = epoch + 1 # Current epoch is the best so far

    # --- Final Evaluation on Test Set (if requested) ---
    test_log = "Test skipped"
    if test_loader:
        logger.info(f"{LOG_EMOJIS['test']} {fold_prefix}Evaluating on Test Set...")
        # Load best model state if early stopping was used and saved a state
        if early_stopper and early_stopper.best_model_state:
            logger.info(f"{LOG_EMOJIS['best']} {fold_prefix}Loading best model state from epoch {best_epoch} for testing.")
            model.load_state_dict(early_stopper.best_model_state)

        model.eval()
        running_loss = 0.0
        test_preds, test_targets = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Calculate test loss if needed
                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())

        test_loss = running_loss / len(test_loader.dataset)
        test_acc = accuracy_score(test_targets, test_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted', zero_division=0)

        results['test_loss'] = test_loss
        results['test_accuracy'] = test_acc
        results['test_precision'] = precision
        results['test_recall'] = recall
        results['test_f1'] = f1
        test_log = f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {f1:.4f}"
        logger.info(f"{LOG_EMOJIS['result']} {fold_prefix}Final Test Results - {test_log}")
    else:
        # Ensure keys exist even if test is skipped
        results['test_loss'] = None
        results['test_accuracy'] = None
        results['test_precision'] = None
        results['test_recall'] = None
        results['test_f1'] = None


    # Store best epoch number
    results['best_epoch'] = best_epoch if early_stopper and early_stopper.early_stop else num_epochs

    logger.info(f"{LOG_EMOJIS['done']} {fold_prefix}Training & Evaluation Cycle Finished.")
    return dict(results)


# --- 7. Pipeline Orchestrator ---

def run_pipeline(config: Dict[str, Any]):
    """
    Runs the image classification old_pipeline based on the provided configuration.

    Args:
        config (Dict[str, Any]): A dictionary containing old_pipeline settings:
            - dataset_path (str): Path to the root dataset directory.
            - model_name (str): Identifier for the model (e.g., 'SimpleCNN', 'ResNet18').
            - output_base_dir (str): Base directory to save results and logs.
            - image_size (int): Target size for resizing images (e.g., 64 for 64x64).
            - batch_size (int): Batch size for DataLoaders.
            - num_workers (int): Number of workers for DataLoaders.
            - device (str): 'cuda', 'cpu', or 'auto'.
            - random_state (int): Seed for reproducibility.

            - run_mode (str): One of 'hp_tuning', 'cv_evaluate', 'simple_split', 'fixed_test'.
            - num_epochs (int): Max epochs for training.
            - early_stopping (Optional[int]): Patience for early stopping (e.g., 7). None to disable.

            - # Mode-specific options:
            - # For 'hp_tuning'
            - hp_param_grid (Optional[Dict[str, List]]): Grid for hyperparameter search (e.g., {'lr': [0.01, 0.001], 'optimizer': ['Adam', 'SGD']}).
            - hp_cv_splits (int): Number of CV folds for hyperparameter tuning (Step 1).

            - # For 'cv_evaluate'
            - eval_cv_splits (int): Number of CV folds for robustness evaluation (Step 2).
            - eval_cv_val_ratio (float): Proportion of non-test data used for validation within each eval fold (e.g., 0.25).
            - best_hyperparams (Optional[Dict]): Best hyperparameters found (or fixed) needed for this mode.

            - # For 'simple_split' or 'fixed_test'
            - fixed_hyperparams (Optional[Dict]): Fixed hyperparameters for these modes (e.g., {'lr': 0.001, 'optimizer': 'Adam'}).
            - simple_split_test_size (float): Proportion for test set in 'simple_split' mode (e.g., 0.2).
            - simple_split_val_size (float): Proportion for val set in 'simple_split' mode (e.g., 0.2).

            - # For 'fixed_test'
            - fixed_test_val_size (float): Proportion of the 'train' data to use for validation (e.g., 0.2).
    """
    start_time = time.time()

    # --- Setup ---
    output_dir = os.path.join(config['output_base_dir'], config['model_name'], os.path.basename(config['dataset_path'].rstrip('/\\')))
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    global logger # Access the global logger
    # Reconfigure file handler for this specific run
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            break # Assume only one file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info(f"{LOG_EMOJIS['start']} Pipeline started with configuration:")
    logger.info(json.dumps(config, indent=2))

    # Seed for reproducibility
    seed = config.get('random_state', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Potentially add deterministic flags, but they can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # Device
    if config['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['device'])
    logger.info(f"{LOG_EMOJIS['info']} Using device: {device}")

    # --- Transforms ---
    img_size = config['image_size']
    # Basic transforms - can be made more complex/configurable
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Add Augmentations here if needed (e.g., RandomHorizontalFlip)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- Load Data ---
    try:
        dataset_handler = DatasetHandler(config['dataset_path'], transform=eval_transform) # Use eval transform initially
    except Exception as e:
        logger.error(f"{LOG_EMOJIS['error']} Failed to initialize DatasetHandler: {e}")
        return

    # --- Compatibility Checks ---
    run_mode = config['run_mode']
    if run_mode == 'cv_evaluate' and dataset_handler.has_fixed_test_set:
        logger.error(f"{LOG_EMOJIS['error']} Compatibility Error: Cannot run 'cv_evaluate' (Step 2) mode with a fixed train/test dataset structure.")
        return
    if run_mode == 'fixed_test' and not dataset_handler.has_fixed_test_set:
        logger.error(f"{LOG_EMOJIS['error']} Compatibility Error: Cannot run 'fixed_test' mode without a fixed train/test dataset structure.")
        return
    if run_mode == 'simple_split' and dataset_handler.has_fixed_test_set:
        logger.warning(f"{LOG_EMOJIS['info']} Running 'simple_split' mode with a fixed test set. The test split will be ignored, and the fixed test set will be used for evaluation after training on a train/val split of the 'train' data.")
        # Adjust behavior later if needed, or consider this an error? For now, proceed but evaluate on fixed test.
        run_mode = 'fixed_test' # Effectively becomes fixed_test mode
        config['run_mode'] = run_mode # Update config for clarity
        logger.info(f"{LOG_EMOJIS['info']} Adjusted run_mode to 'fixed_test'.")


    # --- Define Model --- (Use dummy for now)
    # This part should be flexible to load different models
    if config['model_name'] == 'SimpleCNN':
        model = SimpleCNN(num_classes=dataset_handler.num_classes)
        logger.info(f"{LOG_EMOJIS['model']} Using SimpleCNN model.")
    else:
        # Placeholder for loading other models (e.g., torchvision.models)
        logger.warning(f"{LOG_EMOJIS['model']} Model '{config['model_name']}' not fully implemented, using SimpleCNN.")
        model = SimpleCNN(num_classes=dataset_handler.num_classes)
        # Example:
        # if config['model_name'] == 'resnet18':
        #    import torchvision.models as models
        #    model = models.resnet18(pretrained=config.get('pretrained', False))
        #    num_ftrs = model.fc.in_features
        #    model.fc = nn.Linear(num_ftrs, dataset_handler.num_classes)


    # --- Select Pipeline Branch ---
    results_file_path = None

    try:
        # --- Mode 1: Simple Split (Train/Val/Test) ---
        if run_mode == 'simple_split':
            logger.info(f"{LOG_EMOJIS['split']} --- Running Mode: Simple Train/Val/Test Split ---")
            hyperparams = config.get('fixed_hyperparams', {'lr': 0.001, 'optimizer': 'Adam'})
            logger.info(f"{LOG_EMOJIS['info']} Using fixed hyperparameters: {hyperparams}")

            train_subset, val_subset, test_subset = dataset_handler.get_train_val_test_split(
                test_size=config['simple_split_test_size'],
                val_size=config['simple_split_val_size'],
                random_state=seed
            )
            # Apply train transform to train_subset
            train_subset.dataset.transform = train_transform

            train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
            val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
            test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

            # Create model instance for this run
            current_model = SimpleCNN(num_classes=dataset_handler.num_classes) # Re-instantiate

            # Optimizer and Loss
            lr = hyperparams.get('lr', 0.001)
            if hyperparams.get('optimizer', 'Adam').lower() == 'adam':
                optimizer = optim.Adam(current_model.parameters(), lr=lr)
            else: # Default or SGD
                optimizer = optim.SGD(current_model.parameters(), lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            results = train_and_evaluate(
                model=current_model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                num_epochs=config['num_epochs'],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                early_stopping_patience=config.get('early_stopping')
            )

            # Save results
            hparam_str = "_".join([f"{k}_{v}" for k, v in hyperparams.items()])
            results_file_path = os.path.join(output_dir, f"results_simple_split_{hparam_str}.json")
            logger.info(f"{LOG_EMOJIS['save']} Saving results to {results_file_path}")
            with open(results_file_path, 'w') as f:
                 json.dump(results, f, indent=4)

        # --- Mode 2: Fixed Test Set Evaluation ---
        elif run_mode == 'fixed_test':
             logger.info(f"{LOG_EMOJIS['split']} --- Running Mode: Fixed Test Set Evaluation ---")
             hyperparams = config.get('fixed_hyperparams', {'lr': 0.001, 'optimizer': 'Adam'})
             logger.info(f"{LOG_EMOJIS['info']} Using fixed hyperparameters: {hyperparams}")

             # Get train/val split from the 'train' data
             train_subset, val_subset, _ = dataset_handler.get_train_val_test_split(
                 test_size=0, # No test split from train data
                 val_size=config['fixed_test_val_size'], # Val size relative to train data
                 random_state=seed
             )
             # Apply train transform to train_subset
             train_subset.dataset.transform = train_transform

             # Get the fixed test dataset
             test_dataset = dataset_handler.get_fixed_test_dataset()
             if test_dataset is None:
                  logger.error(f"{LOG_EMOJIS['error']} In 'fixed_test' mode, but could not retrieve fixed test dataset.")
                  return # Should have been caught by compatibility check, but double-check

             train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
             val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
             test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

             # Create model instance for this run
             current_model = SimpleCNN(num_classes=dataset_handler.num_classes)

             # Optimizer and Loss
             lr = hyperparams.get('lr', 0.001)
             if hyperparams.get('optimizer', 'Adam').lower() == 'adam':
                 optimizer = optim.Adam(current_model.parameters(), lr=lr)
             else: # Default or SGD
                 optimizer = optim.SGD(current_model.parameters(), lr=lr, momentum=0.9)
             criterion = nn.CrossEntropyLoss()

             results = train_and_evaluate(
                 model=current_model,
                 optimizer=optimizer,
                 criterion=criterion,
                 device=device,
                 num_epochs=config['num_epochs'],
                 train_loader=train_loader,
                 val_loader=val_loader,
                 test_loader=test_loader,
                 early_stopping_patience=config.get('early_stopping')
             )

             # Save results
             hparam_str = "_".join([f"{k}_{v}" for k, v in hyperparams.items()])
             results_file_path = os.path.join(output_dir, f"results_fixed_test_{hparam_str}.json")
             logger.info(f"{LOG_EMOJIS['save']} Saving results to {results_file_path}")
             with open(results_file_path, 'w') as f:
                 json.dump(results, f, indent=4)


        # --- Mode 3: Hyperparameter Tuning (Step 1) ---
        elif run_mode == 'hp_tuning':
            logger.info(f"{LOG_EMOJIS['hyper']} --- Running Mode: Hyperparameter Tuning (CV on Train) ---")
            param_grid = config.get('hp_param_grid')
            if not param_grid:
                logger.error(f"{LOG_EMOJIS['error']} 'hp_param_grid' is required for 'hp_tuning' mode.")
                return

            n_splits = config.get('hp_cv_splits', 5)
            # TODO: Implement actual grid/random search logic
            # This is a simplified example iterating through a grid manually.
            # A more robust implementation might use libraries or more structured loops.

            best_avg_val_metric = -np.inf # Assuming higher metric is better (e.g., accuracy)
            best_params = None
            all_hp_results = {}

            # Generate hyperparameter combinations (simple grid search example)
            keys = list(param_grid.keys())
            value_combinations = np.array(np.meshgrid(*[param_grid[k] for k in keys])).T.reshape(-1, len(keys))

            logger.info(f"{LOG_EMOJIS['hyper']} Starting search over {len(value_combinations)} hyperparameter combinations.")

            for i, values in enumerate(value_combinations):
                current_hyperparams = {keys[j]: values[j] for j in range(len(keys))}
                hparam_str = "_".join([f"{k}_{v}" for k, v in current_hyperparams.items()])
                logger.info(f"\n--- Testing Hyperparameters ({i+1}/{len(value_combinations)}): {current_hyperparams} ---")

                fold_val_metrics = [] # Store validation metric (e.g., accuracy) for each fold

                try:
                    for fold, (train_subset, val_subset) in enumerate(dataset_handler.get_cv_train_val_folds(n_splits=n_splits, random_state=seed)):
                        logger.info(f"-- Fold {fold+1}/{n_splits} for HP set: {current_hyperparams} --")
                        # Apply train transform only to this fold's train subset
                        # Note: This assumes the base dataset in DatasetHandler used eval_transform.
                        # Need a way to clone the subset or dataset with different transform.
                        # Quick Fix: Modify transform in place (affects subsequent folds if not careful)
                        # Better Fix: Create new Dataset instances or handle transforms in DataLoader?
                        # Let's apply it temporarily to the base dataset referenced by the subset.
                        original_transform = train_subset.dataset.transform
                        train_subset.dataset.transform = train_transform

                        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
                        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

                        # Restore original transform for the next fold's validation set
                        train_subset.dataset.transform = original_transform

                        # Create model instance for this fold/hp combination
                        current_model = SimpleCNN(num_classes=dataset_handler.num_classes)

                        # Optimizer and Loss
                        lr = float(current_hyperparams.get('lr', 0.001)) # Ensure type
                        opt_name = current_hyperparams.get('optimizer', 'Adam')
                        if opt_name.lower() == 'adam':
                            optimizer = optim.Adam(current_model.parameters(), lr=lr)
                        else:
                            optimizer = optim.SGD(current_model.parameters(), lr=lr, momentum=0.9)
                        criterion = nn.CrossEntropyLoss()

                        fold_results = train_and_evaluate(
                            model=current_model,
                            optimizer=optimizer,
                            criterion=criterion,
                            device=device,
                            num_epochs=config['num_epochs'],
                            train_loader=train_loader,
                            val_loader=val_loader,
                            test_loader=None, # No testing during HP tuning folds
                            early_stopping_patience=config.get('early_stopping'),
                            fold_num=fold + 1
                        )

                        # Get validation metric from the last epoch or best epoch if early stopping
                        # Using validation accuracy here, adjust if needed (e.g., val_loss)
                        last_val_acc = fold_results['val_accuracy'][-1] if fold_results['val_accuracy'] else 0
                        fold_val_metrics.append(last_val_acc)
                        logger.info(f"-- Fold {fold+1} Val Accuracy: {last_val_acc:.4f} --")

                except ValueError as e: # Catch KFold errors
                     logger.error(f"{LOG_EMOJIS['error']} Error during CV fold generation for HP set {current_hyperparams}: {e}. Skipping this HP set.")
                     continue # Skip to the next hyperparameter set

                avg_val_metric = np.mean(fold_val_metrics) if fold_val_metrics else 0
                logger.info(f"--- Avg Val Accuracy for HP set {current_hyperparams}: {avg_val_metric:.4f} ---")
                all_hp_results[hparam_str] = {'params': current_hyperparams, 'avg_val_metric': avg_val_metric, 'fold_metrics': fold_val_metrics}

                if avg_val_metric > best_avg_val_metric:
                    best_avg_val_metric = avg_val_metric
                    best_params = current_hyperparams
                    logger.info(f"{LOG_EMOJIS['best']} New best hyperparameters found: {best_params} (Avg Val Metric: {best_avg_val_metric:.4f})")

            logger.info(f"\n{LOG_EMOJIS['hyper']} --- Hyperparameter Tuning Finished ---")
            logger.info(f"{LOG_EMOJIS['best']} Best hyperparameters found: {best_params}")
            logger.info(f"{LOG_EMOJIS['best']} Best average validation metric (accuracy): {best_avg_val_metric:.4f}")

            # Save tuning results
            results_file_path = os.path.join(output_dir, "results_hp_tuning.json")
            logger.info(f"{LOG_EMOJIS['save']} Saving hyperparameter tuning results to {results_file_path}")
            output_data = {
                'best_params': best_params,
                'best_avg_val_metric': best_avg_val_metric,
                'all_hp_results': all_hp_results
            }
            with open(results_file_path, 'w') as f:
                # Convert numpy types for JSON serialization if necessary
                 def convert(o):
                     if isinstance(o, np.generic): return o.item()
                     raise TypeError
                 json.dump(output_data, f, indent=4, default=convert)

        # --- Mode 4: Cross-Validation Evaluation (Step 2) ---
        elif run_mode == 'cv_evaluate':
             logger.info(f"{LOG_EMOJIS['fold']} --- Running Mode: Cross-Validation Evaluation (Robustness) ---")
             best_hyperparams = config.get('best_hyperparams')
             if not best_hyperparams:
                 logger.error(f"{LOG_EMOJIS['error']} 'best_hyperparams' are required for 'cv_evaluate' mode.")
                 return

             n_splits = config.get('eval_cv_splits', 5)
             val_ratio = config.get('eval_cv_val_ratio', 0.25)
             hparam_str = "_".join([f"{k}_{v}" for k, v in best_hyperparams.items()])
             logger.info(f"{LOG_EMOJIS['info']} Evaluating robustness using hyperparameters: {best_hyperparams}")

             fold_test_metrics = defaultdict(list) # Store test metrics per fold (e.g., acc, f1)
             all_fold_results = []

             try:
                 for fold, (train_subset, val_subset, test_subset) in enumerate(
                     dataset_handler.get_cv_eval_folds(n_splits=n_splits, val_size_ratio=val_ratio, random_state=seed)
                 ):
                     logger.info(f"\n--- Evaluating CV Fold {fold+1}/{n_splits} ---")

                     # Apply transforms (similar caution as in hp_tuning)
                     original_transform = train_subset.dataset.transform # Assumes base dataset uses eval
                     train_subset.dataset.transform = train_transform

                     train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
                     val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
                     test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

                     # Restore transform
                     train_subset.dataset.transform = original_transform

                     # Create model instance for this fold
                     current_model = SimpleCNN(num_classes=dataset_handler.num_classes)

                     # Optimizer and Loss using best hyperparameters
                     lr = float(best_hyperparams.get('lr', 0.001))
                     opt_name = best_hyperparams.get('optimizer', 'Adam')
                     if opt_name.lower() == 'adam':
                         optimizer = optim.Adam(current_model.parameters(), lr=lr)
                     else:
                         optimizer = optim.SGD(current_model.parameters(), lr=lr, momentum=0.9)
                     criterion = nn.CrossEntropyLoss()

                     fold_results = train_and_evaluate(
                         model=current_model,
                         optimizer=optimizer,
                         criterion=criterion,
                         device=device,
                         num_epochs=config['num_epochs'],
                         train_loader=train_loader,
                         val_loader=val_loader,
                         test_loader=test_loader,
                         early_stopping_patience=config.get('early_stopping'),
                         fold_num=fold + 1
                     )

                     all_fold_results.append(fold_results)
                     # Collect test metrics from this fold
                     if fold_results.get('test_accuracy') is not None:
                         fold_test_metrics['accuracy'].append(fold_results['test_accuracy'])
                         fold_test_metrics['f1'].append(fold_results['test_f1'])
                         fold_test_metrics['precision'].append(fold_results['test_precision'])
                         fold_test_metrics['recall'].append(fold_results['test_recall'])
                         fold_test_metrics['loss'].append(fold_results['test_loss'])
                     else:
                         logger.warning(f"{LOG_EMOJIS['info']} Fold {fold+1} did not produce test results (possibly skipped).")

             except ValueError as e: # Catch KFold errors
                 logger.error(f"{LOG_EMOJIS['error']} Error during CV fold generation for evaluation: {e}. Aborting CV evaluation.")
                 return # Cannot proceed if folds fail

             # Calculate overall statistics
             final_stats = {}
             if fold_test_metrics['accuracy']: # Check if we got any results
                for metric_name, values in fold_test_metrics.items():
                     final_stats[f'mean_{metric_name}'] = np.mean(values)
                     final_stats[f'std_{metric_name}'] = np.std(values)
                     final_stats[f'min_{metric_name}'] = np.min(values)
                     final_stats[f'max_{metric_name}'] = np.max(values)

                logger.info(f"\n{LOG_EMOJIS['result']} --- Cross-Validation Evaluation Summary ({n_splits} Folds) ---")
                logger.info(f"Hyperparameters: {best_hyperparams}")
                logger.info(f"Test Accuracy: Mean={final_stats['mean_accuracy']:.4f}, Std={final_stats['std_accuracy']:.4f}")
                logger.info(f"Test F1 Score: Mean={final_stats['mean_f1']:.4f}, Std={final_stats['std_f1']:.4f}")
                # Log other stats as needed
             else:
                 logger.warning(f"{LOG_EMOJIS['error']} No test metrics collected across CV folds. Cannot compute final statistics.")


             # Save detailed fold results and summary stats
             results_file_path = os.path.join(output_dir, f"results_cv_evaluate_{hparam_str}.json")
             logger.info(f"{LOG_EMOJIS['save']} Saving CV evaluation results to {results_file_path}")
             output_data = {
                 'best_hyperparams_used': best_hyperparams,
                 'summary_stats': final_stats,
                 'individual_fold_results': all_fold_results # Contains epoch-wise data too
             }
             with open(results_file_path, 'w') as f:
                  # Convert numpy types for JSON serialization if necessary
                  def convert(o):
                      if isinstance(o, np.generic): return o.item()
                      if isinstance(o, np.ndarray): return o.tolist()
                      if isinstance(o, (torch.Tensor)): return o.tolist() # Handle potential tensors
                      raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')
                  json.dump(output_data, f, indent=4, default=convert)


        else:
             logger.error(f"{LOG_EMOJIS['error']} Invalid 'run_mode': {run_mode}. Choose from 'hp_tuning', 'cv_evaluate', 'simple_split', 'fixed_test'.")

    except Exception as e:
         logger.error(f"{LOG_EMOJIS['error']} An unexpected error occurred during old_pipeline execution: {e}", exc_info=True) # Log traceback
    finally:
        end_time = time.time()
        total_duration = end_time - start_time
        logger.info(f"{LOG_EMOJIS['done']} Pipeline finished in {total_duration:.2f} seconds.")
        if results_file_path and os.path.exists(results_file_path):
             logger.info(f"{LOG_EMOJIS['save']} Final results saved to: {results_file_path}")
        else:
             logger.info(f"{LOG_EMOJIS['info']} No results file generated or saved for this run.")


# --- 8. Example Usage ---
if __name__ == "__main__":
    # --- Define Configurations ---

    # Config 1: Simple split on FLAT dataset
    config_simple_flat = {
        "dataset_path": DUMMY_DATA_DIR_FLAT,
        "model_name": "SimpleCNN",
        "output_base_dir": "pipeline_outputs",
        "image_size": 64,
        "batch_size": 8, # Reduced batch size for small dataset
        "num_workers": 0, # Easier for debugging
        "device": "auto",
        "random_state": 42,
        "run_mode": "simple_split",
        "num_epochs": 3, # Few epochs for demo
        "early_stopping": 3,
        "fixed_hyperparams": {"lr": 0.001, "optimizer": "Adam"},
        "simple_split_test_size": 0.2,
        "simple_split_val_size": 0.2,
    }

    # Config 2: Fixed test evaluation on SPLIT dataset
    config_fixed_test = {
        "dataset_path": DUMMY_DATA_DIR_SPLIT,
        "model_name": "SimpleCNN",
        "output_base_dir": "pipeline_outputs",
        "image_size": 64,
        "batch_size": 8,
        "num_workers": 0,
        "device": "auto",
        "random_state": 42,
        "run_mode": "fixed_test",
        "num_epochs": 3,
        "early_stopping": 3,
        "fixed_hyperparams": {"lr": 0.002, "optimizer": "SGD"},
        "fixed_test_val_size": 0.25, # 25% of the 'train' data used for validation
    }

    # Config 3: Hyperparameter tuning on FLAT dataset (Step 1)
    config_hp_tuning = {
        "dataset_path": DUMMY_DATA_DIR_FLAT,
        "model_name": "SimpleCNN",
        "output_base_dir": "pipeline_outputs",
        "image_size": 64,
        "batch_size": 8,
        "num_workers": 0,
        "device": "auto",
        "random_state": 42,
        "run_mode": "hp_tuning",
        "num_epochs": 2, # Very few epochs for HP tuning demo
        "early_stopping": None, # Often disable early stopping during HP tuning
        "hp_param_grid": {
            "lr": [0.01, 0.001],
            "optimizer": ["Adam", "SGD"]
        },
        "hp_cv_splits": 3, # Reduced folds for small dataset
    }

    # Config 4: CV evaluation on FLAT dataset (Step 2) - Requires best HPs
    # Assume HP tuning found {'lr': 0.001, 'optimizer': 'Adam'} as best
    config_cv_eval = {
        "dataset_path": DUMMY_DATA_DIR_FLAT,
        "model_name": "SimpleCNN",
        "output_base_dir": "pipeline_outputs",
        "image_size": 64,
        "batch_size": 8,
        "num_workers": 0,
        "device": "auto",
        "random_state": 42,
        "run_mode": "cv_evaluate",
        "num_epochs": 3,
        "early_stopping": 3,
        "eval_cv_splits": 3, # Must be <= samples per class
        "eval_cv_val_ratio": 0.25,
        "best_hyperparams": {"lr": 0.001, "optimizer": "Adam"}, # Must provide best HPs
    }


    # --- Run Selected Pipeline ---
    print("\n--- Running Pipeline: Simple Split (Flat Dataset) ---")
    run_pipeline(config_simple_flat)

    print("\n--- Running Pipeline: Fixed Test (Split Dataset) ---")
    run_pipeline(config_fixed_test)

    print("\n--- Running Pipeline: HP Tuning (Flat Dataset) ---")
    # Load best params from this run if needed for CV eval later
    run_pipeline(config_hp_tuning)

    # Check if HP tuning results exist to get best params for next step
    hp_results_path = os.path.join(config_cv_eval['output_base_dir'], config_cv_eval['model_name'], os.path.basename(config_cv_eval['dataset_path'].rstrip('/\\')), "results_hp_tuning.json")
    if os.path.exists(hp_results_path):
         with open(hp_results_path, 'r') as f:
             hp_data = json.load(f)
             best_hps = hp_data.get('best_params')
             if best_hps:
                 print(f"\n--- Running Pipeline: CV Evaluation (Flat Dataset using found HPs: {best_hps}) ---")
                 config_cv_eval["best_hyperparams"] = best_hps # Update with found HPs
                 run_pipeline(config_cv_eval)
             else:
                  print(f"\n--- Skipping CV Evaluation: Best HPs not found in {hp_results_path} ---")
    else:
         print(f"\n--- Skipping CV Evaluation: HP tuning results file not found ({hp_results_path}). Running with default best HPs. ---")
         # Optionally run with the default best HPs defined in config_cv_eval
         run_pipeline(config_cv_eval)


    # --- Error Handling Examples ---
    # print("\n--- Running Pipeline: Incompatible Config (CV Eval on Split Dataset) ---")
    # error_config_1 = config_cv_eval.copy()
    # error_config_1["dataset_path"] = DUMMY_DATA_DIR_SPLIT # Change to split dataset
    # run_pipeline(error_config_1) # Should log an error and exit

    # print("\n--- Running Pipeline: Incompatible Config (Fixed Test on Flat Dataset) ---")
    # error_config_2 = config_fixed_test.copy()
    # error_config_2["dataset_path"] = DUMMY_DATA_DIR_FLAT # Change to flat dataset
    # run_pipeline(error_config_2) # Should log an error and exit

    print("\n--- Pipeline Demonstrations Complete ---")