import os
import logging
from typing import Tuple, List, Dict, Optional, Any, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

from utils import logger


class CustomImageDataset(Dataset):
    """Custom PyTorch Dataset that loads images from paths."""

    def __init__(self, image_paths: List[str], labels: List[int], transform: Optional[Callable] = None):
        """
        Args:
            image_paths (List[str]): List of paths to images.
            labels (List[int]): List of corresponding labels (integers).
            transform (Optional[Callable]): torchvision transforms to apply.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            # Open image using PIL (robust to different formats)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"❌ Error loading image {img_path}: {e}")
            # Return a placeholder or raise an error - returning placeholder might hide issues
            # For simplicity, let's return a dummy tensor and the label
            # A better approach might be to filter out problematic images beforehand
            return torch.zeros((3, 224, 224)), label  # Assuming 3 channels, 224x224

    def get_labels(self) -> List[int]:
        """Returns the list of all labels in the dataset."""
        return self.labels


class DatasetHandler:
    """Handles dataset loading, structure detection, and splitting."""

    def __init__(self, dataset_path: str, image_size: Tuple[int, int] = (224, 224), seed: int = 42):
        """
        Args:
            dataset_path (str): Path to the root dataset folder.
            image_size (Tuple[int, int]): Target image size (height, width).
            seed (int): Random seed for reproducible splits.
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.seed = seed
        self._structure: Optional[str] = None
        self._classes: Optional[List[str]] = None
        self._class_to_idx: Optional[Dict[str, int]] = None

        # Data storage
        self._all_image_paths: List[str] = []
        self._all_labels: List[int] = []
        self._train_image_paths: List[str] = []
        self._train_labels: List[int] = []
        self._test_image_paths: List[str] = []
        self._test_labels: List[int] = []

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        self._detect_structure()
        self._load_data()
        logger.info(
            f"📁 Dataset '{self.dataset_path.name}' loaded. Structure: {self._structure}. Classes: {self.get_num_classes()}")

    def _detect_structure(self):
        """Detects if the dataset is FLAT or FIXED."""
        train_path = self.dataset_path / "train"
        test_path = self.dataset_path / "test"

        if train_path.is_dir() and test_path.is_dir():
            self._structure = "FIXED"
            logger.debug("Detected FIXED dataset structure.")
        else:
            # Check if root contains class folders directly
            contains_folders = any(d.is_dir() for d in self.dataset_path.iterdir())
            # Basic check: assume FLAT if no train/test and contains subdirs
            if contains_folders and not train_path.exists() and not test_path.exists():
                self._structure = "FLAT"
                logger.debug("Detected FLAT dataset structure.")
            else:
                # Could add more checks, e.g., if files exist directly in root
                raise ValueError(f"Unknown or invalid dataset structure at {self.dataset_path}. "
                                 f"Expected 'FLAT' (root/class*/images) or 'FIXED' (root/train/class*/images, root/test/class*/images).")

    def _load_data(self):
        """Loads image paths and labels based on the detected structure."""
        if self._structure == "FLAT":
            try:
                # Use ImageFolder to easily get paths and labels
                temp_dataset = ImageFolder(self.dataset_path)
                self._classes = temp_dataset.classes
                self._class_to_idx = temp_dataset.class_to_idx
                # ImageFolder stores samples as (path, label_idx)
                self._all_image_paths = [p for p, l in temp_dataset.samples]
                self._all_labels = [l for p, l in temp_dataset.samples]
                logger.info(f"FLAT: Loaded {len(self._all_image_paths)} images across {len(self._classes)} classes.")
            except Exception as e:
                logger.error(f"❌ Failed to load FLAT dataset from {self.dataset_path}: {e}")
                raise
        elif self._structure == "FIXED":
            try:
                train_dataset = ImageFolder(self.dataset_path / "train")
                test_dataset = ImageFolder(self.dataset_path / "test")

                # Ensure classes are consistent
                if train_dataset.classes != test_dataset.classes:
                    raise ValueError("Train and test sets have different classes!")

                self._classes = train_dataset.classes
                self._class_to_idx = train_dataset.class_to_idx

                self._train_image_paths = [p for p, l in train_dataset.samples]
                self._train_labels = [l for p, l in train_dataset.samples]
                self._test_image_paths = [p for p, l in test_dataset.samples]
                self._test_labels = [l for p, l in test_dataset.samples]
                logger.info(
                    f"FIXED: Loaded {len(self._train_image_paths)} train images and {len(self._test_image_paths)} test images.")
            except Exception as e:
                logger.error(f"❌ Failed to load FIXED dataset from {self.dataset_path}: {e}")
                raise
        else:
            # This should not happen if _detect_structure worked correctly
            raise RuntimeError("Dataset structure is not set.")

    def get_dataset_structure(self) -> Optional[str]:
        """Returns the detected dataset structure ('FLAT' or 'FIXED')."""
        return self._structure

    def get_classes(self) -> List[str]:
        """Returns the list of class names."""
        if self._classes is None:
            raise RuntimeError("Dataset not loaded properly, classes are unknown.")
        return self._classes

    def get_class_to_idx(self) -> Dict[str, int]:
        """Returns the mapping from class name to index."""
        if self._class_to_idx is None:
            raise RuntimeError("Dataset not loaded properly, class mapping is unknown.")
        return self._class_to_idx

    def get_num_classes(self) -> int:
        """Returns the number of classes."""
        return len(self.get_classes())

    def get_all_data(self) -> Tuple[List[str], List[int]]:
        """
        Returns all image paths and labels.
        For FIXED structure, this combines train and test sets.
        """
        if self._structure == "FLAT":
            return self._all_image_paths, self._all_labels
        elif self._structure == "FIXED":
            return self._train_image_paths + self._test_image_paths, self._train_labels + self._test_labels
        else:
            raise RuntimeError("Dataset structure not determined.")

    def get_train_val_data(self, val_size: float = 0.2, stratify: bool = True) -> Tuple[
        List[str], List[int], List[str], List[int]]:
        """
        Provides train and validation splits.
        - For FLAT: Splits the entire dataset.
        - For FIXED: Splits the predefined 'train' set.

        Args:
            val_size (float): Proportion of the data to use for validation.
            stratify (bool): Whether to perform stratified splitting.

        Returns:
            Tuple: (train_paths, train_labels, val_paths, val_labels)
        """
        if self._structure == "FLAT":
            source_paths, source_labels = self._all_image_paths, self._all_labels
            if len(source_paths) == 0:
                raise ValueError("No data loaded for FLAT structure.")
            split_stratify = source_labels if stratify else None
        elif self._structure == "FIXED":
            source_paths, source_labels = self._train_image_paths, self._train_labels
            if len(source_paths) == 0:
                raise ValueError("No training data loaded for FIXED structure.")
            split_stratify = source_labels if stratify else None
        else:
            raise RuntimeError("Dataset structure not determined.")

        if val_size <= 0 or val_size >= 1:
            # Return all data as training data, no validation set
            logger.warning(f"⚠️ val_size ({val_size}) is not in (0, 1). Returning all available data as training set.")
            return source_paths, source_labels, [], []

        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                source_paths,
                source_labels,
                test_size=val_size,
                random_state=self.seed,
                stratify=split_stratify
            )
            logger.info(f"Split data: {len(train_paths)} train, {len(val_paths)} validation samples.")
            return train_paths, train_labels, val_paths, val_labels
        except ValueError as e:
            # This can happen if a class has only one sample and stratify=True
            logger.error(
                f"❌ Error during train/val split (potentially due to small class sizes and stratification): {e}. Trying without stratification.")
            if stratify:  # Try again without stratification if it failed
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    source_paths, source_labels, test_size=val_size, random_state=self.seed, stratify=None
                )
                logger.info(
                    f"Split data (non-stratified): {len(train_paths)} train, {len(val_paths)} validation samples.")
                return train_paths, train_labels, val_paths, val_labels
            else:  # If it failed even without stratification, re-raise
                raise e

    def get_test_data(self) -> Optional[Tuple[List[str], List[int]]]:
        """
        Returns the test set paths and labels.
        Returns None if the structure is FLAT (no predefined test set).
        """
        if self._structure == "FLAT":
            logger.warning("⚠️ Requesting test data for FLAT structure. No predefined test set exists.")
            return None
        elif self._structure == "FIXED":
            if not self._test_image_paths:
                logger.warning("⚠️ FIXED structure detected, but no test data found/loaded.")
                return None
            return self._test_image_paths, self._test_labels
        else:
            raise RuntimeError("Dataset structure not determined.")

    def get_transforms(self) -> Tuple[Callable, Callable]:
        """
        Defines and returns standard transformations for train and validation/test.
        """
        # Normalization values typical for ImageNet-pretrained models
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            # Add more augmentations if needed (e.g., ColorJitter, RandomRotation)
            transforms.ToTensor(),
            normalize,
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize(self.image_size),  # Use Resize+CenterCrop or just Resize
            transforms.CenterCrop(self.image_size),
            # transforms.Resize(self.image_size), # Simpler alternative
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, val_test_transform
