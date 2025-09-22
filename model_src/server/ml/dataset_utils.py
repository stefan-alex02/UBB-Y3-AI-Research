import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import (
    train_test_split
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from torchvision import transforms

from .config import AugmentationStrategy
from .config import RANDOM_SEED, DEFAULT_IMG_SIZE
from ..ml.logger_utils import logger


class DatasetStructure(Enum):
    FLAT = "flat"
    FIXED = "fixed"


class PathImageDataset(Dataset):
    """
    Custom PyTorch Dataset that loads images from a list of paths.

    This dataset handles loading images from file paths during iteration,
    applying transforms at load time, and provides robustness against
    corrupted or missing files through its collate_fn.

    Attributes:
        paths: List of image file paths
        labels: Corresponding integer labels (if any)
        transform: Transform to apply to loaded images
        image_size: Detected image size from transform for fallback purposes
    """
    def __init__(self, paths: List[Union[str, Path]], labels: Optional[List[int]] = None, transform: Optional[Callable] = None):
        """
        Initializes the dataset with paths, labels, and transforms.

        Args:
            paths: List of image file paths
            labels: Corresponding list of integer labels. If None, labels are ignored
            transform: Transform to apply to the images

        Raises:
            ValueError: If paths and labels have different lengths
        """
        if labels is not None and len(paths) != len(labels):
            raise ValueError(f"Paths and labels must have the same length, but got {len(paths)} and {len(labels)}")
        self.paths = [Path(p) for p in paths]
        self.labels = labels
        self.transform = transform
        self.image_size = self._get_image_size_from_transform(transform)

    @staticmethod
    def _get_image_size_from_transform(transform) -> Tuple[int, int]:
        """
        Extracts the target image size from a transform pipeline.

        Searches through the transform pipeline for Resize or RandomResizedCrop
        transforms to determine the expected output image size.

        Args:
            transform: A transform or compose of transforms

        Returns:
            Tuple[int, int]: Height and width of the images after transformation.
                             Defaults to (64, 64) if size cannot be determined.
        """
        if isinstance(transform, transforms.Compose):
            for t in transform.transforms:
                if isinstance(t, (transforms.Resize, transforms.RandomResizedCrop)):
                    size = t.size
                    if isinstance(size, int): return size, size
                    if isinstance(size, (list, tuple)) and len(size) == 2: return tuple(size)
        elif isinstance(transform, (transforms.Resize, transforms.RandomResizedCrop)):
             size = transform.size
             if isinstance(size, int): return size, size
             if isinstance(size, (list, tuple)) and len(size) == 2: return tuple(size)
        logger.debug("Could not determine image size from transform, using default (64, 64) for collate fallback.")
        return 64, 64

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of images in the dataset
        """
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Loads and returns the image and label at the given index.

        Handles potential errors during image loading or transformation
        by returning None values which will be filtered by collate_fn.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            tuple: (transformed_image, label_tensor) if successful,
                   (None, None) if loading or transformation fails
        """
        img_path = self.paths[idx]
        label_val = self.labels[idx] if self.labels is not None else -1 # Use -1 if no labels

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except FileNotFoundError:
             logger.error(f"Image file not found: {img_path}. Returning None.")
             return None, None
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Returning None.")
            return None, None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 logger.warning(f"Error transforming image {img_path}: {e}. Returning None.")
                 return None, None

        label_tensor = torch.tensor(label_val, dtype=torch.long)
        return image, label_tensor

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function that handles None values in batches.

        Filters out any None items from the batch and properly stacks
        the remaining items. This provides robustness against failed
        image loads or transforms.

        Args:
            batch: List of (image, label) tuples from __getitem__

        Returns:
            tuple: (stacked_images, stacked_labels) for valid items

        Raises:
            Exception: If all items in the batch are invalid or stacking fails
        """
        original_batch_size = len(batch)
        batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]

        if not batch:
            logger.warning(f"Collate_fn received empty batch after filtering {original_batch_size} items.")
            return torch.empty((0, 3, 64, 64)), torch.empty(0, dtype=torch.long) # Default 64x64

        try:
            images, labels = zip(*batch)
        except ValueError as e:
            logger.error(f"Error during zip in collate_fn: {e}", exc_info=True)
            raise e

        try:
            images = torch.stack(images, 0)
            labels = torch.stack(labels, 0)
        except Exception as e:
            logger.error(f"Error during torch.stack in collate_fn: {e}", exc_info=True)
            if images: logger.error(f"Shapes of images in failed stack: {[img.shape for img in images]}")
            raise e

        return images, labels


def get_pronounced_augmentations(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a strong augmentation pipeline suitable for images where orientation is less critical.

    This transform pipeline applies significant geometric and color transformations, including:
    - Vertical flips
    - 180-degree rotations
    - Color jittering
    - Gaussian blur

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A composition of transforms for strong augmentation

    Notes:
        Particularly suitable for sky/cloud images or other datasets where the
        orientation of objects is not meaningful for classification.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_moderate_augmentations(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a moderate augmentation pipeline that preserves general orientation.

    This transform pipeline applies controlled geometric and color transformations:
    - Random cropping
    - Horizontal flips only (no vertical flips)
    - Minor rotations (+/-5 degrees)
    - Small affine transformations
    - Moderate color jittering
    - Occasional light blurring

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A composition of transforms for moderate augmentation

    Notes:
        Suitable for images with ground elements or where vertical orientation
        carries important semantic information.
    """
    geometric_transforms = [
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=5)
    ]

    color_intensity_transforms = [
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
    ]

    blur_transform = [
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
    ]

    # Final conversions
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    return transforms.Compose(
        geometric_transforms +
        color_intensity_transforms +
        blur_transform +
        final_transforms
    )


def get_mild_augmentation(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a mild augmentation pipeline.

    This transform pipeline applies:
    - Random resized cropping (scale 0.8-1.0)
    - Horizontal flips
    - Color jittering (brightness, saturation, hue, but no contrast)
    - No rotations

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A mild composition of transforms
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1, contrast=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_paper_replication_augmentation_ccsn(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates an augmentation pipeline replicating the one used in a specific CCSN paper.

    This transform pipeline applies:
    - Random resized cropping (scale 0.8-1.0)
    - Horizontal flips
    - Color jittering (brightness, saturation, hue, but no contrast)
    - No rotations

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A composition of transforms matching the CCSN paper's augmentation

    Notes:
        This is designed to replicate the exact augmentation strategy from a specific
        CCSN paper for comparison or reproduction purposes.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=144),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_paper_replication_augmentation_cloudnet(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates an augmentation pipeline replicating the one used in a specific CloudNet paper.
    This transform pipeline applies:
    - Random resized cropping (scale 0.8-1.0)
    - Horizontal flips
    - No rotations
    - Normalization using ImageNet statistics

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A composition of transforms matching the CloudNet paper's augmentation

    Notes:
        This is designed to replicate the exact augmentation strategy from a specific
        CloudNet paper for comparison or reproduction purposes.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_default_standard_augmentations(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a standard set of image augmentations suitable for general image classification.

    This transform pipeline applies common augmentations:
    - Resizing to target dimensions
    - Horizontal flips
    - Small rotations (+/-10 degrees)
    - Moderate color jittering

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A composition of transforms for standard augmentation

    Notes:
        This is the default augmentation pipeline that works well for most image
        classification tasks without making strong assumptions about the dataset.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_no_augmentation_transform(img_size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a basic preprocessing pipeline without data augmentation.

    This transform pipeline only applies:
    - Resizing to target dimensions
    - Conversion to tensor
    - Normalization using ImageNet statistics

    Args:
        img_size: Target image size as (height, width)

    Returns:
        transforms.Compose: A composition of transforms for basic preprocessing

    Notes:
        This is used both for evaluation and when explicitly specifying no
        augmentation during training. It's identical to the eval_transform.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ImageDatasetHandler:
    """
    Handles loading and managing image datasets for deep learning applications.

    This class provides functionality to work with image datasets in different structures:
    - FLAT: All classes are directly under the root directory
    - FIXED: Dataset has predefined train/test splits in separate directories

    It handles loading image paths and labels, detecting dataset structure,
    applying appropriate transforms, and providing access to different data splits.

    Attributes:
        root_path: Path to the root directory of the dataset
        img_size: Target size for image resizing
        structure: Detected dataset structure (FLAT or FIXED)
        classes: List of class names
        class_to_idx: Dictionary mapping class names to indices
        num_classes: Number of classes in the dataset
    """
    def __init__(self,
                 root_path: Union[str, Path],
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2,
                 augmentation_strategy: Union[str, AugmentationStrategy, Callable, None] = AugmentationStrategy.DEFAULT_STANDARD,
                 use_offline_augmented_data: bool = False,
                 force_flat_for_fixed_cv: bool = False):
        """
        Initializes the ImageDatasetHandler with the specified parameters.

        Args:
            root_path: Path to the dataset root directory
            img_size: Target size for image resizing, as (height, width)
            val_split_ratio: Proportion of data to use for validation (0.0 to 1.0)
            test_split_ratio_if_flat: Proportion of data to use for testing if structure is FLAT
            augmentation_strategy: Strategy for data augmentation, can be:
                                   - A string matching an AugmentationStrategy enum value
                                   - An AugmentationStrategy enum value
                                   - A callable transform function
                                   - None (defaults to NO_AUGMENTATION)
            use_offline_augmented_data: Whether to load additional offline augmented images
            force_flat_for_fixed_cv: If True, combines train and test sets for cross-validation
                                    even when dataset has a FIXED structure

        Raises:
            FileNotFoundError: If the dataset root path doesn't exist
            ValueError: If split ratios are invalid or classes can't be determined
        """
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
        self.force_flat_for_fixed_cv = force_flat_for_fixed_cv
        self.use_offline_augmented_data = use_offline_augmented_data

        # Augmentation_strategy
        if isinstance(augmentation_strategy, str):
            try:
                self.augmentation_strategy_enum = AugmentationStrategy(augmentation_strategy.lower())
            except ValueError:
                logger.warning(
                    f"Invalid augmentation_strategy string: '{augmentation_strategy}'. Defaulting to NO_AUGMENTATION.")
                self.augmentation_strategy_enum = AugmentationStrategy.NO_AUGMENTATION
        elif isinstance(augmentation_strategy, AugmentationStrategy):
            self.augmentation_strategy_enum = augmentation_strategy
        elif callable(augmentation_strategy):
            self.augmentation_strategy_enum = None
            self.custom_train_transform = augmentation_strategy
        elif augmentation_strategy is None:
            self.augmentation_strategy_enum = AugmentationStrategy.NO_AUGMENTATION
        else:
            logger.warning(
                f"Invalid augmentation_strategy type: {type(augmentation_strategy)}. Defaulting to NO_AUGMENTATION.")
            self.augmentation_strategy_enum = AugmentationStrategy.NO_AUGMENTATION

        logger.info(
            f"Using augmentation strategy: {str(self.augmentation_strategy_enum) if self.augmentation_strategy_enum else 'Custom Transform'}")

        # Transforms
        self.eval_transform = get_no_augmentation_transform(self.img_size)

        if hasattr(self, 'custom_train_transform'):
            self.train_transform = self.custom_train_transform
        elif self.augmentation_strategy_enum == AugmentationStrategy.SKY_ONLY_ROTATION:
            self.train_transform = get_pronounced_augmentations(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.CCSN_MODERATE:
            self.train_transform = get_moderate_augmentations(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.SWIMCAT_MILD:
            self.train_transform = get_mild_augmentation(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.PAPER_CLOUDNET:
            self.train_transform = get_paper_replication_augmentation_cloudnet(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.PAPER_CCSN:
            self.train_transform = get_paper_replication_augmentation_ccsn(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.DEFAULT_STANDARD:
            self.train_transform = get_default_standard_augmentations(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.NO_AUGMENTATION:
            self.train_transform = self.eval_transform
        else:
            logger.warning("Unknown augmentation strategy, defaulting to no augmentation.")
            self.train_transform = self.eval_transform

        self.structure = self._detect_structure()
        logger.info(f"Detected dataset structure: {self.structure.value}")
        if self.structure == DatasetStructure.FIXED and self.force_flat_for_fixed_cv:
             logger.warning("Dataset is FIXED, but force_flat_for_fixed_cv=True. "
                            "CV methods will treat train+test as a single dataset. "
                            "Results might not reflect standard fixed-test evaluation.")

        # Load paths and labels
        self._all_paths: List[Path] = []
        self._all_labels: List[int] = []
        self._train_val_paths: List[Path] = []
        self._train_val_labels: List[int] = []
        self._offline_aug_paths: List[Path] = []
        self._offline_aug_labels: List[int] = []
        self._offline_aug_original_basenames: List[str] = []
        self._test_paths: List[Path] = []
        self._test_labels: List[int] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        # Class weights for handling imbalance
        self.class_weights: Optional[torch.Tensor] = None

        self._load_paths_and_labels()
        if not self.classes:
             raise ValueError(f"Could not determine classes for dataset at {self.root_path}")
        self.num_classes = len(self.classes)
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.classes)}")

        if self.classes and self._train_val_labels:
            self._calculate_class_weights()

        # Dataset sizes
        logger.info(f"Dataset sizes: {len(self._train_val_paths)} train+val, {len(self._test_paths)} test. Total: {len(self._all_paths)}")
        if self.structure == DatasetStructure.FIXED and self.force_flat_for_fixed_cv:
            logger.info(f"Total combined size (for forced CV): {len(self._all_paths)}")


    def _detect_structure(self) -> DatasetStructure:
        """
        Detects the structure of the dataset (FLAT or FIXED).

        A dataset is considered FIXED if it has 'train' and 'test' subdirectories
        with matching class subdirectories inside each.
        Otherwise, it's considered FLAT.

        Returns:
            DatasetStructure: The detected structure (FLAT or FIXED)

        Raises:
            RuntimeError: If there's an error accessing the directory
        """
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

    @staticmethod
    def _scan_dir_for_paths_labels(target_dir: Path) -> Tuple[List[Path], List[int], List[str], Dict[str, int]]:
        """
        Scans a directory for image files organized in class subdirectories.

        This method implements a basic ImageFolder-like functionality, where:
        - The target directory contains class subdirectories
        - Each class subdirectory contains image files belonging to that class
        - Only files with common image extensions (.jpg, .jpeg, .png, .bmp, .tif, .tiff) are included

        Args:
            target_dir: Path to the directory to scan

        Returns:
            Tuple containing:
            - List[Path]: Image file paths
            - List[int]: Corresponding integer labels
            - List[str]: Class names (sorted alphabetically)
            - Dict[str, int]: Mapping from class names to indices

        Notes:
            - Empty directories are handled gracefully
            - Classes are sorted alphabetically to ensure consistent class indices
            - If target_dir is not a directory, returns empty lists and dictionary
        """
        paths = []
        labels = []
        target_dir = Path(target_dir)
        if not target_dir.is_dir(): return [], [], [], {}

        class_names = sorted([d.name for d in target_dir.iterdir() if d.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        for class_name, class_idx in class_to_idx.items():
            class_dir = target_dir / class_name
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    paths.append(img_path)
                    labels.append(class_idx)
        return paths, labels, class_names, class_to_idx

    def _calculate_class_weights(self):
        """Calculates class weights for the training data to handle imbalance."""
        logger.info("Calculating class weights for the training set...")

        if not self._train_val_labels:
            logger.warning("Cannot calculate class weights: original training label list is empty.")
            return

        try:
            weights = compute_class_weight(
                class_weight='balanced',
                classes=np.arange(self.num_classes),
                y=self._train_val_labels
            )

            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            logger.info(f"Calculated class weights (for classes 0 to {self.num_classes - 1}):")
            for i, class_name in enumerate(self.classes):
                logger.info(f"  - {class_name} (Class {i}): {self.class_weights[i]:.4f}")

        except Exception as e:
            logger.error(f"Failed to compute class weights: {e}", exc_info=True)
            self.class_weights = None

    @staticmethod
    def _get_original_basename(augmented_path: Path) -> Optional[str]:
        """
        Parses an augmented filename like 'original_image_name_augmented.png'
        to extract 'original_image_name'.
        Returns None if parsing fails.
        """
        match = re.match(r"^(.*?)_generated(\..*)?$", augmented_path.name)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _scan_augmented_dir_for_paths_labels_and_originals(
            target_dir: Path,
            master_class_names: List[str],
            master_class_to_idx: Dict[str, int]
    ) -> Tuple[List[Path], List[int], List[str]]:
        """
        Scans a directory of augmented images and maps them to their original images.

        This method is designed for offline data augmentation workflows where:
        - Augmented images are stored in a separate directory structure
        - Augmented images follow a naming pattern indicating their original source
        - The class structure matches the original dataset

        Args:
            target_dir: Path to the directory containing augmented images
            master_class_names: List of class names from the original dataset
            master_class_to_idx: Mapping from class names to indices from original dataset

        Returns:
            Tuple containing:
            - List[Path]: Paths to augmented image files
            - List[int]: Corresponding class labels (using original dataset indices)
            - List[str]: Original image basenames that were augmented

        Notes:
            - Only files with common image extensions are processed
            - Augmented images must have names like "original_name_generated.ext"
            - Classes not in master_class_names are skipped with a warning
            - Files that don't match the expected naming pattern are skipped
        """
        aug_paths = []
        aug_labels = []
        aug_original_basenames = []

        target_dir = Path(target_dir)
        if not target_dir.is_dir(): return [], [], []

        found_class_subdirs = sorted([d.name for d in target_dir.iterdir() if d.is_dir()])

        for class_name_in_aug in found_class_subdirs:
            if class_name_in_aug not in master_class_to_idx:
                logger.warning(f"Class '{class_name_in_aug}' in augmented set not in master. Skipping.")
                continue

            class_idx = master_class_to_idx[class_name_in_aug]
            class_dir = target_dir / class_name_in_aug
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    original_basename = ImageDatasetHandler._get_original_basename(img_path)
                    if original_basename:
                        aug_paths.append(img_path)
                        aug_labels.append(class_idx)
                        aug_original_basenames.append(original_basename)
                    else:
                        logger.warning(
                            f"Could not determine original basename for augmented file {img_path}. Skipping.")
        return aug_paths, aug_labels, aug_original_basenames

    def _load_paths_and_labels(self) -> None:
        """
        Loads image paths and labels based on the detected dataset structure.

        For FLAT structure:
        - Scans all class directories under root_path
        - Optionally creates a train/test split

        For FIXED structure:
        - Scans 'train' and 'test' directories separately
        - Handles class mismatches between train and test

        If enabled, also loads offline augmented data.

        Raises:
            ValueError: If no images are found or classes can't be determined
        """
        if self.structure == DatasetStructure.FLAT:
            logger.info(f"Scanning FLAT dataset from {self.root_path}...")
            all_paths, all_labels, classes, class_to_idx = self._scan_dir_for_paths_labels(self.root_path)
            if not all_paths: raise ValueError(f"No images found in FLAT dataset at {self.root_path}")
            self.classes = classes
            self.class_to_idx = class_to_idx
            self._all_paths = all_paths
            self._all_labels = np.array(all_labels)

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
                 self._train_val_labels = self._all_labels[train_val_indices].tolist()
                 self._test_paths = [self._all_paths[i] for i in test_indices]
                 self._test_labels = self._all_labels[test_indices].tolist()
                 self._all_labels = self._all_labels.tolist()
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
                 test_labels = [test_c2i.get(test_classes[lbl], -1) for lbl in test_labels]
                 final_labels_test = []
                 test_idx_to_name = {v:k for k,v in test_c2i.items()}
                 for label_idx in test_labels:
                      class_name = test_idx_to_name.get(label_idx)
                      final_idx = train_c2i.get(class_name, -1)
                      if final_idx == -1: logger.warning(f"Class {class_name} from test set not found in train set.")
                      final_labels_test.append(final_idx)
                 test_labels = final_labels_test


            self.classes = train_classes
            self.class_to_idx = train_c2i
            self._train_val_paths = train_paths
            self._train_val_labels = train_labels
            self._test_paths = test_paths
            self._test_labels = test_labels

            if self.force_flat_for_fixed_cv:
                self._all_paths = self._train_val_paths + self._test_paths
                self._all_labels = self._train_val_labels + self._test_labels

        # Load OFFLINE AUGMENTED data
        if self.use_offline_augmented_data:
            original_dataset_name = self.root_path.name
            augmented_dataset_name = f"{original_dataset_name}_augmented"
            augmented_dataset_path = self.root_path.parent / augmented_dataset_name

            if augmented_dataset_path.is_dir():
                if not self.class_to_idx:
                    logger.error("Cannot load augmented data: Main dataset classes not determined.")
                else:
                    self._offline_aug_paths, self._offline_aug_labels, self._offline_aug_original_basenames = \
                        self._scan_augmented_dir_for_paths_labels_and_originals(
                            augmented_dataset_path, self.classes, self.class_to_idx
                        )
                    if self._offline_aug_paths:
                        logger.info(
                            f"Loaded {len(self._offline_aug_paths)} offline augmented samples with original name mapping.")
                    else:
                        logger.warning(f"Augmented dataset directory {augmented_dataset_path} is empty.")
            else:
                logger.warning(f"Offline augmented dataset directory not found: {augmented_dataset_path}.")

        logger.info(f"Final Original Dataset sizes: "
                    f"{len(self._train_val_paths)} original train+val, "
                    f"{len(self._test_paths)} original test. "
                    f"Offline augmented samples loaded: {len(self._offline_aug_paths)}.")

    def get_class_weights(self) -> Optional[torch.Tensor]:
        return self.class_weights

    def get_train_val_paths_labels(self) -> Tuple[List[Path], List[int]]:
        """
        Returns paths and labels for the training and validation set.

        Returns:
            Tuple[List[Path], List[int]]: A tuple containing:
                - List of image file paths
                - List of corresponding integer labels
        """
        return self._train_val_paths, self._train_val_labels

    def get_offline_augmented_paths_labels_with_originals(self) -> Tuple[List[Path], List[int], List[str]]:
        """
        Returns paths, labels, and original basenames for the offline augmented dataset.

        Returns:
            Tuple[List[Path], List[int], List[str]]: A tuple containing:
                - List of augmented image file paths
                - List of corresponding integer labels
                - List of original image basenames that were augmented
        """
        return self._offline_aug_paths, self._offline_aug_labels, self._offline_aug_original_basenames

    def get_test_paths_labels(self) -> Tuple[List[Path], List[int]]:
        """
        Returns paths and labels for the test set.

        Returns:
            Tuple[List[Path], List[int]]: A tuple containing:
                - List of test image file paths
                - List of corresponding integer labels

        Notes:
            - For FLAT structure, may return empty lists if no test split was created
            - For FIXED structure, may return empty lists if the test directory was empty
        """
        if self.structure == DatasetStructure.FLAT and not self._test_paths:
             logger.warning("Requesting test paths/labels for FLAT structure, but no test split was created.")
        elif self.structure == DatasetStructure.FIXED and not self._test_paths:
             logger.warning("Requesting test paths/labels for FIXED structure, but test dir was empty.")
        return self._test_paths, self._test_labels

    def get_full_paths_labels_for_cv(self) -> Tuple[List[Path], List[int]]:
        """
        Returns paths and labels for the entire dataset for cross-validation.

        For FLAT structure, returns all paths and labels.
        For FIXED structure, only returns combined data if force_flat_for_fixed_cv=True.

        Returns:
            Tuple[List[Path], List[int]]: A tuple containing:
                - List of all image file paths for CV
                - List of corresponding integer labels

        Raises:
            ValueError: If trying to get full dataset for FIXED structure without force_flat_for_fixed_cv=True
        """
        if self.structure == DatasetStructure.FLAT:
            return self._all_paths, self._all_labels
        elif self.structure == DatasetStructure.FIXED:
            if self.force_flat_for_fixed_cv:
                 logger.debug("Providing combined train+test paths/labels for forced flat CV.")
                 return self._all_paths, self._all_labels
            else:
                 raise ValueError("Cannot get 'full' dataset for FIXED structure unless force_flat_for_fixed_cv=True.")
        else:
             raise RuntimeError(f"Unknown dataset structure {self.structure}")

    def get_classes(self) -> List[str]:
        """
        Returns the list of class names in the dataset.

        Returns:
            List[str]: List of class names
        """
        return self.classes

    def get_class_to_idx(self) -> Dict[str, int]:
        """
        Returns the mapping from class names to class indices.

        Returns:
            Dict[str, int]: Dictionary mapping class names to indices
        """
        return self.class_to_idx

    def get_train_transform(self) -> Callable:
        """
        Returns the transform pipeline for training data.

        The transform depends on the configured augmentation strategy.

        Returns:
            Callable: A torchvision.transforms.Compose object for training data
        """
        return self.train_transform

    def get_eval_transform(self) -> Callable:
        """
        Returns the transform pipeline for evaluation data.

        This transform only includes basic preprocessing (resize, normalize)
        without augmentation.

        Returns:
            Callable: A torchvision.transforms.Compose object for evaluation data
        """
        return self.eval_transform
