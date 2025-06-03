import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import torch
from PIL import Image  # Needed for PathImageDataset
from sklearn.model_selection import (
    train_test_split
)
from torch.utils.data import Dataset
from torchvision import transforms

from .config import RANDOM_SEED, DEFAULT_IMG_SIZE
from ..ml.logger_utils import logger

from .config import AugmentationStrategy


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
                if isinstance(t, (transforms.Resize, transforms.RandomResizedCrop)):
                    size = t.size
                    if isinstance(size, int): return size, size
                    if isinstance(size, (list, tuple)) and len(size) == 2: return tuple(size)
        elif isinstance(transform, (transforms.Resize, transforms.RandomResizedCrop)): # Handle direct Resize
             size = transform.size
             if isinstance(size, int): return size, size
             if isinstance(size, (list, tuple)) and len(size) == 2: return tuple(size)
        # Fallback size
        logger.debug("Could not determine image size from transform, using default (64, 64) for collate fallback.")
        return 64, 64

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
            return torch.empty((0, 3, 64, 64)), torch.empty(0, dtype=torch.long) # Default 64x64

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


def get_pronounced_augmentations(img_size: Tuple[int, int]) -> transforms.Compose:
    """Augmentations suitable for sky/cloud images where orientation is less critical."""
    return transforms.Compose([
        transforms.Resize(img_size),  # Or transforms.Resize(256) -> transforms.RandomCrop(img_size)
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # <<< Suitable for sky-only
        transforms.RandomRotation(degrees=180),  # <<< Full rotation possible
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_moderate_augmentations(img_size: Tuple[int, int]) -> transforms.Compose:
    """Augmentations for images with ground elements; avoids vertical flips and significant rotations."""
    geometric_transforms = [
        transforms.Resize(img_size),  # Or a slightly larger size then RandomCrop
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


def get_default_standard_augmentations(img_size: Tuple[int, int]) -> transforms.Compose:
    """Your previous standard augmentation set."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_no_augmentation_transform(img_size: Tuple[int, int]) -> transforms.Compose:
    """Only basic preprocessing, no augmentation. Same as eval_transform."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_paper_replication_augmentation_gcd(img_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # Combines zoom and crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15), # Paper says "random rotations", let's pick a moderate value
        # Paper explicitly avoided contrast jitter
        transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1, contrast=0.0), # Example values, no contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_paper_replication_augmentation_ccsn(img_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # Combines zoom and crop
        transforms.RandomHorizontalFlip(p=0.5),
        # Paper: "random rotations to not be appropriate for this data set" for CCSN
        # Your GROUND_AWARE_NO_ROTATION had small rotation and affine, this removes them fully for closer replication.
        transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1, contrast=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ImageDatasetHandler:
    """
    Handles loading image paths and labels from disk, detecting structure,
    and providing access to paths/labels for different splits. Also manages transforms.
    """
    def __init__(self,
                 root_path: Union[str, Path],
                 img_size: Tuple[int, int] = DEFAULT_IMG_SIZE,
                 val_split_ratio: float = 0.2,
                 test_split_ratio_if_flat: float = 0.2, # Ratio for test split if structure is FLAT
                 augmentation_strategy: Union[str, AugmentationStrategy, Callable, None] = AugmentationStrategy.DEFAULT_STANDARD,
                 use_offline_augmented_data: bool = False,
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
        self.force_flat_for_fixed_cv = force_flat_for_fixed_cv
        self.use_offline_augmented_data = use_offline_augmented_data

        # --- Process augmentation_strategy ---
        if isinstance(augmentation_strategy, str):
            try:
                self.augmentation_strategy_enum = AugmentationStrategy(augmentation_strategy.lower())
            except ValueError:
                logger.warning(
                    f"Invalid augmentation_strategy string: '{augmentation_strategy}'. Defaulting to NO_AUGMENTATION.")
                self.augmentation_strategy_enum = AugmentationStrategy.NO_AUGMENTATION
        elif isinstance(augmentation_strategy, AugmentationStrategy):
            self.augmentation_strategy_enum = augmentation_strategy
        elif callable(augmentation_strategy):  # User passed a custom transform function
            self.augmentation_strategy_enum = None  # Mark that it's custom
            self.custom_train_transform = augmentation_strategy
        elif augmentation_strategy is None:  # Explicitly no augmentation
            self.augmentation_strategy_enum = AugmentationStrategy.NO_AUGMENTATION
        else:
            logger.warning(
                f"Invalid augmentation_strategy type: {type(augmentation_strategy)}. Defaulting to NO_AUGMENTATION.")
            self.augmentation_strategy_enum = AugmentationStrategy.NO_AUGMENTATION

        logger.info(
            f"Using augmentation strategy: {str(self.augmentation_strategy_enum) if self.augmentation_strategy_enum else 'Custom Transform'}")

        # --- Transforms ---
        self.eval_transform = get_no_augmentation_transform(self.img_size)  # Eval transform is fixed

        if hasattr(self, 'custom_train_transform'):
            self.train_transform = self.custom_train_transform
        elif self.augmentation_strategy_enum == AugmentationStrategy.SKY_ONLY_ROTATION:
            self.train_transform = get_pronounced_augmentations(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.CCSN_MODERATE:
            self.train_transform = get_moderate_augmentations(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.PAPER_GCD:
            self.train_transform = get_paper_replication_augmentation_gcd(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.PAPER_CCSN:
            self.train_transform = get_paper_replication_augmentation_ccsn(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.DEFAULT_STANDARD:
            self.train_transform = get_default_standard_augmentations(self.img_size)
        elif self.augmentation_strategy_enum == AugmentationStrategy.NO_AUGMENTATION:
            self.train_transform = self.eval_transform  # Same as eval
        else:  # Fallback, should not be reached if logic above is correct
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
        self._offline_aug_original_basenames: List[str] = []  # <<< NEW: Store original basenames
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

    @staticmethod
    def _get_original_basename(augmented_path: Path) -> Optional[str]:
        """
        Parses an augmented filename like 'original_image_name_augmented.png'
        to extract 'original_image_name'.
        Returns None if parsing fails.
        """
        # Define the suffix pattern more robustly
        # It looks for "_augmented" followed by an optional dot and extension.
        match = re.match(r"^(.*?)_generated(\..*)?$", augmented_path.name)
        if match:
            return match.group(1)  # The part before "_augmented"
        # Fallback or stricter parsing if needed
        # logger.warning(f"Could not parse original basename from augmented file: {augmented_path.name}")
        return None

    @staticmethod
    def _scan_augmented_dir_for_paths_labels_and_originals(
            target_dir: Path,
            master_class_names: List[str],
            master_class_to_idx: Dict[str, int]
    ) -> Tuple[List[Path], List[int], List[str]]:  # Returns paths, labels, original_basenames
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

        # 2. Load OFFLINE AUGMENTED data (if requested)
        if self.use_offline_augmented_data:
            original_dataset_name = self.root_path.name
            augmented_dataset_name = f"{original_dataset_name}_augmented"
            augmented_dataset_path = self.root_path.parent / augmented_dataset_name

            if augmented_dataset_path.is_dir():
                if not self.class_to_idx:  # Should have been set by now
                    logger.error("Cannot load augmented data: Main dataset classes not determined.")
                else:
                    # Use the new scanning function
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

    # --- Public Accessors ---
    def get_train_val_paths_labels(self) -> Tuple[List[Path], List[int]]:
        """Returns paths and labels for the training + validation set."""
        return self._train_val_paths, self._train_val_labels

    def get_offline_augmented_paths_labels_with_originals(self) -> Tuple[List[Path], List[int], List[str]]:
        """
        Returns paths, labels, and corresponding original basenames for the OFFLINE AUGMENTED dataset.
        """
        return self._offline_aug_paths, self._offline_aug_labels, self._offline_aug_original_basenames

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
