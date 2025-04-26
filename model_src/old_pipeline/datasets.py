import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image

from model_src.old_pipeline.data_splitter import create_train_test_split_dataset


class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading images.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Constructor for CustomImageDataset.
        Args:
            image_paths: List of image paths.
            labels: List of labels.
            transform: Image transformations.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Always load as PIL
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_path(self, idx):
        return self.image_paths[idx]


def get_image_path_from_dataset(subset, idx):
    """
    Helper function to get image path from a dataset.
    Args:
        subset: Subset object.
        idx: Index of the image.
    Returns:
        Image path, if the dataset is a Subset or CustomImageDataset.
    """
    if isinstance(subset, Subset):
        if isinstance(subset.dataset, CustomImageDataset):
            return subset.dataset.get_path(subset.indices[idx])
    elif isinstance(subset, CustomImageDataset):
        return subset.get_path(idx)


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.
    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(config: dict, k_folds: int = 1, logger: logging.Logger = None) -> tuple:
    """Load dataset for either train-val-test split or K-fold cross-validation.

    Args:
        config (dict): Configuration dictionary.
        k_folds (int, optional): Number of folds for cross-validation. Default is 1 (indicating no cross-validation, but a train-val-test split).
        logger (logging.Logger, optional): Logger for logging messages.

    Returns:
        (train_loader, val_loader, test_loader, class_names) if k_folds is 1, or
        (k_folds, dataset, indices, class_names), if k_folds > 1
    """
    # Get configurations
    batch_size: int = config.get("batch-size", 16)
    random_seed: int = config.get("random-seed", 42)
    test_size: float = config.get("dataset", {}).get("test-size", 0.2)  # Default 20% test split
    val_size: float = config.get("dataset", {}).get("val-size", 0.2)  # Default 20% validation split

    # Set random seed for reproducibility
    set_random_seed(random_seed)

    # Prepare dataset directory and subdirectories
    dataset_name, train_dir, test_dir = get_dataset_split(config, logger)

    # Collect all image file paths and labels
    train_val_image_paths, train_val_labels, class_names = load_images(train_dir, logger)
    test_image_paths, test_labels, _ = load_images(test_dir)

    # Define transformations
    transform_train_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Create dataset instance
    train_val_dataset = CustomImageDataset(train_val_image_paths, train_val_labels, transform=transform_train_val)
    test_dataset = CustomImageDataset(test_image_paths, test_labels, transform=transform_test)

    if k_folds > 1:
        # K-Fold Cross-Validation Mode
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
        indices = list(skf.split(train_val_image_paths, train_val_labels))

        if logger:
            logger.info(f"Dataset: {dataset_name}")
            logger.info("=" * 30)
            logger.info(f"Number of folds: {k_folds}")
            logger.info("=" * 30)

        return k_folds, train_val_dataset, indices, test_dataset, class_names

    else:
        # Train-Val-Test Split Mode
        train_indices, val_indices = train_test_split(
            range(len(train_val_labels)),
            test_size=val_size / (1 - test_size),  # Adjust train size based on test and val sizes
            stratify=train_val_labels,
            random_state=random_seed
        )

        # Subset data
        train_dataset = Subset(train_val_dataset, train_indices)
        val_dataset = Subset(train_val_dataset, val_indices)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Logging
        if logger:
            logger.info(f"Dataset: {dataset_name}")
            logger.info("=" * 30)
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Validation samples: {len(val_dataset)}")
            logger.info(f"Test samples: {len(test_dataset)}")
            logger.info("=" * 30)

        return train_loader, val_loader, test_loader, class_names


def get_dataset_split(config: dict, logger: logging.Logger = None) -> tuple:
    """
    Get dataset split directories or create a train-test split dataset if not found.
    :param config: configuration dictionary
    :param logger: logger object
    :return: dataset_name, train_dir, test_dir
    """
    # Get dataset directories specified in the configuration
    root_folder: str = config["data-folder"]
    dataset_name: str = config["dataset"]["selected"]
    dataset_dir = os.path.join(root_folder, dataset_name)
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    # Check if train and test directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        if logger: logger.warning(f"Train or test directories not found in {dataset_dir}. Checking for '-split' suffix.")

        # Check if the dataset has a duplicate with a "-split" suffix
        dataset_name = dataset_name + "-split"
        dataset_dir = os.path.join(root_folder, dataset_name)
        train_dir = os.path.join(dataset_dir, "train")
        test_dir = os.path.join(dataset_dir, "test")
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            if logger: logger.warning(f"Train or test directories not found in {dataset_dir}.\n")

            # Create train-test split dataset
            create_train_test_split_dataset(config, logger)
        else:
            if logger: logger.info(f"Found train and test directories in {dataset_dir}.\n")

    return dataset_name, train_dir, test_dir


def load_images(directory: str, logger: logging.Logger = None):
    """Load images from a directory.

    Args:
        directory (str): Directory path.
        logger (logging.Logger, optional): Logger for logging messages.

    Returns:
        (image_paths, labels, class_names)
    """
    # Collect all image file paths and labels
    class_names = sorted(os.listdir(directory))  # Get class names from folders
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

    image_paths = []
    labels = []
    for class_name in class_names:
        class_folder = os.path.join(directory, class_name)
        for img_file in os.listdir(class_folder):
            image_paths.append(os.path.join(class_folder, img_file))
            labels.append(class_to_idx[class_name])

    return image_paths, labels, class_names



