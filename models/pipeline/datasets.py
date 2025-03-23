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


class CustomImageDataset(Dataset):
    """
    Custom dataset class for loading images.
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom dataset class for loading
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
    root_folder: str = config["data-folder"]
    dataset_name: str = config["dataset"]["selected"]
    batch_size: int = config["batch-size"]
    random_seed: int = config.get("random-seed", 42)
    test_size: float = config.get("test-size", 0.2)  # Default 20% test split
    val_size: float = config.get("val-size", 0.2)  # Default 20% validation split

    # Set random seed for reproducibility
    set_random_seed(random_seed)

    # Prepare dataset directory and subdirectories
    dataset_dir = os.path.join(root_folder, dataset_name)
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

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
            test_size=val_size / (1 - test_size),
            stratify=train_val_labels,
            random_state=random_seed
        )

        # Subset datasets
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


def plot_datasets_statistics(dataset_info, fold=None):
    """
    Plots dataset distribution statistics.

    Args:
        dataset_info: The output of `load_datasets()`, containing dataloaders (for train-val-test mode) or dataset & indices (for cross-validation mode).
        fold (int, optional): None if it is a train-val-test split, or the fold index if it is a cross-validation mode.
    """
    datasets, class_names = pack_datasets(dataset_info, fold)

    # Display split distribution
    show_datasets_split_distribution(datasets)

    # Unpack datasets
    train_dataset, val_dataset, test_dataset = datasets.values()

    # Display class distributions
    show_datasets_class_distributions(class_names, test_dataset, train_dataset, val_dataset)

    # Display sample images
    show_sample_images(train_dataset, class_names, k=5, title="Sampled Train Images")
    show_sample_images(test_dataset, class_names, k=5, title="Sampled Test Images")


def pack_datasets(dataset_info, fold=None):
    if fold is None:
        train_loader, val_loader, test_loader, class_names = dataset_info
        datasets = {
            "Train": train_loader.dataset,
            "Validation": val_loader.dataset,
            "Test": test_loader.dataset
        }
    else:
        k_folds, dataset, indices, test_dataset, class_names = dataset_info

        if not (0 <= fold < k_folds):
            raise ValueError(f"Invalid fold index. Must be in range [0, {k_folds}).")

        train_indices, val_indices = indices[fold]
        datasets = {
            f"Train (Fold {fold})": Subset(dataset, train_indices),
            f"Validation (Fold {fold})": Subset(dataset, val_indices),
            "Test": test_dataset
        }

    return datasets, class_names


def show_datasets_split_distribution(datasets):
    """
    Plots the dataset split distribution.
    :param datasets: Dictionary containing dataset splits.
    """

    # Plot dataset split distribution
    plt.figure(figsize=(8, 5))
    sizes = [len(d) for d in datasets.values()]
    labels = list(datasets.keys())
    plt.bar(labels, sizes, color=['blue', 'green', 'red'])
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Samples")
    plt.title("Dataset Split Distribution")
    plt.show()


def show_datasets_class_distributions(class_names, test_dataset, train_dataset, val_dataset):
    # Get class counts
    def get_class_counts(dataset):
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "labels"):
            labels = [dataset.dataset.labels[i] for i in dataset.indices]
        elif hasattr(dataset, "labels"):  # Custom dataset case
            labels = dataset.labels
        else:
            labels = [label for _, label in dataset]  # General fallback
        return Counter(labels)

    train_counts = get_class_counts(train_dataset)
    val_counts = get_class_counts(val_dataset)
    test_counts = get_class_counts(test_dataset)
    class_indices = np.arange(len(class_names))
    width = 0.25  # Width of each bar
    plt.figure(figsize=(8, 6))
    plt.bar(class_indices - width, [train_counts.get(i, 0) for i in range(len(class_names))], width=width,
            label="Train", color="blue")
    plt.bar(class_indices, [val_counts.get(i, 0) for i in range(len(class_names))], width=width, label="Validation",
            color="green")
    plt.bar(class_indices + width, [test_counts.get(i, 0) for i in range(len(class_names))], width=width,
            label="Test",
            color="red")
    plt.xticks(class_indices, class_names, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution Across Train, Validation, and Test Sets")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_sample_images(dataset, class_names, k=3, title="Sampled Images"):
    """
    Displays `k` sampled images from a dataset, ensuring images are taken from distinct parts.

    Args:
        dataset: The dataset to sample from.
        class_names: List of class names.
        k (int): Number of images to sample.
        title (str): Title for the plot.
    """
    num_samples = len(dataset)
    if k > num_samples:
        k = num_samples  # Avoid errors if dataset is small

    # Select k evenly spaced indices
    indices = np.linspace(0, num_samples - 1, k, dtype=int)

    plt.figure(figsize=(k * 3, 3))

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        name = get_image_path_from_dataset(dataset, idx).split("\\")[-1]

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()  # Convert tensor to image

        plt.subplot(1, k, i + 1)
        plt.imshow(image)
        plt.title(f"Class: {class_names[label]}\n({name})")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
