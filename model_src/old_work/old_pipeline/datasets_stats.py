import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from collections import Counter

from model_src.old.old_pipeline.datasets import get_image_path_from_dataset


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

    # Display class distributions
    show_datasets_class_distributions(class_names, datasets)

    # Unpack data
    train_dataset, val_dataset, test_dataset = datasets.values()

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

    # Calculate sizes and percentages
    sizes = [len(d) for d in datasets.values()]
    total_size = sum(sizes)
    percentages = [size / total_size * 100 for size in sizes]
    labels = datasets.keys()

    # Plot dataset split distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, sizes, color=['blue', 'green', 'red'])
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Samples")
    plt.title("Dataset Split Distribution")

    # Add percentages on top of the bars
    for bar, percent, size in zip(bars, percentages, sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{percent:.1f}% ({size} instances)', ha='center', va='bottom')

    plt.show()


def show_datasets_class_distributions(class_names, datasets: dict):
    """
    Plots the class distribution across different data.
    :param class_names: List of class names.
    :param datasets: Dictionary containing dataset splits.
    """
    # Get class counts
    def get_class_counts(dataset):
        if isinstance(dataset, Subset) and hasattr(dataset.dataset, "labels"):
            labels = [dataset.dataset.labels[i] for i in dataset.indices]
        elif hasattr(dataset, "labels"):  # Custom dataset case
            labels = dataset.labels
        else:
            labels = [label for _, label in dataset]  # General fallback
        return Counter(labels)

    class_indices = np.arange(len(class_names))
    width = 0.25  # Width of each bar
    plt.figure(figsize=(8, 6))

    [plt.bar(class_indices - width + ci * width,
             [counts.get(i, 0) for i in range(len(class_names))],
             width=width,
             label=label,
             color=color)
    for ci, (counts, label, color) in enumerate(zip(map(lambda d: get_class_counts(d), datasets.values()),
                                                    datasets.keys(),
                                                    ["blue", "green", "red"]))]

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