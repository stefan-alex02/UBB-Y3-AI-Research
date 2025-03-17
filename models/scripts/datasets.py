import logging
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def load_datasets(config: dict, logger: logging.Logger = None):
    # Get required configurations
    dataset_name: str = config["dataset"]["selected"]
    root_folder: str = config["root-data-folder"]
    batch_size: int = config["batch-size"]

    # Prepare other variables
    dataset_dir = os.path.join(root_folder, dataset_name)

    # Specify the dataset directories
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    # Define train dataloader with online data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Define validation dataloader
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Define test dataloader
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load the datasets
    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = ImageFolder(root=val_dir, transform=transform_val)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Obtain Class Names
    class_names = train_dataset.classes

    # Log the details (if logger is provided)
    if logger:
        logger.info(f"Dataset: {dataset_name}")
        logger.info("=" * 30)

        # Show number of training samples per class
        train_class_counts = {class_names[i]: 0 for i in range(len(class_names))}
        for _, label in train_dataset:
            train_class_counts[class_names[label]] += 1
        logger.info("Training class counts:")
        for class_name, count in train_class_counts.items():
            logger.info(f"{class_name}: {count}")
        logger.info("=" * 30)

        # Display the number of training samples
        logger.info(f"Number of training samples: {len(train_dataset)}\n")

    return train_loader, val_loader, test_loader, class_names