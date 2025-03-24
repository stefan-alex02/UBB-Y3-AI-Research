import os
import shutil
import logging
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_train_test_split_dataset(config: dict, logger: logging.Logger = None):
    dataset_path = Path(config["data-folder"]) / config["dataset"]["selected"]
    test_size = config["dataset"].get("test-size", 0.2)  # Default 20% test split
    output_path = dataset_path.parent / f"{dataset_path.name}-split"

    if logger: logger.info(f"Creating train-test split dataset at {output_path} with test_size={test_size}")

    # Ensure the output folder is empty
    if output_path.exists():
        shutil.rmtree(output_path)  # Delete existing folder to avoid conflicts
        if logger: logger.warning(f"Output folder already exists. Deleting existing folder.")
    output_path.mkdir(parents=True)

    # Create train and test subdirectories
    train_path = output_path / "train"
    test_path = output_path / "test"
    train_path.mkdir()
    test_path.mkdir()

    # Iterate over each class folder in the dataset
    for class_folder in dataset_path.iterdir():
        if not class_folder.is_dir():
            continue  # Ignore non-directory files

        class_name = class_folder.name
        images = list(class_folder.glob("*"))  # Collect all images

        if not images:
            if logger: logger.warning(f"Skipping empty class folder: {class_folder}")
            continue

        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)

        # Create class folders in train and test directories
        (train_path / class_name).mkdir()
        (test_path / class_name).mkdir()

        # Move images to respective train/test folders
        for img in train_images:
            shutil.copy(img, train_path / class_name / img.name)
        for img in test_images:
            shutil.copy(img, test_path / class_name / img.name)

        if logger: logger.info(f"Class '{class_name}': {len(train_images)} train, {len(test_images)} test")

    if logger: logger.info("Train-test split completed successfully!\n")
