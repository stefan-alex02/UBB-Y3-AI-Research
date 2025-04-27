import os
import logging
import emoji
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from enum import Enum
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, datasets, models
from torchvision.transforms import v2

from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_validate, cross_val_score, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, Checkpoint

# Global configurations
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Configure logging
class CustomFormatter(logging.Formatter):
    """Custom formatter to include emojis and consistent formatting"""

    def __init__(self):
        super().__init__('%(asctime)s [%(levelname)s] %(message)s',
                         datefmt='%Y-%m-%d %H:%M:%S')

    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = emoji.emojize(":information: ") + record.msg
        elif record.levelno == logging.WARNING:
            record.msg = emoji.emojize(":warning: ") + record.msg
        elif record.levelno == logging.ERROR:
            record.msg = emoji.emojize(":red_exclamation_mark: ") + record.msg
        elif record.levelno == logging.DEBUG:
            record.msg = emoji.emojize(":magnifying_glass_tilted_left: ") + record.msg
        return super().format(record)


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Set up logger with custom formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = CustomFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Create logger instance
logger = setup_logger('image_classification', 'logs/classification.log')


# Dataset Structure Enum
class DatasetStructure(Enum):
    """Enum to represent the structure of the dataset"""
    FLAT = "flat"
    FIXED = "fixed"


class ImageDatasetHandler:
    """
    Handler for image datasets with different structures.

    Attributes:
        root_path (str): Path to the dataset root directory
        structure (DatasetStructure): Structure of the dataset (FLAT or FIXED)
        classes (List[str]): List of class names
        num_classes (int): Number of classes
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset (if available)
        test_dataset (Dataset): Test dataset
    """

    def __init__(self,
                 root_path: str,
                 img_size: Tuple[int, int] = (224, 224),
                 val_split: float = 0.2,
                 data_augmentation: bool = True):
        """
        Initialize the dataset handler.

        Args:
            root_path: Path to the dataset root directory
            img_size: Size to resize images to (height, width)
            val_split: Fraction of training data to use for validation (only for FLAT structure)
            data_augmentation: Whether to apply data augmentation to training data
        """
        self.root_path = root_path
        self.img_size = img_size
        self.val_split = val_split
        self.data_augmentation = data_augmentation

        # Detect dataset structure
        self.structure = self._detect_structure()
        logger.info(f"Detected dataset structure: {self.structure.value}")

        # Setup transforms
        self.train_transform = self._setup_train_transform() if data_augmentation else self._setup_eval_transform()
        self.eval_transform = self._setup_eval_transform()

        # Load dataset based on structure
        self._load_dataset()

        # Get class information
        self.classes = self._get_classes()
        self.num_classes = len(self.classes)
        logger.info(f"Found {self.num_classes} classes: {', '.join(self.classes)}")

    def _detect_structure(self) -> DatasetStructure:
        """
        Detect the structure of the dataset.

        Returns:
            DatasetStructure: FLAT or FIXED
        """
        root_subdirs = [d for d in os.listdir(self.root_path)
                        if os.path.isdir(os.path.join(self.root_path, d))]

        if 'train' in root_subdirs and 'test' in root_subdirs:
            # Check if train and test have the same class structure
            train_subdirs = [d for d in os.listdir(os.path.join(self.root_path, 'train'))
                             if os.path.isdir(os.path.join(self.root_path, 'train', d))]
            test_subdirs = [d for d in os.listdir(os.path.join(self.root_path, 'test'))
                            if os.path.isdir(os.path.join(self.root_path, 'test', d))]

            if set(train_subdirs) == set(test_subdirs):
                return DatasetStructure.FIXED

        # Default to FLAT structure if not FIXED
        return DatasetStructure.FLAT

    def _setup_train_transform(self) -> transforms.Compose:
        """
        Set up data augmentation transforms for training data.

        Returns:
            transforms.Compose: Composition of transforms
        """
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _setup_eval_transform(self) -> transforms.Compose:
        """
        Set up transforms for evaluation data (no augmentation).

        Returns:
            transforms.Compose: Composition of transforms
        """
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_dataset(self) -> None:
        """Load datasets based on the detected structure."""
        if self.structure == DatasetStructure.FLAT:
            self._load_flat_dataset()
        else:  # FIXED
            self._load_fixed_dataset()

    def _load_flat_dataset(self) -> None:
        """Load dataset with FLAT structure and create train/val/test splits."""
        logger.info("Loading FLAT structure dataset")

        # Load the entire dataset with eval transform (we'll apply train transform during training)
        full_dataset = datasets.ImageFolder(self.root_path, transform=self.eval_transform)

        # Create stratified splits
        targets = np.array(full_dataset.targets)
        train_indices, test_indices = next(iter(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(
                np.zeros(len(targets)), targets)
        ))

        # Further split train into train and validation
        train_targets = targets[train_indices]
        train_train_indices, train_val_indices = next(iter(
            StratifiedKFold(n_splits=int(1 / self.val_split), shuffle=True, random_state=RANDOM_SEED).split(
                np.zeros(len(train_targets)), train_targets)
        ))

        # Create final indices
        final_train_indices = train_indices[train_train_indices]
        final_val_indices = train_indices[train_val_indices]
        final_test_indices = test_indices

        # Create subsets
        self.full_dataset = full_dataset
        self.train_dataset = Subset(full_dataset, final_train_indices)
        self.val_dataset = Subset(full_dataset, final_val_indices)
        self.test_dataset = Subset(full_dataset, final_test_indices)

        # Apply different transforms to train dataset
        self.train_dataset_augmented = TransformedSubset(
            full_dataset, final_train_indices, transform=self.train_transform
        )

        logger.info(f"Dataset split: {len(self.train_dataset)} train, "
                    f"{len(self.val_dataset)} validation, "
                    f"{len(self.test_dataset)} test samples")

    def _load_fixed_dataset(self) -> None:
        """Load dataset with FIXED structure (predefined train/test splits)."""
        logger.info("Loading FIXED structure dataset")

        # Load train and test datasets
        train_full = datasets.ImageFolder(
            os.path.join(self.root_path, 'train'),
            transform=self.eval_transform
        )

        # Split train into train and validation
        targets = np.array(train_full.targets)
        train_indices, val_indices = next(iter(
            StratifiedKFold(n_splits=int(1 / self.val_split), shuffle=True, random_state=RANDOM_SEED).split(
                np.zeros(len(targets)), targets)
        ))

        # Create subsets
        self.full_dataset = None  # Not applicable for FIXED structure
        self.train_dataset = Subset(train_full, train_indices)
        self.val_dataset = Subset(train_full, val_indices)

        # Apply different transforms to train dataset
        self.train_dataset_augmented = TransformedSubset(
            train_full, train_indices, transform=self.train_transform
        )

        # Load test dataset
        self.test_dataset = datasets.ImageFolder(
            os.path.join(self.root_path, 'test'),
            transform=self.eval_transform
        )

        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, "
                    f"{len(self.val_dataset)} validation, "
                    f"{len(self.test_dataset)} test samples")

    def _get_classes(self) -> List[str]:
        """
        Get the list of class names.

        Returns:
            List[str]: List of class names
        """
        if self.structure == DatasetStructure.FLAT:
            return self.full_dataset.classes
        else:  # FIXED
            # Both train and test should have the same classes
            train_path = os.path.join(self.root_path, 'train')
            return [d for d in os.listdir(train_path)
                    if os.path.isdir(os.path.join(train_path, d))]

    def get_train_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Get DataLoader for training data.

        Args:
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader: DataLoader for training data
        """
        return DataLoader(
            self.train_dataset_augmented if self.data_augmentation else self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def get_val_dataloader(self, batch_size: int = 32) -> DataLoader:
        """
        Get DataLoader for validation data.

        Args:
            batch_size: Batch size for DataLoader

        Returns:
            DataLoader: DataLoader for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def get_test_dataloader(self, batch_size: int = 32) -> DataLoader:
        """
        Get DataLoader for test data.

        Args:
            batch_size: Batch size for DataLoader

        Returns:
            DataLoader: DataLoader for test data
        """
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

    def get_full_dataset(self) -> Dataset:
        """
        Get the full dataset (only available for FLAT structure).

        Returns:
            Dataset: Full dataset or None if FIXED structure

        Raises:
            ValueError: If called on a FIXED structure dataset
        """
        if self.structure == DatasetStructure.FLAT:
            return self.full_dataset
        else:
            raise ValueError("Full dataset is not available for FIXED structure datasets")


class TransformedSubset(Dataset):
    """
    A dataset that applies a transform to a subset of another dataset.

    Attributes:
        dataset (Dataset): The original dataset
        indices (List[int]): Indices of the subset
        transform (Optional[Callable]): Transform to apply
    """

    def __init__(self,
                 dataset: Dataset,
                 indices: List[int],
                 transform: Optional[Callable] = None):
        """
        Initialize the transformed subset.

        Args:
            dataset: The original dataset
            indices: Indices of the subset
            transform: Transform to apply (overrides the original dataset's transform)
        """
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple: (transformed_image, label)
        """
        image, label = self.dataset.samples[self.indices[idx]]
        image = self.dataset.loader(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.indices)


# Model definitions
class SimpleCNN(nn.Module):
    """
    A simple CNN model for image classification.

    Attributes:
        num_classes (int): Number of output classes
    """

    def __init__(self, num_classes: int):
        """
        Initialize the CNN model.

        Args:
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SimpleViT(nn.Module):
    """
    A simplified Vision Transformer model using the pre-trained ViT from torchvision.

    Attributes:
        num_classes (int): Number of output classes
    """

    def __init__(self, num_classes: int):
        """
        Initialize the Vision Transformer model.

        Args:
            num_classes: Number of output classes
        """
        super(SimpleViT, self).__init__()
        # Use a pre-trained ViT model and modify the head
        self.model = models.vit_b_16(pretrained=True)

        # Replace the head
        self.model.heads = nn.Linear(self.model.hidden_dim, num_classes)

        # Freeze some layers to speed up training
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.model(x)


class DiffusionClassifier(nn.Module):
    """
    A simplified diffusion-based classifier.
    This is a placeholder implementation for demonstration purposes.

    Attributes:
        num_classes (int): Number of output classes
    """

    def __init__(self, num_classes: int):
        """
        Initialize the diffusion classifier model.

        Args:
            num_classes: Number of output classes
        """
        super(DiffusionClassifier, self).__init__()
        # Using ResNet50 as backbone for the diffusion classifier
        self.backbone = models.resnet50(pretrained=True)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 512)

        # Additional layers to simulate a diffusion-based approach
        self.diffusion_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.backbone(x)
        x = self.diffusion_head(x)
        return x


# Model wrapper with skorch
class SkorchModelAdapter(NeuralNetClassifier):
    """
    Adapter for PyTorch models using skorch to make them compatible with scikit-learn.

    Attributes:
        module (nn.Module): PyTorch model
        criterion: Loss function
        optimizer: Optimizer class
        train_split: Function to split data into train/validation
        callbacks: List of callbacks
        max_epochs: Maximum number of epochs
        batch_size: Batch size
        device: Device to use for computation
    """

    def __init__(self,
                 model_class: Any = None,
                 model_kwargs: Dict[str, Any] = None,
                 criterion: Any = nn.CrossEntropyLoss,
                 optimizer: Any = torch.optim.Adam,
                 optimizer_kwargs: Dict[str, Any] = None,
                 max_epochs: int = 10,
                 batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 train_split: Optional[Callable] = skorch.dataset.ValidSplit(0.2),
                 **kwargs):
        """
        Initialize the adapter.

        Args:
            model_class: PyTorch model class
            model_kwargs: Keyword arguments for the model
            criterion: Loss function
            optimizer: Optimizer class
            optimizer_kwargs: Keyword arguments for the optimizer
            max_epochs: Maximum number of epochs
            batch_size: Batch size
            device: Device to use for computation
            train_split: Function to split data into train/validation
            **kwargs: Additional keyword arguments for NeuralNetClassifier
        """
        # Store these as instance attributes
        self.model_class = model_class
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        # Set model name
        self.model_name = model_class.__name__ if model_class is not None else "Unknown"

        # Handle optimizer kwargs
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': 0.001, 'weight_decay': 0.0001}
        self.optimizer_kwargs = optimizer_kwargs

        # Setup callbacks
        callbacks = [
            ('early_stopping', EarlyStopping(patience=5)),
            ('checkpoint', Checkpoint(monitor='valid_loss_best', f_params='best_model.pt')),
            ('lr_scheduler', LRScheduler(
                policy='ReduceLROnPlateau',
                monitor='valid_loss',
                mode='min',
                patience=3,
                factor=0.5)
             )
        ]

        # Build parameters for parent class without duplicate module
        init_kwargs = {}

        # Only set module if model_class is provided directly (not during clone)
        if 'module' not in kwargs and model_class is not None:
            init_kwargs['module'] = model_class
            if 'num_classes' in self.model_kwargs:
                init_kwargs['module__num_classes'] = self.model_kwargs['num_classes']

        # Add other parameters
        init_kwargs.update({
            'criterion': criterion,
            'optimizer': optimizer,
            'optimizer__lr': optimizer_kwargs.get('lr', 0.001),
            'optimizer__weight_decay': optimizer_kwargs.get('weight_decay', 0.0001),
            'max_epochs': max_epochs,
            'batch_size': batch_size,
            'device': device,
            'train_split': train_split,
            'callbacks': callbacks,
        })

        # Add any other kwargs
        init_kwargs.update(kwargs)

        # Initialize parent class
        super(SkorchModelAdapter, self).__init__(**init_kwargs)

    def fit(self, X: Any, y: Any, **kwargs) -> 'SkorchModelAdapter':
        """
        Fit the model to the data.
        """
        logger.info(f"Training {self.model_name} model")
        return super(SkorchModelAdapter, self).fit(X, y, **kwargs)

    def transform(self, X: Any) -> np.ndarray:
        """
        Transform the data (get probabilities).
        """
        return self.predict_proba(X)


class ClassificationPipeline:
    """
    Pipeline for image classification tasks.

    Attributes:
        dataset_handler (ImageDatasetHandler): Handler for the dataset
        model_adapter (SkorchModelAdapter): Adapter for the model
        results_dir (str): Directory to save results
    """

    def __init__(self,
                 dataset_path: str,
                 model_type: str = 'cnn',
                 model_kwargs: Dict[str, Any] = None,
                 img_size: Tuple[int, int] = (224, 224),
                 results_dir: str = 'results',
                 val_split: float = 0.2,
                 data_augmentation: bool = True):
        """
        Initialize the pipeline.

        Args:
            dataset_path: Path to the dataset
            model_type: Type of model ('cnn', 'vit', or 'diffusion')
            model_kwargs: Keyword arguments for the model
            img_size: Size to resize images to
            results_dir: Directory to save results
            val_split: Fraction of training data to use for validation
            data_augmentation: Whether to apply data augmentation to training data
        """
        # Initialize dataset handler
        self.dataset_handler = ImageDatasetHandler(
            root_path=dataset_path,
            img_size=img_size,
            val_split=val_split,
            data_augmentation=data_augmentation
        )

        # Set up results directory
        self.results_dir = os.path.join(
            results_dir,
            os.path.basename(dataset_path),
            model_type
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Set default model kwargs if not provided
        if model_kwargs is None:
            model_kwargs = {'num_classes': self.dataset_handler.num_classes}
        else:
            model_kwargs['num_classes'] = self.dataset_handler.num_classes

        # Select model based on type
        model_class = self._get_model_class(model_type)

        # Initialize model adapter
        self.model_adapter = SkorchModelAdapter(
            model_class=model_class,
            model_kwargs=model_kwargs
        )

        self.model_type = model_type
        logger.info(f"Pipeline initialized with {model_type} model and {self.dataset_handler.structure.value} dataset")

    def _get_model_class(self, model_type: str) -> Any:
        """
        Get the model class based on the type.

        Args:
            model_type: Type of model ('cnn', 'vit', or 'diffusion')

        Returns:
            Any: Model class

        Raises:
            ValueError: If the model type is not supported
        """
        if model_type.lower() == 'cnn':
            return SimpleCNN
        elif model_type.lower() == 'vit':
            return SimpleViT
        elif model_type.lower() == 'diffusion':
            return DiffusionClassifier
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _prepare_data_for_sklearn(self,
                                  dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for scikit-learn functions.

        Args:
            dataset: PyTorch dataset

        Returns:
            Tuple[np.ndarray, np.ndarray]: X and y arrays
        """
        # For SimpleDataset, just convert to numpy arrays
        if isinstance(dataset, datasets.ImageFolder):
            images = []
            labels = []
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

            with torch.no_grad():
                for X, y in dataloader:
                    images.append(X.numpy())
                    labels.append(y.numpy())

            X = np.vstack(images)
            y = np.concatenate(labels)

            return X, y

        # For Subset, extract the original dataset and indices
        elif isinstance(dataset, Subset) or isinstance(dataset, TransformedSubset):
            if isinstance(dataset, Subset):
                indices = dataset.indices
                original_dataset = dataset.dataset
            else:  # TransformedSubset
                indices = dataset.indices
                original_dataset = dataset.dataset

            # Extract images and labels from the original dataset
            images = []
            labels = []
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

            with torch.no_grad():
                for X, y in dataloader:
                    images.append(X.numpy())
                    labels.append(y.numpy())

            X = np.vstack(images)
            y = np.concatenate(labels)

            return X, y

        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def _compute_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_score: np.ndarray) -> Dict[str, Any]:
        """
        Compute evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_score: Prediction scores (probabilities)

        Returns:
            Dict[str, Any]: Dictionary of metrics
        """
        # Compute per-class metrics
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        # Initialize metrics dictionary
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'per_class': {}
        }

        # Compute one-vs-rest metrics for each class
        for i, class_label in enumerate(unique_classes):
            # One-vs-rest for current class
            y_true_bin = (y_true == class_label).astype(int)
            y_pred_bin = (y_pred == class_label).astype(int)
            y_score_bin = y_score[:, i] if y_score.ndim > 1 else y_score

            # Compute metrics
            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            specificity = recall_score(1 - y_true_bin, 1 - y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

            # Compute ROC AUC if possible
            try:
                roc_auc = roc_auc_score(y_true_bin, y_score_bin)
            except:
                roc_auc = np.nan

            # Compute PR AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin, y_score_bin)
            pr_auc = auc(recall_curve, precision_curve) if len(np.unique(y_true_bin)) > 1 else np.nan

            # Store metrics for this class
            class_name = self.dataset_handler.classes[class_label] if hasattr(self.dataset_handler,
                                                                              'classes') else f"Class {class_label}"
            metrics['per_class'][class_name] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }

        # Compute macro-averaged metrics
        metrics['macro_avg'] = {
            'precision': np.mean([metrics['per_class'][c]['precision'] for c in metrics['per_class']]),
            'recall': np.mean([metrics['per_class'][c]['recall'] for c in metrics['per_class']]),
            'specificity': np.mean([metrics['per_class'][c]['specificity'] for c in metrics['per_class']]),
            'f1': np.mean([metrics['per_class'][c]['f1'] for c in metrics['per_class']]),
            'roc_auc': np.mean([metrics['per_class'][c]['roc_auc'] for c in metrics['per_class'] if
                                not np.isnan(metrics['per_class'][c]['roc_auc'])]),
            'pr_auc': np.mean([metrics['per_class'][c]['pr_auc'] for c in metrics['per_class'] if
                               not np.isnan(metrics['per_class'][c]['pr_auc'])])
        }

        return metrics

    def _save_results(self, metrics: Dict[str, Any], method_name: str, **kwargs) -> None:
        """
        Save evaluation results to file.

        Args:
            metrics: Dictionary of metrics
            method_name: Name of the method used
            **kwargs: Additional parameters to include in the filename
        """
        # Create results filename based on parameters
        params_str = '_'.join([f"{k}={v}" for k, v in kwargs.items() if k != 'self'])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{method_name}_{params_str}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        # Save metrics to file
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)

        logger.info(f"Results saved to {filepath}")

        # Also save a summary to CSV for easy comparison
        summary = {
            'method': method_name,
            'timestamp': timestamp,
            'accuracy': metrics['accuracy'],
            'macro_avg_precision': metrics['macro_avg']['precision'],
            'macro_avg_recall': metrics['macro_avg']['recall'],
            'macro_avg_f1': metrics['macro_avg']['f1'],
            'macro_avg_roc_auc': metrics['macro_avg']['roc_auc'],
            'macro_avg_pr_auc': metrics['macro_avg']['pr_auc']
        }

        # Add parameters
        for k, v in kwargs.items():
            if k != 'self':
                summary[k] = v

        # Create or append to CSV file
        csv_path = os.path.join(self.results_dir, 'summary.csv')
        df = pd.DataFrame([summary])

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

    def non_nested_grid_search(self,
                               param_grid: Dict[str, List],
                               cv: int = 5,
                               n_iter: int = 10,
                               method: str = 'grid',
                               save_results: bool = True) -> Dict[str, Any]:
        """
        Perform non-nested grid search for hyperparameter tuning.

        Args:
            param_grid: Grid of parameters to search
            cv: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
            method: Method to use ('grid' for GridSearchCV or 'random' for RandomizedSearchCV)
            save_results: Whether to save results to file

        Returns:
            Dict[str, Any]: Dictionary of results

        Raises:
            ValueError: If the dataset structure is not compatible with the method
        """
        logger.info(f"Performing non-nested {method} search with {cv}-fold cross-validation")

        # Prepare data
        X_train, y_train = self._prepare_data_for_sklearn(self.dataset_handler.train_dataset_augmented)
        X_val, y_val = self._prepare_data_for_sklearn(self.dataset_handler.val_dataset)

        # Combine train and validation data
        X = np.vstack([X_train, X_val])
        y = np.concatenate([y_train, y_val])

        # Create cross-validator
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

        # Create search object
        if method.lower() == 'grid':
            search = GridSearchCV(
                estimator=self.model_adapter,
                param_grid=param_grid,
                cv=skf,
                scoring='accuracy',
                n_jobs=1,  # Neural nets usually don't work well with n_jobs > 1
                verbose=1,
                return_train_score=True
            )
        elif method.lower() == 'random':
            search = RandomizedSearchCV(
                estimator=self.model_adapter,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=skf,
                scoring='accuracy',
                n_jobs=1,
                verbose=1,
                random_state=RANDOM_SEED,
                return_train_score=True
            )
        else:
            raise ValueError(f"Unsupported search method: {method}")

        # Fit search
        search.fit(X, y)

        # Create results dictionary
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': search.cv_results_['mean_train_score'].tolist(),
                'std_train_score': search.cv_results_['std_train_score'].tolist(),
            }
        }

        # Update model with best params
        self.model_adapter = search.best_estimator_

        # Evaluate on test set
        X_test, y_test = self._prepare_data_for_sklearn(self.dataset_handler.test_dataset)
        y_pred = search.predict(X_test)
        y_score = search.predict_proba(X_test)

        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_score)
        results['test_metrics'] = metrics

        # Save results
        if save_results:
            self._save_results(
                results,
                f"non_nested_{method}_search",
                cv=cv,
                n_iter=n_iter if method.lower() == 'random' else len(param_grid)
            )

        logger.info(f"Non-nested {method} search completed")
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")

        return results

    def nested_grid_search(self,
                           param_grid: Dict[str, List],
                           outer_cv: int = 5,
                           inner_cv: int = 3,
                           n_iter: int = 10,
                           method: str = 'grid',
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Perform nested grid search for unbiased performance estimation.

        Args:
            param_grid: Grid of parameters to search
            outer_cv: Number of outer cross-validation folds
            inner_cv: Number of inner cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
            method: Method to use ('grid' for GridSearchCV or 'random' for RandomizedSearchCV)
            save_results: Whether to save results to file

        Returns:
            Dict[str, Any]: Dictionary of results

        Raises:
            ValueError: If the dataset structure is not compatible with the method
        """
        logger.info(f"Performing nested {method} search with {outer_cv}-fold outer CV and {inner_cv}-fold inner CV")

        # Check dataset compatibility
        if self.dataset_handler.structure == DatasetStructure.FIXED:
            logger.warning("Using nested CV with FIXED dataset structure might not be appropriate")
            # We'll use train+val for inner CV and test for final evaluation
            X, y = self._prepare_data_for_sklearn(self.dataset_handler.train_dataset_augmented)
            X_val, y_val = self._prepare_data_for_sklearn(self.dataset_handler.val_dataset)
            X = np.vstack([X, X_val])
            y = np.concatenate([y, y_val])
            X_test, y_test = self._prepare_data_for_sklearn(self.dataset_handler.test_dataset)
        else:
            # Use full dataset for nested CV
            try:
                X, y = self._prepare_data_for_sklearn(self.dataset_handler.full_dataset)
                X_test, y_test = None, None  # Will be determined by outer CV
            except ValueError:
                logger.error("Full dataset is not available. Using combined train and val datasets.")
                X_train, y_train = self._prepare_data_for_sklearn(self.dataset_handler.train_dataset_augmented)
                X_val, y_val = self._prepare_data_for_sklearn(self.dataset_handler.val_dataset)
                X_test, y_test = self._prepare_data_for_sklearn(self.dataset_handler.test_dataset)
                X = np.vstack([X_train, X_val])
                y = np.concatenate([y_train, y_val])

        # Create cross-validators
        outer_cv_obj = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=RANDOM_SEED)
        inner_cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_SEED)

        # Create base estimator
        base_estimator = clone(self.model_adapter)

        # Create inner search object
        if method.lower() == 'grid':
            inner_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=inner_cv_obj,
                scoring='accuracy',
                n_jobs=1,
                verbose=0
            )
        elif method.lower() == 'random':
            inner_search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=inner_cv_obj,
                scoring='accuracy',
                n_jobs=1,
                verbose=0,
                random_state=RANDOM_SEED
            )
        else:
            raise ValueError(f"Unsupported search method: {method}")

        # Perform nested cross-validation
        if X_test is None:  # Use outer CV to generate test sets
            cv_results = cross_validate(
                inner_search,
                X, y,
                cv=outer_cv_obj,
                scoring={
                    'accuracy': 'accuracy',
                    'precision_macro': 'precision_macro',
                    'recall_macro': 'recall_macro',
                    'f1_macro': 'f1_macro'
                },
                return_estimator=True,
                n_jobs=1,
                verbose=1
            )

            # Get best parameters from each outer fold
            best_params_list = [est.best_params_ for est in cv_results['estimator']]

            # Create results dictionary
            results = {
                'cv_results': {
                    'test_accuracy': cv_results['test_accuracy'].tolist(),
                    'test_precision_macro': cv_results['test_precision_macro'].tolist(),
                    'test_recall_macro': cv_results['test_recall_macro'].tolist(),
                    'test_f1_macro': cv_results['test_f1_macro'].tolist(),
                },
                'best_params_per_fold': best_params_list,
                'mean_test_accuracy': np.mean(cv_results['test_accuracy']),
                'std_test_accuracy': np.std(cv_results['test_accuracy']),
                'mean_test_precision_macro': np.mean(cv_results['test_precision_macro']),
                'std_test_precision_macro': np.std(cv_results['test_precision_macro']),
                'mean_test_recall_macro': np.mean(cv_results['test_recall_macro']),
                'std_test_recall_macro': np.std(cv_results['test_recall_macro']),
                'mean_test_f1_macro': np.mean(cv_results['test_f1_macro']),
                'std_test_f1_macro': np.std(cv_results['test_f1_macro']),
            }
        else:  # Use fixed test set
            # Perform inner CV to find best parameters
            inner_search.fit(X, y)

            # Get predictions on test set
            y_pred = inner_search.predict(X_test)
            y_score = inner_search.predict_proba(X_test)

            # Compute metrics
            metrics = self._compute_metrics(y_test, y_pred, y_score)

            # Create results dictionary
            results = {
                'best_params': inner_search.best_params_,
                'best_score': inner_search.best_score_,
                'test_metrics': metrics
            }

            # Update model with best params
            self.model_adapter = inner_search.best_estimator_

        # Save results
        if save_results:
            self._save_results(
                results,
                f"nested_{method}_search",
                outer_cv=outer_cv,
                inner_cv=inner_cv,
                n_iter=n_iter if method.lower() == 'random' else len(param_grid)
            )

        logger.info(f"Nested {method} search completed")
        if X_test is None:
            logger.info(
                f"Mean CV accuracy: {results['mean_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}")
        else:
            logger.info(f"Best parameters: {results['best_params']}")
            logger.info(f"Best CV score: {results['best_score']:.4f}")
            logger.info(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")

        return results

    def cv_model_evaluation(self,
                            cv: int = 5,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation for model evaluation.

        Args:
            cv: Number of cross-validation folds
            save_results: Whether to save results to file

        Returns:
            Dict[str, Any]: Dictionary of results

        Raises:
            ValueError: If the dataset structure is not compatible with the method
        """
        logger.info(f"Performing CV model evaluation with {cv}-fold cross-validation")

        # Check dataset compatibility
        if self.dataset_handler.structure == DatasetStructure.FIXED:
            raise ValueError("CV model evaluation is not compatible with FIXED dataset structure")

        # Use full dataset for CV
        try:
            X, y = self._prepare_data_for_sklearn(self.dataset_handler.full_dataset)
        except ValueError:
            logger.error("Full dataset is not available for CV evaluation")
            raise

        # Create cross-validator
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

        # Perform cross-validation
        cv_results = cross_validate(
            self.model_adapter,
            X, y,
            cv=skf,
            scoring={
                'accuracy': 'accuracy',
                'precision_macro': 'precision_macro',
                'recall_macro': 'recall_macro',
                'f1_macro': 'f1_macro'
            },
            return_estimator=True,
            n_jobs=1,
            verbose=1
        )

        # Create results dictionary
        results = {
            'cv_results': {
                'test_accuracy': cv_results['test_accuracy'].tolist(),
                'test_precision_macro': cv_results['test_precision_macro'].tolist(),
                'test_recall_macro': cv_results['test_recall_macro'].tolist(),
                'test_f1_macro': cv_results['test_f1_macro'].tolist(),
            },
            'mean_test_accuracy': np.mean(cv_results['test_accuracy']),
            'std_test_accuracy': np.std(cv_results['test_accuracy']),
            'mean_test_precision_macro': np.mean(cv_results['test_precision_macro']),
            'std_test_precision_macro': np.std(cv_results['test_precision_macro']),
            'mean_test_recall_macro': np.mean(cv_results['test_recall_macro']),
            'std_test_recall_macro': np.std(cv_results['test_recall_macro']),
            'mean_test_f1_macro': np.mean(cv_results['test_f1_macro']),
            'std_test_f1_macro': np.std(cv_results['test_f1_macro']),
        }

        # Save results
        if save_results:
            self._save_results(results, "cv_model_evaluation", cv=cv)

        logger.info(f"CV model evaluation completed")
        logger.info(f"Mean CV accuracy: {results['mean_test_accuracy']:.4f} ± {results['std_test_accuracy']:.4f}")

        return results

    def single_train(self,
                     max_epochs: int = 30,
                     early_stopping: bool = True,
                     save_model: bool = True) -> Dict[str, Any]:
        """
        Perform a single training run.

        Args:
            max_epochs: Maximum number of epochs
            early_stopping: Whether to use early stopping
            save_model: Whether to save the trained model

        Returns:
            Dict[str, Any]: Dictionary of results
        """
        logger.info(f"Performing single training run with max_epochs={max_epochs}")

        # Update model parameters
        self.model_adapter.set_params(max_epochs=max_epochs)

        # Prepare data
        X_train, y_train = self._prepare_data_for_sklearn(self.dataset_handler.train_dataset_augmented)

        # Train model
        self.model_adapter.fit(X_train, y_train)

        # Get training history
        history = {
            'train_loss': self.model_adapter.history[:, 'train_loss'],
            'valid_loss': self.model_adapter.history[:, 'valid_loss'],
            'epoch': self.model_adapter.history[:, 'epoch'],
            'dur': self.model_adapter.history[:, 'dur'],
        }

        # Create results dictionary
        results = {
            'history': history,
            'best_epoch': int(self.model_adapter.history[-1, 'epoch']),
            'best_valid_loss': float(self.model_adapter.history[-1, 'valid_loss']),
            'train_loss': float(self.model_adapter.history[-1, 'train_loss']),
        }

        # Save model
        if save_model:
            model_filename = f"{self.model_type}_e{results['best_epoch']}_vl{results['best_valid_loss']:.4f}.pt"
            model_path = os.path.join(self.results_dir, model_filename)
            torch.save(self.model_adapter.module_.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            results['model_path'] = model_path

        logger.info(f"Single training completed")
        logger.info(f"Best epoch: {results['best_epoch']}")
        logger.info(f"Best validation loss: {results['best_valid_loss']:.4f}")

        return results

    def single_eval(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.

        Args:
            save_results: Whether to save results to file

        Returns:
            Dict[str, Any]: Dictionary of results
        """
        logger.info("Performing model evaluation on test set")

        # Prepare data
        X_test, y_test = self._prepare_data_for_sklearn(self.dataset_handler.test_dataset)

        # Make predictions
        y_pred = self.model_adapter.predict(X_test)
        y_score = self.model_adapter.predict_proba(X_test)

        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_score)

        # Save results
        if save_results:
            self._save_results(metrics, "single_eval")

        logger.info(f"Model evaluation completed")
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro avg precision: {metrics['macro_avg']['precision']:.4f}")
        logger.info(f"Macro avg recall: {metrics['macro_avg']['recall']:.4f}")
        logger.info(f"Macro avg F1: {metrics['macro_avg']['f1']:.4f}")

        return metrics

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from file.

        Args:
            model_path: Path to the model file
        """
        logger.info(f"Loading model from {model_path}")

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.model_adapter.device)

        # Set to model
        self.model_adapter.initialize()
        self.model_adapter.module_.load_state_dict(state_dict)

        logger.info("Model loaded successfully")

class PipelineExecutor:
    """
    Executor for running multiple pipeline methods in sequence.

    Attributes:
        pipeline (ClassificationPipeline): Classification pipeline
        methods (List[str]): List of methods to run
        params (Dict[str, Dict]): Parameters for each method
    """

    def __init__(self,
                 dataset_path: str,
                 model_type: str = 'cnn',
                 model_kwargs: Dict[str, Any] = None,
                 results_dir: str = 'results',
                 methods: List[str] = None,
                 params: Dict[str, Dict] = None):
        """
        Initialize the pipeline executor.

        Args:
            dataset_path: Path to the dataset
            model_type: Type of model ('cnn', 'vit', or 'diffusion')
            model_kwargs: Keyword arguments for the model
            results_dir: Directory to save results
            methods: List of methods to run
            params: Parameters for each method
        """
        # Initialize pipeline
        self.pipeline = ClassificationPipeline(
            dataset_path=dataset_path,
            model_type=model_type,
            model_kwargs=model_kwargs,
            results_dir=results_dir
        )

        # Set methods and parameters
        self.methods = methods if methods is not None else []
        self.params = params if params is not None else {}

        # Validate methods
        self._validate_methods()

        logger.info(f"Pipeline executor initialized with methods: {', '.join(self.methods)}")

    def _validate_methods(self) -> None:
        """
        Validate that the specified methods are compatible with the dataset structure.

        Raises:
            ValueError: If a method is not valid or not compatible with the dataset structure
        """
        valid_methods = [
            'non_nested_grid_search',
            'nested_grid_search',
            'cv_model_evaluation',
            'single_train',
            'single_eval'
        ]

        # Check that all methods are valid
        for method in self.methods:
            if method not in valid_methods:
                raise ValueError(f"Invalid method: {method}")

        # Check compatibility with dataset structure
        if self.pipeline.dataset_handler.structure == DatasetStructure.FIXED:
            if 'cv_model_evaluation' in self.methods:
                raise ValueError("CV model evaluation is not compatible with FIXED dataset structure")

    def run(self) -> Dict[str, Any]:
        """
        Run the specified methods in sequence.

        Returns:
            Dict[str, Any]: Dictionary of results from each method
        """
        results = {}

        # Run each method
        for method in self.methods:
            logger.info(f"Running method: {method}")

            # Get method parameters
            params = self.params.get(method, {})

            # Run method
            if method == 'non_nested_grid_search':
                results[method] = self.pipeline.non_nested_grid_search(**params)
            elif method == 'nested_grid_search':
                results[method] = self.pipeline.nested_grid_search(**params)
            elif method == 'cv_model_evaluation':
                results[method] = self.pipeline.cv_model_evaluation(**params)
            elif method == 'single_train':
                results[method] = self.pipeline.single_train(**params)
            elif method == 'single_eval':
                results[method] = self.pipeline.single_eval(**params)

        return results

# Example usage
if __name__ == "__main__":
    # Example configuration
    dataset_path = "data/mini-GCD"
    model_type = "cnn"  # or "vit" or "diffusion"

    # Parameter grid for grid search
    # param_grid = {
    #     'lr': [0.001, 0.0001],
    #     'max_epochs': [10, 20],
    #     'batch_size': [16, 32],
    # }

    param_grid = {
        'lr': [0.001, 0.0001],
        'max_epochs': [10],
        'batch_size': [16],
    }

    # Create executor with methods and parameters
    executor = PipelineExecutor(
        dataset_path=dataset_path,
        model_type=model_type,
        methods=[
            'single_train',
            'single_eval',
            'non_nested_grid_search'
        ],
        params={
            'single_train': {
                'max_epochs': 30,
                'early_stopping': True,
                'save_model': True
            },
            'non_nested_grid_search': {
                'param_grid': param_grid,
                'cv': 3,
                'method': 'grid'
            }
        }
    )

    # Run the pipeline
    results = executor.run()