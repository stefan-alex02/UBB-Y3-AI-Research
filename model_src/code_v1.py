import os
import logging
from enum import Enum, auto
from typing import Tuple, List, Optional, Dict, Any, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms, datasets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetStructure(Enum):
    """Enum for dataset directory structure types."""
    SINGLE_ROOT = auto()  # root/class1, root/class2, ...
    TRAIN_TEST_SPLIT = auto()  # root/train/class1, root/test/class1, ...


class LoadingMethod(Enum):
    """Enum for dataset loading methods."""
    FIXED_SPLIT = auto()  # Fixed train-val-test split
    CV_EVALUATION = auto()  # Cross-validation for evaluation (test is one fold)
    CV_TRAINING = auto()  # Cross-validation on training (test is fixed)


def _default_train_transforms() -> transforms.Compose:
    """Default transformations for training data."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _default_test_transforms() -> transforms.Compose:
    """Default transformations for validation and test data."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def _create_dataloaders(
        train_set: Dataset,
        val_set: Dataset,
        test_set: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        train_set: Training dataset
        val_set: Validation dataset
        test_set: Test dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders

    Returns:
        Dict containing train, validation, and test DataLoaders and data
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_set,
        'val_dataset': val_set,
        'test_dataset': test_set,
    }


class GenericDatasetLoader:
    """
    A flexible dataset loader for classification that supports multiple loading methods
    and dataset structures.
    """

    def __init__(
            self,
            root_dir: str,
            loading_method: LoadingMethod,
            num_folds: int = 5,
            test_size: float = 0.2,
            val_size: float = 0.1,
            train_transforms: Optional[transforms.Compose] = None,
            test_transforms: Optional[transforms.Compose] = None,
            seed: int = 42
    ):
        """
        Initialize the dataset loader. Peeks at the dataset structure and checks compatibility with the loading method.

        Args:
            root_dir: Path to the dataset root directory
            loading_method: Method to use for loading the dataset
            num_folds: Number of folds for cross-validation
            test_size: Proportion of the dataset to use for testing (for fixed split)
            val_size: Proportion of the non-test data to use for validation
            train_transforms: Transformations to apply to training data
            test_transforms: Transformations to apply to validation and test data
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.loading_method = loading_method
        self.num_folds = num_folds
        self.test_size = test_size
        self.val_size = val_size
        self.train_transforms = train_transforms or _default_train_transforms()
        self.test_transforms = test_transforms or _default_test_transforms()
        self.seed = seed
        self.structure = self._detect_dataset_structure()

        # Validate loading method and structure compatibility
        self._validate_loading_method()

        # Initialize data
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.num_classes = None

    def _detect_dataset_structure(self) -> DatasetStructure:
        """
        Detect the structure of the dataset based on the directory layout.

        Returns:
            DatasetStructure: The detected dataset structure
        """
        # Check if the dataset contains train and test subdirectories
        if os.path.isdir(os.path.join(self.root_dir, 'train')) and os.path.isdir(os.path.join(self.root_dir, 'test')):
            # Check if train/test directories contain class subdirectories with images
            train_dir = os.path.join(self.root_dir, 'train')
            train_subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

            if train_subdirs and any(
                    any(file.lower().endswith(('.jpg', '.jpeg', '.png'))
                        for file in os.listdir(os.path.join(train_dir, subdir)))
                    for subdir in train_subdirs
            ):
                logger.info("🗃️ Detected TRAIN_TEST_SPLIT dataset structure")
                return DatasetStructure.TRAIN_TEST_SPLIT
            else:
                raise ValueError("❌ Train directory does not contain proper class subdirectories with images")
        else:
            # Get all subdirectories in the root directory
            subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

            # Check if there are image files in subdirectories
            for subdir in subdirs:
                subdir_path = os.path.join(self.root_dir, subdir)
                if any(file.lower().endswith(('.jpg', '.jpeg', '.png')) for file in os.listdir(subdir_path)):
                    logger.info("🗃️ Detected SINGLE_ROOT dataset structure")
                    return DatasetStructure.SINGLE_ROOT

        raise ValueError("❌ Could not detect a valid dataset structure")

    def _validate_loading_method(self) -> None:
        """
        Validate that the loading method is compatible with the dataset structure.

        Raises:
            ValueError: If an incompatible loading method is selected
        """
        if self.structure == DatasetStructure.TRAIN_TEST_SPLIT and self.loading_method == LoadingMethod.CV_EVALUATION:
            raise ValueError(
                "🚨 Cross-validation for evaluation is not supported with the TRAIN_TEST_SPLIT dataset structure. "
                "Please use FIXED_SPLIT or CV_TRAINING loading methods instead."
            )

    def load(self) -> Dict[str, Any]:
        """
        Load the dataset according to the specified loading method.

        Returns:
            Dict containing dataset information and loaders based on the loading method
        """
        if self.structure == DatasetStructure.SINGLE_ROOT:
            self._load_single_root_dataset()
        else:  # TRAIN_TEST_SPLIT structure
            self._load_train_test_split_dataset()

        # Load data based on the specified method
        if self.loading_method == LoadingMethod.FIXED_SPLIT:
            return self._load_fixed_split()
        elif self.loading_method == LoadingMethod.CV_EVALUATION:
            return self._load_cv_evaluation()
        else:  # CV_TRAINING
            return self._load_cv_training()

    def _load_single_root_dataset(self) -> None:
        """Load a dataset with the SINGLE_ROOT structure."""
        logger.info(f"⏳ Loading dataset from {self.root_dir} with SINGLE_ROOT structure")
        self.full_dataset = datasets.ImageFolder(
            root=self.root_dir,
            transform=self.test_transforms
        )
        self.class_names = self.full_dataset.classes
        self.num_classes = len(self.class_names)
        logger.info(f"📂 Loaded {len(self.full_dataset)} images with {self.num_classes} classes")

    def _load_train_test_split_dataset(self) -> None:
        """Load a dataset with the TRAIN_TEST_SPLIT structure."""
        logger.info(f"⏳ Loading dataset from {self.root_dir} with TRAIN_TEST_SPLIT structure")

        # Load train and test data
        train_dir = os.path.join(self.root_dir, 'train')
        test_dir = os.path.join(self.root_dir, 'test')

        self.train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=self.train_transforms
        )

        self.test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=self.test_transforms
        )

        # Ensure class indices match between train and test sets
        if self.train_dataset.class_to_idx != self.test_dataset.class_to_idx:
            logger.warning("⚠️ Class indices do not match between train and test sets! Adjusting test dataset...")
            self.test_dataset.class_to_idx = self.train_dataset.class_to_idx

        self.class_names = self.train_dataset.classes
        self.num_classes = len(self.class_names)
        logger.info(
            f"📂 Loaded {len(self.train_dataset)} training images and {len(self.test_dataset)} test images with {self.num_classes} classes")

    def _load_fixed_split(self) -> Dict[str, Any]:
        """
        Load the dataset with a fixed train-validation-test split.

        Returns:
            Dict containing dataset information and DataLoaders
        """
        if self.structure == DatasetStructure.SINGLE_ROOT:
            # Split the full dataset into train, validation, and test
            num_samples = len(self.full_dataset)
            num_test = int(num_samples * self.test_size)
            num_train_val = num_samples - num_test
            num_val = int(num_train_val * self.val_size)
            num_train = num_train_val - num_val

            train_set, val_set, test_set = random_split(
                self.full_dataset,
                [num_train, num_val, num_test],
                generator=torch.Generator().manual_seed(self.seed)
            )

            # Apply transformations
            train_set = TransformSubset(train_set, transform=self.train_transforms)
            val_set = TransformSubset(val_set, transform=self.test_transforms)
            test_set = TransformSubset(test_set, transform=self.test_transforms)

        else:  # TRAIN_TEST_SPLIT structure
            # The test set is already fixed, we just need to split train into train and validation
            num_train_val = len(self.train_dataset)
            num_val = int(num_train_val * self.val_size)
            num_train = num_train_val - num_val

            train_indices, val_indices = train_test_split(
                list(range(num_train_val)),
                test_size=self.val_size,
                random_state=self.seed
            )

            # Create subsets
            train_set = Subset(self.train_dataset, train_indices)
            val_set = Subset(self.train_dataset, val_indices)

            # No need to apply transformations as train_dataset already has train_transforms
            # But we do need to apply test_transforms to the validation set
            val_set = TransformSubset(val_set, transform=self.test_transforms)

            test_set = self.test_dataset  # Already has test_transforms

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

        return {
            'loader_type': 'fixed_split',
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_dataset': train_set,
            'val_dataset': val_set,
            'test_dataset': test_set,
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }

    def _load_cv_evaluation(self) -> Dict[str, Any]:
        """
        Load the dataset with cross-validation for evaluation.
        Test set is one fold, and the rest is split between train and validation.

        Returns:
            Dict containing dataset information and k-fold cross-validation DataLoaders
        """
        if self.structure == DatasetStructure.TRAIN_TEST_SPLIT:
            raise ValueError("💥 CV_EVALUATION is not supported with the TRAIN_TEST_SPLIT structure")

        # Get all labels for stratification
        all_labels = [label for _, label in self.full_dataset]

        # Initialize Stratified KFold for better class distribution
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        # Prepare cross-validation folds
        fold_dataloaders = []
        indices = list(range(len(self.full_dataset)))

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, all_labels)):
            # Get labels for train_val subset for stratified split
            train_val_labels = [all_labels[i] for i in train_val_idx]

            # Split train_val into train and validation (stratified)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_size,
                random_state=self.seed,
                stratify=train_val_labels
            )

            # Create subsets and apply transforms
            train_set, val_set, test_set = self._create_and_transform_subsets(train_idx, val_idx, test_idx)

            # Create dataloaders
            fold_loaders = _create_dataloaders(train_set, val_set, test_set)

            fold_dataloaders.append({
                'fold': fold_idx,
                **fold_loaders
            })

        return {
            'loader_type': 'cv_evaluation',
            'folds': fold_dataloaders,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'num_folds': self.num_folds
        }

    def _load_cv_training(self) -> Dict[str, Any]:
        """
        Load the dataset with cross-validation on training only.
        Test set is fixed, and the rest is split into folds.

        Returns:
            Dict containing dataset information and k-fold cross-validation DataLoaders
        """
        if self.structure == DatasetStructure.SINGLE_ROOT:
            # Split into test and train_val sets
            num_samples = len(self.full_dataset)
            num_test = int(num_samples * self.test_size)
            num_train_val = num_samples - num_test

            indices = list(range(num_samples))
            np.random.seed(self.seed)
            np.random.shuffle(indices)

            train_val_indices = indices[:num_train_val]
            test_indices = indices[num_train_val:]

            test_set = Subset(self.full_dataset, test_indices)
            test_set = TransformSubset(test_set, transform=self.test_transforms)

            # Initialize KFold for cross-validation on the training set
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        else:  # TRAIN_TEST_SPLIT structure
            train_val_indices = list(range(len(self.train_dataset)))
            test_set = self.test_dataset

            # Initialize KFold for cross-validation on the training set
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        # Prepare cross-validation folds
        fold_dataloaders = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_indices)):
            if self.structure == DatasetStructure.SINGLE_ROOT:
                # Map to the original indices
                train_subset_indices = [train_val_indices[i] for i in train_idx]
                val_subset_indices = [train_val_indices[i] for i in val_idx]

                # Create subsets
                train_set = Subset(self.full_dataset, train_subset_indices)
                val_set = Subset(self.full_dataset, val_subset_indices)

                # Apply transformations
                train_set = TransformSubset(train_set, transform=self.train_transforms)
                val_set = TransformSubset(val_set, transform=self.test_transforms)

            else:  # TRAIN_TEST_SPLIT structure
                # Create subsets from the train dataset
                train_set = Subset(self.train_dataset, [train_val_indices[i] for i in train_idx])
                val_set = Subset(self.train_dataset, [train_val_indices[i] for i in val_idx])

            # Create data loaders
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

            fold_dataloaders.append({
                'fold': fold_idx,
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'train_dataset': train_set,
                'val_dataset': val_set,
                'test_dataset': test_set,
            })

        return {
            'loader_type': 'cv_training',
            'folds': fold_dataloaders,
            'test_dataset': test_set,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'num_folds': self.num_folds
        }

    def _create_and_transform_subsets(
            self,
            train_indices: List[int],
            val_indices: List[int],
            test_indices: List[int],
            dataset: Optional[Dataset] = None
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create and transform subsets from indices.

        Args:
            train_indices: Indices for training set
            val_indices: Indices for validation set
            test_indices: Indices for test set
            dataset: Dataset to create subsets from (defaults to self.full_dataset)

        Returns:
            Tuple of (train_set, val_set, test_set) with transforms applied
        """
        dataset = dataset or self.full_dataset

        # Create subsets
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)

        # Apply transformations
        train_set = TransformSubset(train_set, transform=self.train_transforms)
        val_set = TransformSubset(val_set, transform=self.test_transforms)
        test_set = TransformSubset(test_set, transform=self.test_transforms)

        return train_set, val_set, test_set

    def get_cv_folds(self) -> Generator[Dict[str, Any], None, None]:
        """
        Generator function to yield data for each fold in cross-validation.
        This is more memory-efficient for large data.

        Yields:
            Dict containing data loaders and data for the current fold
        """
        if self.loading_method not in [LoadingMethod.CV_EVALUATION, LoadingMethod.CV_TRAINING]:
            raise ValueError(f"Cannot get CV folds for loading method {self.loading_method}")

        if self.loading_method == LoadingMethod.CV_EVALUATION:
            yield from self._yield_cv_evaluation_folds()
        else:  # CV_TRAINING
            yield from self._yield_cv_training_folds()

    def _yield_cv_evaluation_folds(self) -> Generator[Dict[str, Any], None, None]:
        """Yield folds for CV_EVALUATION loading method."""
        if self.structure == DatasetStructure.TRAIN_TEST_SPLIT:
            raise ValueError("💥 CV_EVALUATION is not supported with the TRAIN_TEST_SPLIT structure")

        # Get all labels for stratification
        all_labels = [label for _, label in self.full_dataset]

        # Initialize Stratified KFold for better class distribution
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        # Prepare cross-validation folds
        indices = list(range(len(self.full_dataset)))

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(indices, all_labels)):
            # Get labels for train_val subset for stratified split
            train_val_labels = [all_labels[i] for i in train_val_idx]

            # Split train_val into train and validation (stratified)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_size,
                random_state=self.seed,
                stratify=train_val_labels
            )

            # Create subsets and apply transforms
            train_set, val_set, test_set = self._create_and_transform_subsets(train_idx, val_idx, test_idx)

            # Create dataloaders
            fold_loaders = self._create_dataloaders(train_set, val_set, test_set)

            yield {
                'fold': fold_idx,
                **fold_loaders
            }

    def _yield_cv_training_folds(self) -> Generator[Dict[str, Any], None, None]:
        """Yield folds for CV_TRAINING loading method."""
        if self.structure == DatasetStructure.SINGLE_ROOT:
            # Split into test and train_val sets
            num_samples = len(self.full_dataset)
            num_test = int(num_samples * self.test_size)
            num_train_val = num_samples - num_test

            indices = list(range(num_samples))
            np.random.seed(self.seed)
            np.random.shuffle(indices)

            train_val_indices = indices[:num_train_val]
            test_indices = indices[num_train_val:]

            test_set = Subset(self.full_dataset, test_indices)
            test_set = TransformSubset(test_set, transform=self.test_transforms)

            # Get labels for stratification
            all_labels = [self.full_dataset[i][1] for i in train_val_indices]

            # Initialize Stratified KFold for cross-validation on the training set
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, all_labels)):
                # Map to the original indices
                train_subset_indices = [train_val_indices[i] for i in train_idx]
                val_subset_indices = [train_val_indices[i] for i in val_idx]

                # Create and transform subsets
                train_set = Subset(self.full_dataset, train_subset_indices)
                val_set = Subset(self.full_dataset, val_subset_indices)

                train_set = TransformSubset(train_set, transform=self.train_transforms)
                val_set = TransformSubset(val_set, transform=self.test_transforms)

                # Create dataloaders
                train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)
                test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

                yield {
                    'fold': fold_idx,
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'test_loader': test_loader,
                    'train_dataset': train_set,
                    'val_dataset': val_set,
                    'test_dataset': test_set,
                }

        else:  # TRAIN_TEST_SPLIT structure
            train_val_indices = list(range(len(self.train_dataset)))
            train_val_labels = [self.train_dataset[i][1] for i in train_val_indices]
            test_set = self.test_dataset

            # Initialize Stratified KFold for cross-validation on the training set
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_labels)):
                # Create subsets from the train dataset
                train_subset = Subset(self.train_dataset, [train_val_indices[i] for i in train_idx])
                val_subset = Subset(self.train_dataset, [train_val_indices[i] for i in val_idx])

                # Apply transformations to val_set (train_set already has train_transforms applied)
                val_subset = TransformSubset(val_subset, transform=self.test_transforms)

                # Create dataloaders
                train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
                val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
                test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

                yield {
                    'fold': fold_idx,
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'test_loader': test_loader,
                    'train_dataset': train_subset,
                    'val_dataset': val_subset,
                    'test_dataset': test_set,
                }


class TransformSubset(Dataset):
    """
    A custom subset with different transformations.
    This allows applying different transformations to train/val/test splits
    of the same dataset.
    """

    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None):
        """
        Initialize a transform subset.

        Args:
            subset: The subset to apply transformations to
            transform: The transformations to apply
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        """Get an item from the subset and apply transformations."""
        x, y = self.subset[idx]

        # If x is already a tensor, it means transformations have been applied
        # In this case, we need to convert it back to PIL image
        if isinstance(x, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            x = to_pil_image(x)

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        """Get the length of the subset."""
        return len(self.subset)


class ModelTrainer:
    """Model trainer with configurable training parameters."""

    def __init__(
            self,
            model: torch.nn.Module,
            criterion: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            device: torch.device = None,
            config: Dict[str, Any] = None,
    ):
        """
        Initialize trainer with model and training configurations.

        Args:
            model: PyTorch model to train
            criterion: Loss function (defaults to CrossEntropyLoss if None)
            optimizer: Optimizer (defaults to Adam if None)
            scheduler: Learning rate scheduler
            device: Device to use (defaults to cuda if available)
            config: Training configuration parameters
        """
        self.model = model
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default configuration
        self.config = {
            'num_epochs': 10,
            'early_stopping_patience': 5,
            'grad_clip_value': None,
            'log_interval': 10,
            'eval_interval': 1,
            'save_best_model': True,
            'metrics': ['accuracy']
        }

        # Update with user-provided config
        if config:
            self.config.update(config)

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Setup metrics for evaluation."""
        self.metrics = {}
        for metric_name in self.config['metrics']:
            if metric_name == 'accuracy':
                self.metrics[metric_name] = lambda preds, labels: (preds.argmax(dim=1) == labels).float().mean().item()
            # Add more metrics as needed

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            fold_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            fold_idx: Optional fold index for cross-validation

        Returns:
            Dict containing training history and best model state
        """
        fold_prefix = f"Fold {fold_idx}: " if fold_idx is not None else ""
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': {name: [] for name in self.metrics},
            'val_metrics': {name: [] for name in self.metrics}
        }

        best_val_loss = float('inf')
        best_model_state = self.model.state_dict().copy()
        patience_counter = 0

        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation phase (if eval_interval matches)
            if (epoch + 1) % self.config['eval_interval'] == 0:
                val_metrics = self._eval_epoch(val_loader)

                # Update history
                for k, v in train_metrics.items():
                    history[f'train_{k}'].append(v)
                for k, v in val_metrics.items():
                    history[f'val_{k}'].append(v)

                # Print statistics
                metrics_str = ', '.join([
                                            f"Train {k}: {v:.4f}" for k, v in train_metrics.items()
                                        ] + [
                                            f"Val {k}: {v:.4f}" for k, v in val_metrics.items()
                                        ])
                logger.info(f"{fold_prefix}Epoch {epoch + 1}/{self.config['num_epochs']} - {metrics_str}")

                # Early stopping check
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    if self.config['save_best_model']:
                        best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        logger.info(f"{fold_prefix}Early stopping at epoch {epoch + 1}")
                        break

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

        # Load the best model if saved
        if self.config['save_best_model']:
            self.model.load_state_dict(best_model_state)

        return {
            'model': self.model,
            'history': history,
            'best_val_loss': best_val_loss,
            'best_model_state': best_model_state
        }

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch and return metrics."""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in self.metrics}
        num_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()

            # Gradient clipping if configured
            if self.config['grad_clip_value']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip_value']
                )

            self.optimizer.step()

            # Update running loss
            epoch_loss += loss.item()

            # Calculate metrics
            with torch.no_grad():
                for name, metric_fn in self.metrics.items():
                    epoch_metrics[name] += metric_fn(outputs, labels)

            # Log batch progress if configured
            if self.config['log_interval'] and batch_idx % self.config['log_interval'] == 0:
                logger.debug(f"Epoch: {epoch + 1}/{self.config['num_epochs']} "
                             f"[{batch_idx + 1}/{num_batches} "
                             f"({100. * batch_idx / num_batches:.0f}%)] "
                             f"Loss: {loss.item():.6f}")

        # Compute averages
        avg_loss = epoch_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in epoch_metrics.items()}

        return {'loss': avg_loss, **avg_metrics}

    def _eval_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on data loader and return metrics."""
        self.model.eval()
        epoch_loss = 0.0
        epoch_metrics = {name: 0.0 for name in self.metrics}
        num_batches = len(data_loader)

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Update running loss
                epoch_loss += loss.item()

                # Calculate metrics
                for name, metric_fn in self.metrics.items():
                    epoch_metrics[name] += metric_fn(outputs, labels)

        # Compute averages
        avg_loss = epoch_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in epoch_metrics.items()}

        return {'loss': avg_loss, **avg_metrics}

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dict containing evaluation metrics and predictions
        """
        self.model.eval()
        test_metrics = self._eval_epoch(test_loader)

        # Get all predictions and labels for additional metrics
        all_predictions = []
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Save predictions and labels
                all_outputs.append(outputs.cpu())
                all_predictions.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

        # Concatenate lists
        all_outputs = torch.cat(all_outputs, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        logger.info(f"Test evaluation - " + ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))

        return {
            **test_metrics,
            'outputs': all_outputs.numpy(),
            'predictions': all_predictions.numpy(),
            'labels': all_labels.numpy()
        }


# Main function to demonstrate how to use the dataset loader with training and evaluation
def main(
        root_dir: str,
        loading_method: LoadingMethod,
        model_factory: Callable[..., torch.nn.Module],
        training_config: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[Dict[str, Any]] = None,
        seed: int = 42
) -> Dict[str, Any]:
    """
    Main function to load a dataset, train and evaluate a model.

    Args:
        root_dir: Path to the dataset root directory
        loading_method: Method to use for loading the dataset
        model_factory: Function to create a model
        training_config: Configuration for the training process
        dataset_config: Configuration for the dataset loader
        seed: Random seed for reproducibility

    Returns:
        Dict containing results
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Default dataset config
    default_dataset_config = {
        'num_folds': 5,
        'test_size': 0.2,
        'val_size': 0.1,
        'batch_size': 32,
        'num_workers': 4
    }

    # Update with user-provided config
    dataset_config = {**default_dataset_config, **(dataset_config or {})}

    # Initialize the dataset loader
    dataset_loader = GenericDatasetLoader(
        root_dir=root_dir,
        loading_method=loading_method,
        num_folds=dataset_config['num_folds'],
        test_size=dataset_config['test_size'],
        val_size=dataset_config['val_size'],
        seed=seed
    )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize results dictionary
    results = {
        'loading_method': loading_method,
    }

    if loading_method == LoadingMethod.FIXED_SPLIT:
        # Load the dataset with fixed split
        data = dataset_loader.load()
        results['num_classes'] = data['num_classes']
        results['class_names'] = data['class_names']

        # Create model and trainer
        model = model_factory(num_classes=data['num_classes'])
        trainer = ModelTrainer(model, device=device, config=training_config)

        # Train the model
        train_results = trainer.train(
            train_loader=data['train_loader'],
            val_loader=data['val_loader']
        )

        # Evaluate the model
        eval_results = trainer.evaluate(data['test_loader'])

        results['train_results'] = train_results
        results['eval_results'] = eval_results

    else:  # CV_EVALUATION or CV_TRAINING
        # Load basic info
        data_info = dataset_loader.load()
        results['num_classes'] = data_info['num_classes']
        results['class_names'] = data_info['class_names']

        # For memory efficiency, use generator to iterate through folds
        fold_results = []

        for fold_data in dataset_loader.get_cv_folds():
            # Create a new model for each fold
            model = model_factory(num_classes=data_info['num_classes'])
            trainer = ModelTrainer(model, device=device, config=training_config)

            # Train the model
            train_results = trainer.train(
                train_loader=fold_data['train_loader'],
                val_loader=fold_data['val_loader'],
                fold_idx=fold_data['fold']
            )

            # Evaluate the model
            eval_results = trainer.evaluate(fold_data['test_loader'])

            fold_results.append({
                'fold': fold_data['fold'],
                'train_results': train_results,
                'eval_results': eval_results
            })

        results['fold_results'] = fold_results

        # Calculate average metrics across folds
        avg_metrics = {}
        for metric in fold_results[0]['eval_results'].keys():
            if metric not in ['outputs', 'predictions', 'labels']:
                avg_metrics[f'avg_{metric}'] = np.mean([
                    fold['eval_results'][metric] for fold in fold_results
                ])
                logger.info(f"Average {metric}: {avg_metrics[f'avg_{metric}']:.4f}")

        results.update(avg_metrics)

    return results


# Example usage
if __name__ == "__main__":
    # Define a simple model factory function
    def create_resnet18(num_classes):
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model


    # Define training configuration
    training_config = {
        'num_epochs': 10,
        'early_stopping_patience': 3,
        'grad_clip_value': 1.0,
        'metrics': ['accuracy']
    }

    # Define dataset configuration
    dataset_config = {
        'num_folds': 5,
        'test_size': 0.2,
        'val_size': 0.15,
        'batch_size': 64,
        'num_workers': 4
    }

    # Example for SINGLE_ROOT structure with FIXED_SPLIT loading method
    results = main(
        root_dir="/path/to/cloud/dataset",
        loading_method=LoadingMethod.FIXED_SPLIT,
        model_factory=create_resnet18,
        training_config=training_config,
        dataset_config=dataset_config
    )

    # Example for TRAIN_TEST_SPLIT structure with CV_TRAINING loading method
    results = main(
        root_dir="/path/to/cloud/dataset_with_split",
        loading_method=LoadingMethod.CV_TRAINING,
        model_factory=create_resnet18,
        num_epochs=5
    )

    # Example for SINGLE_ROOT structure with CV_EVALUATION loading method
    results = main(
        root_dir="/path/to/cloud/dataset",
        loading_method=LoadingMethod.CV_EVALUATION,
        model_factory=create_resnet18,
        num_epochs=5
    )