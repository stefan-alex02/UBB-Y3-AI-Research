import os
import logging
from enum import Enum, auto
from typing import Tuple, List, Optional, Dict, Any, Union, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.model_selection import KFold, train_test_split
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

        # Initialize datasets
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
            logger.info("🗃️ Detected TRAIN_TEST_SPLIT dataset structure")
            # TODO - Check if train/test directories contain class subdirectories with images (like below)
            return DatasetStructure.TRAIN_TEST_SPLIT
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

        # Load datasets based on the specified method
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

        # Load train and test datasets
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

            train_set, val_set = random_split(
                self.train_dataset,
                [num_train, num_val],
                generator=torch.Generator().manual_seed(self.seed)
            )

            # TODO - Apply transformations to train_set and val_set in this case !!

            test_set = self.test_dataset

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

        # Initialize KFold
        # TODO - Use stratified KFold for better class distribution
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        # Prepare cross-validation folds
        fold_dataloaders = []
        indices = list(range(len(self.full_dataset)))

        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
            # Split train_val into train and validation
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_size,
                random_state=self.seed
            )

            # Create subsets
            train_set = Subset(self.full_dataset, train_idx)
            val_set = Subset(self.full_dataset, val_idx)
            test_set = Subset(self.full_dataset, test_idx)

            # Apply transformations
            train_set = TransformSubset(train_set, transform=self.train_transforms)
            val_set = TransformSubset(val_set, transform=self.test_transforms)
            test_set = TransformSubset(test_set, transform=self.test_transforms)

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
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        else:  # TRAIN_TEST_SPLIT structure
            train_val_indices = list(range(len(self.train_dataset)))
            test_set = self.test_dataset

            # Initialize KFold for cross-validation on the training set
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

        # Prepare cross-validation folds
        fold_dataloaders = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
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


# Main training and evaluation functions
def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_epochs: int = 10,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 5
) -> Dict[str, Any]:
    """
    Train a model.

    Args:
        model: The model to train
        train_loader: DataLoader for the training data
        val_loader: DataLoader for the validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        num_epochs: Number of epochs to train for
        scheduler: Learning rate scheduler
        early_stopping_patience: Number of epochs to wait before stopping if validation loss doesn't improve

    Returns:
        Dict containing training history and best model state
    """
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    best_val_loss = float('inf')
    best_model_state = model.state_dict().copy()
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Calculate average losses and accuracies
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_train_acc = train_correct / train_total
        epoch_val_acc = val_correct / val_total

        # Update history
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Print statistics
        logger.info(f'Epoch {epoch + 1}/{num_epochs} - '
                    f'Train Loss: {epoch_train_loss:.4f}, '
                    f'Val Loss: {epoch_val_loss:.4f}, '
                    f'Train Acc: {epoch_train_acc:.4f}, '
                    f'Val Acc: {epoch_val_acc:.4f}')

        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping at epoch {epoch + 1}')
                break

    # Load the best model
    model.load_state_dict(best_model_state)

    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'best_model_state': best_model_state
    }


def evaluate_model(
        model: torch.nn.Module,
        test_loader: DataLoader,
        criterion: torch.nn.Module,
        device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate a model on the test set.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test data
        criterion: Loss function
        device: Device to use for evaluation

    Returns:
        Dict containing evaluation metrics
    """
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            # Save predictions and labels for additional metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy
    avg_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total

    logger.info(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }


# Main function to demonstrate how to use the dataset loader with training and evaluation
def main(
        root_dir: str,
        loading_method: LoadingMethod,
        model_factory: Callable[..., torch.nn.Module],
        num_folds: int = 5,
        test_size: float = 0.2,
        val_size: float = 0.1,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        seed: int = 42
) -> Dict[str, Any]:
    """
    Main function to load a dataset, train and evaluate a model.

    Args:
        root_dir: Path to the dataset root directory
        loading_method: Method to use for loading the dataset
        model_factory: Function to create a model
        num_folds: Number of folds for cross-validation
        test_size: Proportion of the dataset to use for testing
        val_size: Proportion of the non-test data to use for validation
        num_epochs: Number of epochs to train for
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        seed: Random seed for reproducibility

    Returns:
        Dict containing results
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize the dataset loader
    dataset_loader = GenericDatasetLoader(
        root_dir=root_dir,
        loading_method=loading_method,
        num_folds=num_folds,
        test_size=test_size,
        val_size=val_size,
        seed=seed
    )

    # Load the dataset
    data = dataset_loader.load()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize results dictionary
    results = {
        'loading_method': loading_method,
        'num_classes': data['num_classes'],
        'class_names': data['class_names']
    }

    # Initialize criterion
    criterion = torch.nn.CrossEntropyLoss()

    if loading_method == LoadingMethod.FIXED_SPLIT:
        # Create model
        model = model_factory(num_classes=data['num_classes'])

        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train the model
        train_results = train_model(
            model=model,
            train_loader=data['train_loader'],
            val_loader=data['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            scheduler=scheduler
        )

        # Evaluate the model
        eval_results = evaluate_model(
            model=train_results['model'],
            test_loader=data['test_loader'],
            criterion=criterion,
            device=device
        )

        results['train_results'] = train_results
        results['eval_results'] = eval_results

    else:  # CV_EVALUATION or CV_TRAINING
        fold_results = []

        for fold_data in data['folds']:
            # Create model
            model = model_factory(num_classes=data['num_classes'])

            # Initialize optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

            # Train the model
            train_results = train_model(
                model=model,
                train_loader=fold_data['train_loader'],
                val_loader=fold_data['val_loader'],
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
                scheduler=scheduler
            )

            # Evaluate the model
            eval_results = evaluate_model(
                model=train_results['model'],
                test_loader=fold_data['test_loader'],
                criterion=criterion,
                device=device
            )

            fold_results.append({
                'fold': fold_data['fold'],
                'train_results': train_results,
                'eval_results': eval_results
            })

        results['fold_results'] = fold_results

        # Calculate average metrics across folds
        avg_test_accuracy = np.mean([fold['eval_results']['test_accuracy'] for fold in fold_results])
        avg_test_loss = np.mean([fold['eval_results']['test_loss'] for fold in fold_results])

        results['avg_test_accuracy'] = avg_test_accuracy
        results['avg_test_loss'] = avg_test_loss

        logger.info(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
        logger.info(f"Average Test Loss: {avg_test_loss:.4f}")

    return results


# Example usage
if __name__ == "__main__":
    # Define a simple model factory function
    def create_resnet18(num_classes):
        import torchvision.models as models
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model


    # Example for SINGLE_ROOT structure with FIXED_SPLIT loading method
    results = main(
        root_dir="/path/to/cloud/dataset",
        loading_method=LoadingMethod.FIXED_SPLIT,
        model_factory=create_resnet18,
        num_epochs=5
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