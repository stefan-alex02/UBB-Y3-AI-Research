import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import os
from pathlib import Path
import time
import random

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    PredefinedSplit, BaseCrossValidator
)
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder # Useful if not using ImageFolder directly for labels

from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset as SkorchDataset, ValidSplit  # For compatibility if needed

# --- Configuration ---
DATA_DIR = '../../data/mini-GCD-flat'  # CHANGE THIS: Path to root dataset folder
# e.g., ./your_dataset_folder/class_a/img1.jpg, ./your_dataset_folder/class_b/img1.jpg
OUTPUT_DIR = './output/' # Directory to save results/models
NUM_CLASSES = -1 # Will be determined from data
IMAGE_SIZE = 64 # Target image size
MEAN = [0.485, 0.456, 0.406] # Example values, calculate for your dataset
STD = [0.229, 0.224, 0.225] # Example values, calculate for your dataset
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# --- Set Seed for Reproducibility ---
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
# Potentially add torch.backends.cudnn.deterministic = True / benchmark = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Define PyTorch Model ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5): # Make params tunable
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 64 -> 32

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 32 -> 16

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2) # 16 -> 8

        self.flatten = nn.Flatten()
        # Calculate flattened size: 64 channels * 8 * 8
        self.fc1 = nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 2. Define Transforms ---
train_transforms = transforms.Compose([
    # transforms.ToPILImage(), # Needed if input is numpy HWC initially
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # Add more aggressive augmentations if needed
    transforms.ToTensor(), # Converts PIL/numpy (HxWxC or HxW) to CxHxW tensor [0,1]
    transforms.Normalize(mean=MEAN, std=STD),
])

valid_transforms = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# --- 3. Custom PyTorch Dataset (Loads from paths) ---
class PathImageDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None, class_to_idx=None):
        # ... (init remains the same) ...
        self.paths = paths
        self.labels = labels # This can be None
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label_val = self.labels[idx] if self.labels is not None else -1 # Get label or -1 if None

        # --- Load Image ---
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Error loading image {img_path}: {e}. Skipping item.")
            # Return None for both elements if loading fails, collate_fn will filter
            return None, None

        # --- Apply transformations ---
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                 print(f"Warning: Error transforming image {img_path}: {e}. Skipping item.")
                 # Return None for both elements if transform fails, collate_fn will filter
                 return None, None

        # --- Convert label to tensor ---
        label_tensor = torch.tensor(label_val, dtype=torch.long)

        # --- Always return a tuple (image_tensor, label_tensor) ---
        return image, label_tensor

    # --- Collate function (handles None filtering) ---
    @staticmethod
    def collate_fn(batch):
        # Filter out items where __getitem__ returned (None, None) due to errors
        original_batch_size = len(batch)
        # Ensure item itself is not None, and both elements within the tuple are not None
        batch = [item for item in batch if item is not None and item[0] is not None and item[1] is not None]

        if not batch:
            # Handle case where all items in the batch failed to load/transform
            print(f"Warning: Collate_fn received an empty batch after filtering {original_batch_size} items.")
            # Return empty tensors. Adjust channel/size if necessary.
            # Need to know the expected shape. Assuming C=3, H=W=IMAGE_SIZE
            # Find IMAGE_SIZE from global scope or pass it in. For now, assuming it's accessible.
            # Might need a more robust way to get IMAGE_SIZE if it's not global.
            try:
                 img_c, img_h, img_w = 3, IMAGE_SIZE, IMAGE_SIZE # Assuming global IMAGE_SIZE
            except NameError:
                 img_c, img_h, img_w = 3, 64, 64 # Fallback if global not found
                 print("Warning: Global IMAGE_SIZE not found in collate_fn, using fallback 64x64.")

            return torch.empty((0, img_c, img_h, img_w)), torch.empty((0), dtype=torch.long)


        # Unzip the batch (items are now guaranteed to be pairs if not None)
        try:
             images, labels = zip(*batch)
        except ValueError as e:
            # This error is less likely now, but good to keep for diagnostics
            print(f"Error during zip in collate_fn: {e}")
            print(f"Problematic batch content (first 5 items): {[str(b)[:100] for b in batch[:5]]}") # Log snippet
            raise e

        # Stack tensors
        try:
            images = torch.stack(images, 0)
            labels = torch.stack(labels, 0)
        except Exception as e:
            print(f"Error during torch.stack in collate_fn: {e}")
            # This could happen if images have different sizes (Resize should prevent this)
            print(f"Number of images: {len(images)}, Number of labels: {len(labels)}")
            # Example: Log shape of first image if possible
            if images: print(f"Shape of first image tensor: {images[0].shape}")
            raise e

        return images, labels


# --- 4. Subclass skorch.NeuralNetClassifier ---
class AugmentingNet(NeuralNetClassifier):
    def __init__(self, *args, train_transform=None, valid_transform=None, class_to_idx=None, **kwargs):
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.class_to_idx_ = class_to_idx # Store mapping

        # Pass kwargs BEFORE potentially modifying them
        # Check if 'cv' param suggests external CV (int or CV object)
        # Sklearn's cross_val_score/GridSearchCV pass the estimator and data,
        # they don't modify the estimator's 'cv' param directly during the loop.
        # The key is that skorch's internal train_split should be disabled
        # if external splitting is happening (which GridSearchCV does by default).
        # Let's explicitly set train_split=None if we intend external CV.
        # If the user provides a train_split value (e.g. 0.2), skorch's
        # internal validation will run *in addition* to GridSearchCV's validation.
        # Usually, we want GridSearchCV/cross_val_score to handle the validation split.
        if 'train_split' not in kwargs:
             kwargs['train_split'] = None # Default to None if not specified

        # Ensure appropriate collate_fn if using custom dataset that might return None
        if 'iterator_train__collate_fn' not in kwargs:
            kwargs['iterator_train__collate_fn'] = PathImageDataset.collate_fn
        if 'iterator_valid__collate_fn' not in kwargs:
            kwargs['iterator_valid__collate_fn'] = PathImageDataset.collate_fn

        # Initialize the parent class FIRST
        super().__init__(*args, **kwargs)

    # Override get_dataset to inject the correct transform
    def get_dataset(self, X, y=None):
        # X is expected to be the list/array of image paths

        # IMPORTANT FIX: Check the 'training' attribute of the underlying module
        # This attribute is set by skorch via module.train() and module.eval() calls
        # Need to ensure the module is initialized first.
        if not self.initialized_ or not hasattr(self, 'module_'):
            # If the net is not initialized, it's likely before the first fit/train loop
            # or during scoring before fit. Defaulting to valid_transform is safer.
            print("Warning: Module not initialized in get_dataset, using valid_transform.")
            current_transform = self.valid_transform
        else:
            # Check the training status of the initialized PyTorch module
            current_transform = self.train_transform if self.module_.training else self.valid_transform
            # print(f"get_dataset called. module.training = {self.module_.training}. Using {'train' if self.module_.training else 'valid'} transform.") # DEBUG line


        return PathImageDataset(X, y, transform=current_transform, class_to_idx=self.class_to_idx_)

    # Optional: Override methods like predict/predict_proba if you need to ensure
    # valid_transform is used explicitly during prediction outside of fit/score cycle.
    # However, skorch should handle setting module to eval mode correctly.
    # def predict(self, X):
    #     # Ensure module is in eval mode before calling super().predict
    #     # skorch does this internally, but being explicit doesn't hurt
    #     if self.initialized_:
    #         self.module_.eval()
    #     return super().predict(X)

    # def predict_proba(self, X):
    #     if self.initialized_:
    #         self.module_.eval()
    #     return super().predict_proba(X)


# --- 5. Data Loading Function ---
def load_image_paths_and_labels(data_dir):
    """Loads image paths and labels from a directory structured like ImageFolder."""
    data_dir = Path(data_dir)
    paths = []
    labels = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = data_dir / class_name
        for img_path in class_dir.glob('*.*'): # Adjust glob pattern if needed (e.g., '*.jpg')
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                paths.append(str(img_path))
                labels.append(class_to_idx[class_name])

    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}. Check directory structure and patterns.")

    print(f"Found {len(paths)} images in {len(class_names)} classes.")
    global NUM_CLASSES # Update global variable
    NUM_CLASSES = len(class_names)
    # Convert to numpy arrays for easier handling with sklearn
    return np.array(paths), np.array(labels), class_names, class_to_idx


# --- 6. Evaluation Strategy Functions ---

def run_gridsearch_cv(X, y, param_grid, cv_folds, class_to_idx, test_size=0.2):
    """
    Performs standard GridSearchCV:
    1. Splits data into Train+Validation / Test sets.
    2. Runs GridSearchCV on the Train+Validation set.
    3. Evaluates the best model found by GridSearch on the held-out Test set.
    """
    print("\n--- Running Standard Cross-Validation with Grid Search ---")
    start_time = time.time()

    # 1. Split into Trainval and Test
    if test_size > 0:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
        )
        print(f"Data split: {len(y_trainval)} trainval / {len(y_test)} test samples.")
    else:
         X_trainval, y_trainval = X, y
         X_test, y_test = None, None # No final test evaluation
         print(f"Using all {len(y_trainval)} samples for GridSearch (no final test set).")


    # 2. Setup Skorch Net and GridSearchCV
    net = AugmentingNet(
        module=SimpleCNN,
        module__num_classes=NUM_CLASSES,  # Pass fixed param
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        # Default params (will be overridden by grid search)
        lr=0.01,
        batch_size=32,
        max_epochs=10,
        # Pass transforms and other fixed params
        train_transform=train_transforms,
        valid_transform=valid_transforms,
        class_to_idx=class_to_idx,
        device=DEVICE,

        # === FIX HERE ===
        # Define an internal validation split for skorch to use during fit()
        # This allows EarlyStopping(monitor='valid_loss') to work.
        # Adjust the fraction (e.g., 0.15 = 15%) as needed.
        train_split=ValidSplit(cv=0.15, stratified=True, random_state=RANDOM_SEED),  # Use skorch's ValidSplit

        callbacks=[('early_stop', EarlyStopping(patience=5, monitor='valid_loss', load_best=True))],
        iterator_train__shuffle=True,
        # train_split=None IS REMOVED/OVERRIDDEN by the line above
    )

    # Define CV strategy for GridSearchCV
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)

    gs = GridSearchCV(
        estimator=net,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=make_scorer(accuracy_score), # Use accuracy or other metrics
        refit=True, # Refit the best estimator on the whole trainval set
        verbose=2,
        error_score='raise',
        n_jobs=1 # Set > 1 for parallel folds if resources allow (be careful with GPU memory)
    )

    # 3. Run GridSearch Fit
    print("Starting GridSearchCV Fit...")
    gs.fit(X_trainval, y_trainval) # Pass paths and labels

    print("\nGridSearchCV Results:")
    print(f"Best parameters found: {gs.best_params_}")
    print(f"Best cross-validation score (accuracy): {gs.best_score_:.4f}")

    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'gridsearch_cv_results.csv'), index=False)
    print(f"CV results saved to {os.path.join(OUTPUT_DIR, 'gridsearch_cv_results.csv')}")


    # 4. Evaluate on Test Set (if available)
    test_accuracy = None
    if X_test is not None and y_test is not None and hasattr(gs, 'best_estimator_'):
        print("\nEvaluating best model on the held-out test set...")
        best_model = gs.best_estimator_
        # Ensure model is in eval mode and uses valid_transforms (skorch handles this)
        test_accuracy = best_model.score(X_test, y_test)
        print(f"Test set accuracy: {test_accuracy:.4f}")
        # Save the best model
        torch.save(best_model.module_.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_standard_cv.pth'))
        print(f"Best model state dict saved to {os.path.join(OUTPUT_DIR, 'best_model_standard_cv.pth')}")
    elif X_test is None:
        print("\nNo test set provided, skipping final evaluation.")
        # Optionally save the model trained on the full trainval set
        if hasattr(gs, 'best_estimator_'):
             torch.save(gs.best_estimator_.module_.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_standard_cv_full_trainval.pth'))
             print(f"Best model (trained on full trainval) state dict saved.")


    total_time = time.time() - start_time
    print(f"\nStandard CV finished in {total_time:.2f} seconds.")
    return gs.best_score_, test_accuracy, gs.best_params_


def run_nested_cv(X, y, param_grid, inner_cv_folds, outer_cv_folds, class_to_idx):
    """
    Performs Nested Cross-Validation:
    - Outer loop splits data into K Train/Test folds.
    - Inner loop (GridSearchCV) runs on each Outer Train fold to find best hyperparameters FOR THAT FOLD.
    - The model with best inner-loop hyperparameters is evaluated on the corresponding Outer Test fold.
    - Scores are averaged across Outer Test folds. Provides a less biased estimate of generalization error.
    """
    print("\n--- Running Nested Cross-Validation ---")
    start_time = time.time()

    # Base Skorch Net (hyperparameters will be set by inner GridSearchCV)
    # Note: Callbacks like EarlyStopping might behave subtly here, as they reset each inner loop.
    # Consider adjusting patience or using simpler callbacks if needed.
    base_net = AugmentingNet(
        module=SimpleCNN,
        module__num_classes=NUM_CLASSES,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        # Set placeholder/default HPs - will be tuned
        lr=0.01,
        batch_size=32,
        max_epochs=10,
        # Pass transforms and other fixed params
        train_transform=train_transforms,
        valid_transform=valid_transforms,
        class_to_idx=class_to_idx,
        device=DEVICE,
        train_split=ValidSplit(cv=0.15, stratified=True, random_state=RANDOM_SEED),  # <-- ADD THIS
        callbacks=[('early_stop', EarlyStopping(patience=5, monitor='valid_loss', load_best=True))],
        iterator_train__shuffle=True,
    )

    # Inner CV strategy (used by GridSearchCV)
    inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=RANDOM_SEED)

    # Setup GridSearchCV (will be used within the outer loop)
    # refit=True here ensures the best model from the *inner* CV is available for scoring on outer test set
    gs = GridSearchCV(
        estimator=base_net,
        param_grid=param_grid,
        cv=inner_cv,
        scoring=make_scorer(accuracy_score),
        refit=True,
        verbose=0, # Keep inner loop less verbose
        error_score='raise',
        n_jobs=1 # Parallelism here applies to inner CV folds
    )

    # Outer CV strategy
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=RANDOM_SEED + 1) # Use different seed

    # Run nested cross-validation using cross_val_score
    print(f"Running Nested CV with {outer_cv_folds} outer folds and {inner_cv_folds} inner folds...")
    # cross_val_score handles the outer loop, fitting gs (which includes inner CV) on each outer train split
    # and scoring on the outer test split.
    nested_scores = cross_val_score(
        gs, X=X, y=y, cv=outer_cv, scoring=make_scorer(accuracy_score), n_jobs=1 # Set n_jobs > 1 for parallel OUTER folds if desired
    )

    total_time = time.time() - start_time
    print(f"\nNested CV finished in {total_time:.2f} seconds.")
    print(f"Outer fold scores: {nested_scores}")
    print(f"Mean Nested CV Accuracy: {np.mean(nested_scores):.4f} (+/- {np.std(nested_scores):.4f})")

    # Note: Nested CV primarily gives a performance estimate. It doesn't inherently produce
    # a single 'best' model trained on all data, as hyperparameters might vary per outer fold.
    # To get a final model, you could re-run standard GridSearchCV on the *entire* dataset (X, y)
    # using the parameter grid, find the best overall parameters, and refit.
    print("\nTo get a final production model, run standard GridSearchCV on the full dataset.")

    return np.mean(nested_scores), np.std(nested_scores)


def run_single_split(X, y, params, class_to_idx, val_size=0.2, test_size=0.15):
    """
    Performs a single training run with dedicated validation and test sets.
    - Splits data initially into Train / Validation / Test.
    - Trains the model on the Train set.
    - Uses the Validation set DURING training for monitoring (e.g., EarlyStopping),
      specified using PredefinedSplit wrapped in ValidSplit.
    - Evaluates the final model on the Test set.
    """
    print("\n--- Running Single Train/Validation/Test Split ---")
    start_time = time.time()

    # 1. Split into Train+Val / Test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED, shuffle=True
    )
    # 2. Split Train+Val into Train / Validation
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else val_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val_size, stratify=y_trainval,
        random_state=RANDOM_SEED + 1, shuffle=True
    )

    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    print(f"Data split: {n_train} train / {n_val} validation / {n_test} test samples.")
    if n_train == 0 or n_val == 0:
         print("Error: Train or Validation set is empty after splitting. Adjust sizes.")
         return None, None

    # 3. Combine Train and Validation data for skorch's fit method
    X_fit = np.concatenate((X_train, X_val))
    y_fit = np.concatenate((y_train, y_val))

    # 4. Create the PredefinedSplit object based on the combined data
    test_fold = np.full(X_fit.shape[0], -1, dtype=int)
    validation_indices_in_fit = np.arange(n_train, n_train + n_val)
    test_fold[validation_indices_in_fit] = 0
    ps = PredefinedSplit(test_fold=test_fold)

    # 5. Setup Skorch Net with specific parameters
    net = AugmentingNet(
        module=SimpleCNN,
        module__num_classes=NUM_CLASSES,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        # Pass hyperparameters directly from the 'params' dict
        lr=params.get('lr', 0.001),
        batch_size=params.get('batch_size', 32),
        max_epochs=params.get('max_epochs', 10),
        module__dropout_rate=params.get('module__dropout_rate', 0.5),
        # Pass transforms and other fixed params
        train_transform=train_transforms,
        valid_transform=valid_transforms,
        class_to_idx=class_to_idx,
        device=DEVICE,

        # --- FIX HERE: Wrap PredefinedSplit in ValidSplit ---
        # skorch's train_split expects a callable or None.
        # ValidSplit IS callable and can accept a CV iterator (like PredefinedSplit)
        train_split=ValidSplit(cv=ps, stratified=False), # Wrap ps in ValidSplit

        callbacks=[
            ('early_stop', EarlyStopping(patience=5, monitor='valid_loss', load_best=True)),
            ],
        iterator_train__shuffle=True,
        verbose=1
    )

    # 6. Train the model
    print("Starting model training...")
    # Pass the combined data. Skorch's internal call to self.train_split
    # will now call the ValidSplit instance, which will use the PredefinedSplit (ps)
    # correctly to determine the train/validation indices.
    net.fit(X_fit, y_fit)

    # 7. Retrieve Validation Score
    print("\nEvaluating final model on the validation set...")
    val_accuracy = net.score(X_val, y_val)
    print(f"Validation set accuracy (final model): {val_accuracy:.4f}")

    # 8. Evaluate on Test Set
    test_accuracy = None
    if X_test is not None and y_test is not None and n_test > 0:
        print("\nEvaluating final model on the held-out test set...")
        test_accuracy = net.score(X_test, y_test)
        print(f"Test set accuracy: {test_accuracy:.4f}")
        try:
            net.save_params(f_params=os.path.join(OUTPUT_DIR, 'best_model_single_split.pt'))
            print(f"Model parameters saved to {os.path.join(OUTPUT_DIR, 'best_model_single_split.pt')}")
        except Exception as e:
            print(f"Error saving model parameters: {e}")
    else:
         print("\nNo test set provided or it was empty, skipping final test evaluation.")


    total_time = time.time() - start_time
    print(f"\nSingle Split Training finished in {total_time:.2f} seconds.")
    return val_accuracy, test_accuracy


def run_cv_for_evaluation(X, y, params, cv_folds, class_to_idx):
    """
    Performs K-Fold Cross-Validation for evaluation.
    - Splits data into K folds.
    - In each iteration i:
        - Fold i is used as the TEST set.
        - Folds != i are used as the TRAINING set.
        - A portion of the TRAINING set is used for internal validation (e.g., EarlyStopping).
        - A model with FIXED hyperparameters ('params') is trained and evaluated.
    - Reports the performance on each test fold.
    """
    print(f"\n--- Running {cv_folds}-Fold CV for Evaluation (Fixed Params) ---")
    start_time = time.time()

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_test_scores = []
    fold_val_scores = [] # Keep track of internal validation scores too

    for fold_idx, (train_indices, test_indices) in enumerate(cv_strategy.split(X, y)):
        print(f"\n--- Processing Fold {fold_idx + 1}/{cv_folds} ---")
        X_train_outer, X_test_outer = X[train_indices], X[test_indices]
        y_train_outer, y_test_outer = y[train_indices], y[test_indices]

        print(f"Outer split: {len(y_train_outer)} train / {len(y_test_outer)} test samples.")
        if len(y_train_outer) == 0:
             print(f"Warning: Training set for fold {fold_idx+1} is empty. Skipping fold.")
             continue

        # 1. Setup Skorch Net for this fold
        # Use the provided fixed 'params' dictionary
        # Include an internal validation split for EarlyStopping within this fold's training data
        internal_val_split_fraction = 0.15 # e.g., use 15% of the outer train set for internal validation
        net = AugmentingNet(
            module=SimpleCNN,
            module__num_classes=NUM_CLASSES,
            criterion=nn.CrossEntropyLoss,
            optimizer=optim.Adam,
            # Pass hyperparameters directly from the 'params' dict
            lr=params.get('lr', 0.001),
            batch_size=params.get('batch_size', 32),
            max_epochs=params.get('max_epochs', 10),
            module__dropout_rate=params.get('module__dropout_rate', 0.5),
            # Fixed settings
            train_transform=train_transforms,
            valid_transform=valid_transforms,
            class_to_idx=class_to_idx,
            device=DEVICE,
            # Use skorch's internal split on the outer training data for monitoring
            train_split=ValidSplit(cv=internal_val_split_fraction, stratified=True, random_state=RANDOM_SEED + fold_idx),
            callbacks=[('early_stop', EarlyStopping(patience=5, monitor='valid_loss', load_best=True))],
            iterator_train__shuffle=True,
            verbose=1 # Show epoch progress for each fold
        )

        # 2. Train the model on the outer training set
        print(f"Starting training for fold {fold_idx + 1}...")
        net.fit(X_train_outer, y_train_outer)

        # 3. Evaluate on the outer test set for this fold
        print(f"Evaluating fold {fold_idx + 1} on its test set...")
        test_score = net.score(X_test_outer, y_test_outer)
        fold_test_scores.append(test_score)
        print(f"Fold {fold_idx + 1} Test Accuracy: {test_score:.4f}")

        # Optional: Record the best internal validation score achieved during training
        # Best score might be in history (need care due to load_best) or recalculate
        # try:
        #     best_val_epoch_data = net.history.get_best_epoch(monitor='valid_loss_best') # Check skorch syntax
        #     best_val_score = best_val_epoch_data.get('valid_acc', float('nan')) # Or re-score on internal val set
        #     fold_val_scores.append(best_val_score)
        # except Exception:
        #      fold_val_scores.append(float('nan'))


    # 4. Report final results
    print("\n--- CV for Evaluation Summary ---")
    if fold_test_scores:
        mean_acc = np.mean(fold_test_scores)
        std_acc = np.std(fold_test_scores)
        print(f"Individual Fold Test Accuracies: {[f'{s:.4f}' for s in fold_test_scores]}")
        print(f"Mean Test Accuracy across {len(fold_test_scores)} folds: {mean_acc:.4f}")
        print(f"Standard Deviation of Test Accuracy: {std_acc:.4f}")
    else:
        print("No folds were successfully processed.")

    total_time = time.time() - start_time
    print(f"\nCV for Evaluation finished in {total_time:.2f} seconds.")
    return fold_test_scores


# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Load Data (Paths and Labels)
    try:
        X_paths, y_labels, class_names, class_to_idx = load_image_paths_and_labels(DATA_DIR)
    except FileNotFoundError as e:
        print(e)
        exit()

    # 2. Define Hyperparameter Grid (for CV methods)
    param_grid = {
        'lr': [0.005, 0.001],
        'batch_size': [32],
        'max_epochs': [10],
        'module__dropout_rate': [0.4],
        # Add other optimiser params like weight_decay if needed: 'optimizer__weight_decay': [0.0001, 0.0]
    }

    # Define Single Config (for single split method)
    single_config = {
        'lr': 0.001,
        'batch_size': 32,
        'max_epochs': 20,
        'module__dropout_rate': 0.5,
    }


    # 3. Choose and Run Evaluation Strategy
    # --- Option 1: Standard GridSearch CV ---
    # print("="*30)
    # standard_cv_score, standard_test_score, best_params = run_gridsearch_cv(
    #     X=X_paths,
    #     y=y_labels,
    #     param_grid=param_grid,
    #     cv_folds=3, # Number of folds for CV within trainval set
    #     class_to_idx=class_to_idx,
    #     test_size=0.2 # Fraction of data held out for final testing
    # )
    # print(f"\nStandard CV Summary: Best CV Acc: {standard_cv_score:.4f}, Test Acc: {standard_test_score:.4f if standard_test_score else 'N/A'}")
    # print(f"Best Params Found: {best_params}")
    # print("="*30)


    # --- Option 2: Nested CV ---
    # print("="*30)
    # nested_mean_acc, nested_std_acc = run_nested_cv(
    #     X=X_paths,
    #     y=y_labels,
    #     param_grid=param_grid,
    #     inner_cv_folds=3, # Folds for hyperparameter tuning inside each outer fold
    #     outer_cv_folds=4, # Folds for evaluating generalization performance
    #     class_to_idx=class_to_idx
    # )
    # print(f"\nNested CV Summary: Mean Acc: {nested_mean_acc:.4f}, Std Dev: {nested_std_acc:.4f}")
    # print("="*30)

    # --- Option 3: Single Train/Validation/Test Split (Direct Training) ---
    print("="*30)
    single_val_acc, single_test_acc = run_single_split(
        X=X_paths, y=y_labels, params=single_config, # Use fixed config
        class_to_idx=class_to_idx, val_size=0.2, test_size=0.15
    )
    print(f"\nSingle Split Summary: Val Acc: {single_val_acc:.4f}, Test Acc: {single_test_acc:.4f if single_test_acc else 'N/A'}")
    print("="*30)

    # --- Option 4: CV for Evaluation (K-Fold Testing with Fixed Params) ---
    # print("=" * 30)
    # cv_eval_scores = run_cv_for_evaluation(
    #     X=X_paths, y=y_labels, params=single_config,  # Use fixed config
    #     cv_folds=5,  # Number of folds (each serves as test set once)
    #     class_to_idx=class_to_idx
    # )
    # # Summary is printed inside the function
    # print("=" * 30)


    print("\nExperiment finished.")