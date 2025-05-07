import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import logging

def train_model(
    model: nn.Module,
    dataset_info: tuple,
    config: dict,
    logger: logging.Logger = None,
    device: torch.device = None
):
    """Train a PyTorch model with either train-val-test split or K-Fold cross-validation.

    Args:
        model (nn.Module): PyTorch model to train.
        dataset_info (tuple): Output from `load_datasets()`, containing either
                              (train_loader, val_loader, test_loader, class_names) for standard split
                              or (k_folds, dataset, indices, test_dataset, class_names) for cross-validation.
        config (dict): Configuration dictionary.
        logger (logging.Logger, optional): Logger for logging messages.
        device (torch.device, optional): Device to use (CPU or CUDA).
    """
    # Load training hyperparameters
    num_epochs = config.get("num_epochs", 10)
    batch_size = config.get("batch_size", 32)
    learning_rate = config.get("learning_rate", 0.001)
    patience = config.get("early_stopping_patience", 3)  # Number of epochs to wait before stopping

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to device
    model.to(device)

    # Check if using K-Fold cross-validation
    if isinstance(dataset_info[0], int):
        k_folds, dataset, indices, test_dataset, class_names = dataset_info
        is_kfold = True
    else:
        train_loader, val_loader, test_loader, class_names = dataset_info
        is_kfold = False

    # Run either K-Fold or standard training
    if is_kfold:
        kfold_results = []
        for fold, (train_idx, val_idx) in enumerate(indices):
            logger.info(f"Starting Fold {fold + 1}/{k_folds}")

            # Create DataLoaders for current fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            # Train on current fold
            val_loss = train_loop(
                model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, logger, device
            )
            kfold_results.append(val_loss)

        logger.info(f"K-Fold Cross-Validation Complete. Avg Val Loss: {np.mean(kfold_results):.4f}")

    else:
        # Standard train-val-test split mode
        train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, logger, device)

        # Final evaluation on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        logger.info(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


def train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, logger, device):
    """Training loop with early stopping."""
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss /= total
        train_acc = correct / total

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break  # Stop training if validation loss does not improve

    return best_val_loss


def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance on a given dataset."""
    model.eval()
    loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss /= total
    accuracy = correct / total
    return loss, accuracy
