# Import necessary functions and libraries
from utils import load_config
from logger import create_logger
from utils import load_device
from datasets import load_datasets, plot_datasets_statistics, show_sample_images

# Define configurations
config = load_config("config.json")

# Load logger
logger = create_logger()

try:
    # Load device (cuda or cpu)
    device = load_device(logger=logger)

    # Configure dataset parameters
    k_folds = config["dataset"].get("k-folds", 1)
    fold = 1 if k_folds > 1 else None

    # Load datasets
    dataset_info = load_datasets(config, k_folds=k_folds, logger=logger)
    plot_datasets_statistics(dataset_info, fold=fold)

    train_loader, val_loader, test_loader, class_names = dataset_info

    # Get tensors from the dataset

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models

    # Define a basic model (e.g., a simple CNN)
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.features = models.resnet18(pretrained=True)
            self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

        def forward(self, x):
            x = self.features(x)
            return x

    # Initialize the model, loss function, and optimizer
    num_classes = len(class_names)
    model = SimpleCNN(num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Basic model training function
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                # Move data to the device (GPU or CPU)
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

            # Validation phase
            model.eval()
            val_loss = 0.0
            corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move data to the device (GPU or CPU)
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    corrects += torch.sum(preds == labels.data)

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = corrects.double() / len(val_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    # Train the model
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

#catch exceptions
except Exception as e:
    if logger:
        logger.exception(e)
    raise e
