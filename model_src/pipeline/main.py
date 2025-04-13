# Import necessary functions and libraries
from torch.utils.data import DataLoader

from model_src.pipeline.datasets_stats import plot_datasets_statistics
from model_src.pipeline.evaluate import evaluate_model_metrics
from model_src.pipeline.models import SimpleCNN
from model_src.pipeline.train import train_model
from utils import load_config
from logger import create_logger
from utils import load_device
from __datasets import load_datasets

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

    # Define a PyTorch model
    model = SimpleCNN(num_classes=len(dataset_info[-1]))

    # Train the model
    train_model(model, dataset_info, config, logger, device)

    # Move the model to the appropriate device
    model.to(device)

    # Extract the test loader and class names
    if k_folds > 1:
        _, _, _, test_dataset, class_names = dataset_info  # K-fold mode
        test_loader = DataLoader(test_dataset, batch_size=config["batch-size"], shuffle=False)
    else:
        _, _, test_loader, class_names = dataset_info  # Regular train-val-test split

    # Evaluate the model
    evaluate_model_metrics(model, test_loader, class_names, device, logger)

#catch exceptions
except Exception as e:
    if logger:
        logger.exception(e)
    raise e
