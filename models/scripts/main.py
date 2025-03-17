# Define configurations
from utils import load_config
config = load_config("config.json")

# Load logger
from logger import create_logger
logger = create_logger()

# Load device (cuda or cpu)
from utils import load_device
device = load_device(logger)

# Load datasets
from datasets import load_datasets
train_loader, val_loader, test_loader, _ = load_datasets(config, logger)
