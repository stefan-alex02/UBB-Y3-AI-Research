import random
from enum import Enum

import numpy as np
import torch

# --- Default Parameters ---
DEFAULT_IMG_SIZE = (224, 224) # Default image size for the model

# --- Global Configurations ---
NUM_WORKERS = 0 # Set to 0 for stability with image loading, especially on Windows
logger_name_global = 'ImgClassPipe' # Define the name once

# --- Seed Initialization ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED) # Ensure random module is also seeded

# Commenting these out might sometimes resolve unrelated CUDA errors, but reduces reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True # This ensures that the results are reproducible
torch.backends.cudnn.benchmark = False   # This is set to False to ensure reproducibility, but may affect performance

# --- Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- ANSI Color Codes ---
class LogColors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

# Dataset dictionary
DATASET_DICT = {
    'GCD': "data/GCD",
    'mGCD': "data/mini-GCD",
    'mGCDf': "data/mini-GCD-flat",
    'swimcat': "data/Swimcat-extend",
    'ccsn': "data/CCSN",
}

class AugmentationStrategy(str, Enum):
    DEFAULT_STANDARD = "default_standard" # Your current versatile one
    SKY_ONLY_ROTATION = "sky_only_rotation" # Allows full rotation, good for GCD/Swimcat
    CCSN_MODERATE = "ground_aware_no_rotation" # For CCSN - no vertical flips/major rotations
    NO_AUGMENTATION = "no_augmentation" # Just resize, ToTensor, Normalize
    PAPER_GCD = "paper_replication_gcd" # Specific to GCD paper replication
    PAPER_CCSN = "paper_replication_ccsn" # Specific to CCSN paper replication
    # You can add more named strategies here
