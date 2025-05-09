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

class ModelType(str, Enum):  # Inheriting from str makes it directly usable as a string
    CNN = "cnn"
    SIMPLE_VIT = "vit"
    FLEXIBLE_VIT = "fvit"  # New type for the more advanced ViT
    DIFFUSION = "diff"

    @classmethod
    def _missing_(cls, value):  # Optional: for case-insensitive matching or aliases
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return super()._missing_(value)

    def __str__(self):  # Ensures that str(ModelType.CNN) is "cnn"
        return self.value
