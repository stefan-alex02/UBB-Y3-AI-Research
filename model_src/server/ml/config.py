import random
from enum import Enum

import numpy as np
import torch

# Default Parameters
DEFAULT_IMG_SIZE = (224, 224)

# Global Configurations
NUM_WORKERS = 0
logger_name_global = 'ImgClassPipe'

# Seed Initialization
RANDOM_SEED = 42

def apply_random_seed(seed: int = RANDOM_SEED):
    """Applies a fixed random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ANSI Color Codes
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
    'GCD': "GCD",
    'GCDf': "GCD-flat",
    'mGCD': "mini-GCD",
    'mGCDf': "mini-GCD-flat",
    'swimcat': "Swimcat-extend",
    'ccsn': "CCSN",
}

class AugmentationStrategy(str, Enum):
    DEFAULT_STANDARD = "default_standard"
    SKY_ONLY_ROTATION = "sky_only_rotation"
    CCSN_MODERATE = "ground_aware_no_rotation"
    SWIMCAT_MILD = "swimcat_mild"
    NO_AUGMENTATION = "no_augmentation"
    PAPER_GCD = "paper_replication_gcd"
    PAPER_CCSN = "paper_replication_ccsn"
