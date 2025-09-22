import logging
import sys
import os
import random
import numpy as np
import torch
from typing import Optional

LOG_FILE = 'pipeline.log'


def setup_logger(log_file: str = LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """Sets up the logger for console and file."""
    logger = logging.getLogger('ClassificationPipelineLogger')

    # Prevent setting up handlers multiple times if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Format
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_format)
    logger.addHandler(stdout_handler)

    # File Handler (UTF-8)
    # Ensure results directory exists if logging there
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.propagate = False  # Prevent root logger from handling messages again

    # Add emojis for levels (optional, requires terminal support)
    logging.addLevelName(logging.DEBUG, "🐛 DEBUG")
    logging.addLevelName(logging.INFO, "✨ INFO")
    logging.addLevelName(logging.WARNING, "⚠️ WARNING")
    logging.addLevelName(logging.ERROR, "❌ ERROR")
    logging.addLevelName(logging.CRITICAL, "💥 CRITICAL")

    return logger


def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms are used where available
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize logger globally (can be accessed by importing utils)
logger = setup_logger()
