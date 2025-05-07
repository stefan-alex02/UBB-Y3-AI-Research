import json
import logging
import os
import shutil
import torch

def load_config(config_path):
    # load as json
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def delete_directory_recursively(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

def load_device(logger: logging.Logger = None) -> torch.device:
    """
    Load device (cuda or cpu)
    :param logger: Logger object
    :return: Device
    """
    is_cuda = torch.cuda.is_available()
    if logger:
        if is_cuda:
            logger.info("CUDA is available.")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            logger.info(f"Current device index: {torch.cuda.current_device()}")
            logger.info(f"Current device name: {torch.cuda.get_device_name()}\n")
        else:
            logger.warning("CUDA is not available. Switching to CPU.")
    return torch.device("cuda" if is_cuda else "cpu")
