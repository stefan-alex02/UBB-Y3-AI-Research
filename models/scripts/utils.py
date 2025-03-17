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

def load_device(logger: logging.Logger = None) -> str:
    """
    Load device (cuda or cpu)
    :param logger: Logger object
    :return: Device
    """
    is_cuda = torch.cuda.is_available()
    if logger:
        logger.info(f"Is CUDA available: {is_cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current device index: {torch.cuda.current_device()}")
        logger.info(f"Current device name: {torch.cuda.get_device_name()}\n")
    return 'cuda' if is_cuda else 'cpu'
