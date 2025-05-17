import logging

import torch
import torch.nn as nn

from ..config import DEFAULT_IMG_SIZE

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        """
        Simple CNN model for image classification.
        Args:
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate for the classifier.
        """
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        image_size = DEFAULT_IMG_SIZE
        height, width = image_size

        # Define layers (using example calculation for IMAGE_SIZE=64 from simplified code)
        # Make this calculation dynamic or ensure image size is passed if needed.
        # For AdaptiveAvgPool, the input size doesn't matter as much.
        img_h_after_pools = height // 8
        img_w_after_pools = width // 8

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.MaxPool2d(2), # 64->32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.MaxPool2d(2), # 32->16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(128), nn.MaxPool2d(2) # 16->8
        )
        # Use AdaptiveAvgPool2d to handle variable input sizes gracefully
        self.avgpool = nn.AdaptiveAvgPool2d((max(1, img_h_after_pools//2), max(1, img_w_after_pools//2))) # Pool down further to e.g., 4x4
        pooled_size = 128 * max(1, img_h_after_pools//2) * max(1, img_w_after_pools//2) # Calculate pooled feature size

        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, 512), # Input size depends on avgpool output
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), # Use passed dropout rate
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"SimpleCNN initialized with {num_classes} classes and image size {image_size}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
