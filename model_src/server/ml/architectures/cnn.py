import torch
import torch.nn as nn

from ..config import DEFAULT_IMG_SIZE
from ..logger_utils import logger


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

        img_h_after_pools = height // 8
        img_w_after_pools = width // 8

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.MaxPool2d(2), # 64->32
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(64), nn.MaxPool2d(2), # 32->16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(128), nn.MaxPool2d(2) # 16->8
        )
        self.avgpool = nn.AdaptiveAvgPool2d((max(1, img_h_after_pools//2), max(1, img_w_after_pools//2)))
        pooled_size = 128 * max(1, img_h_after_pools//2) * max(1, img_w_after_pools//2)

        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"SimpleCNN initialized with {num_classes} classes and image size {image_size}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
