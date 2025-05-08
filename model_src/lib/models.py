import torch
import torch.nn as nn
from torchvision import models

from .config import DEFAULT_IMG_SIZE
from .logger_utils import logger


# --- Model Definitions

# --- SimpleCNN ---
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

# --- SimpleViT (Keep as before, uses models.vit_b_16) ---
class SimpleViT(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        logger.debug("Loading pre-trained vit_b_16 model...")
        vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        logger.debug("Pre-trained model loaded.")
        original_hidden_dim = vit_model.heads.head.in_features
        vit_model.heads.head = nn.Linear(original_hidden_dim, self.num_classes)
        logger.debug(f"Replaced ViT head for {num_classes} classes.")

        num_layers_to_unfreeze = 4
        total_params = len(list(vit_model.parameters()))
        unfrozen_count = 0
        for i, param in enumerate(vit_model.parameters()):
            if i < total_params - num_layers_to_unfreeze:
                 param.requires_grad = False
            else:
                 param.requires_grad = True
                 unfrozen_count += 1
        logger.info(f"SimpleViT: Froze layers, unfroze last {unfrozen_count} parameter groups.")
        self.model = vit_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --- DiffusionClassifier (Keep as before, uses models.resnet50) ---
class DiffusionClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout_rate: float = 0.4): # Add dropout rate
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        logger.debug("Loading pre-trained ResNet50 backbone...")
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        logger.debug("Pre-trained ResNet50 loaded.")
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet

        num_layers_to_unfreeze = 5
        total_params = len(list(self.backbone.parameters()))
        unfrozen_count = 0
        for i, param in enumerate(self.backbone.parameters()):
            if i < total_params - num_layers_to_unfreeze: param.requires_grad = False
            else: param.requires_grad = True; unfrozen_count+=1
        logger.info(f"DiffusionClassifier (ResNet50): Froze layers, unfroze last {unfrozen_count} backbone param groups.")

        self.diffusion_head = nn.Sequential(
            nn.Linear(in_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), # Use passed rate
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), # Use passed rate
            nn.Linear(512, self.num_classes)
        )
        logger.debug(f"DiffusionClassifier initialized with ResNet50 backbone and custom head for {num_classes} classes.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.diffusion_head(features)
        return logits
