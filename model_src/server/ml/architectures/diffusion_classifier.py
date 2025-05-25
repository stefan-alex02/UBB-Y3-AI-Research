import torch
import torch.nn as nn
from torchvision import models

from ..logger_utils import logger


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