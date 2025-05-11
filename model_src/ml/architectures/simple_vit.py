import torch
import torch.nn as nn
from torchvision import models

from ..logger_utils import logger


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
