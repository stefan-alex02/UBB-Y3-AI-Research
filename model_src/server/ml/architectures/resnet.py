import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # For clarity on weights
from typing import Optional

from ..logger_utils import logger  # Assuming logger is available


class ResNet18BasedCloud(nn.Module):
    def __init__(self,
                 num_classes: int,
                 pretrained: bool = True,
                 dropout_rate_fc: float = 0.3,
                 fc_hidden_neurons: int = 128):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
            logger.info("Initializing ResNet18BasedCloud without ImageNet pre-trained weights.")

        base_model = models.resnet18(weights=weights)

        num_ftrs = base_model.fc.in_features  # Should be 512 for ResNet18

        # Remove the original fully connected layer
        # We'll use all layers *before* the original fc layer as the feature extractor
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # New classifier head based on the script
        self.classifier_head = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the feature extractor (after global avg pooling in ResNet18)
            nn.Linear(num_ftrs, fc_hidden_neurons),
            nn.BatchNorm1d(fc_hidden_neurons),
            nn.ReLU(inplace=True),  # Added ReLU after BatchNorm, common practice
            nn.Dropout(dropout_rate_fc),
            nn.Linear(fc_hidden_neurons, num_classes)
        )

        logger.info(f"ResNet18BasedCloud initialized. Pretrained: {pretrained}. "
                    f"Classifier head: Linear({num_ftrs}->{fc_hidden_neurons})->BN->ReLU->Dropout({dropout_rate_fc})->Linear({fc_hidden_neurons}->{num_classes})")

    def forward(self, x):
        x = self.features(x)  # Output of ResNet18 before original FC is (B, 512, 1, 1) due to internal AdaptiveAvgPool2d
        x = self.classifier_head(x)
        return x
