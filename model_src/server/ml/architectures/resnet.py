import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from ..logger_utils import logger


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

        num_ftrs = base_model.fc.in_features

        self.features = nn.Sequential(*list(base_model.children())[:-1])

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, fc_hidden_neurons),
            nn.BatchNorm1d(fc_hidden_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(fc_hidden_neurons, num_classes)
        )

        logger.info(f"ResNet18BasedCloud initialized. Pretrained: {pretrained}. "
                    f"Classifier head: Linear({num_ftrs}->{fc_hidden_neurons})->BN->ReLU->Dropout({dropout_rate_fc})->Linear({fc_hidden_neurons}->{num_classes})")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier_head(x)
        return x
