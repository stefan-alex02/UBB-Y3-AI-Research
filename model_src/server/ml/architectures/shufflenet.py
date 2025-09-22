import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ShuffleNet_V2_X1_0_Weights

from ..logger_utils import logger


class ShuffleNetCloud(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights_arg = None
        if pretrained:
            try:
                weights_arg = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            except AttributeError:
                logger.warning("ShuffleNet_V2_X1_0_Weights enum not found, using pretrained=True boolean.")
                weights_arg = True
        else:
            logger.info("Initializing ShuffleNetCloud without ImageNet pre-trained weights.")

        if isinstance(weights_arg, bool) and weights_arg is True:
            base_model = models.shufflenet_v2_x1_0(pretrained=True)
        else:
            base_model = models.shufflenet_v2_x1_0(weights=weights_arg)

        num_ftrs = base_model.fc.in_features

        # Take all layers except the final fc layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])

        # Add the Global Average Pooling layer explicitly
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_fc = nn.Linear(num_ftrs, num_classes)

        nn.init.normal_(self.classifier_fc.weight, 0, 0.01)
        nn.init.constant_(self.classifier_fc.bias, 0)

        logger.info(f"ShuffleNetCloud (based on shufflenet_v2_x1_0) initialized. Pretrained: {pretrained}. "
                    f"Classifier head: Linear({num_ftrs}->{num_classes})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier_fc(x)
        return x
