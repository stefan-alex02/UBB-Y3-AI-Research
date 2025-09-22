import timm
import torch.nn as nn
from torchvision import models

from ..logger_utils import logger


class XceptionBasedCloudNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dense_neurons1: int = 128):
        super().__init__()

        model_name_timm = 'xception'
        self.base_model_timm = timm.create_model(model_name_timm, pretrained=pretrained, num_classes=0)

        num_ftrs = self.base_model_timm.num_features

        self.features = self.base_model_timm

        self.classifier_head = nn.Sequential(
            nn.Linear(num_ftrs, dense_neurons1),
            nn.ReLU(inplace=True),
            nn.Linear(dense_neurons1, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier_head(x)
        return x


class MobileNetBasedCloudNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dense_neurons1: int = 128):
        super().__init__()

        weights_arg_mn = None
        if pretrained:
            try:
                weights_arg_mn = models.MobileNet_V2_Weights.IMAGENET1K_V1
            except AttributeError:
                logger.warning("MobileNet_V2_Weights enum not found, falling back to pretrained=True boolean.")
                weights_arg_mn = True

        if isinstance(weights_arg_mn, bool) and weights_arg_mn is True:
            base_model_mn = models.mobilenet_v2(pretrained=True)
        else:
            base_model_mn = models.mobilenet_v2(weights=weights_arg_mn)

        num_ftrs_mn = base_model_mn.classifier[1].in_features
        self.features = base_model_mn.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs_mn, dense_neurons1),
            nn.ReLU(inplace=True),
            nn.Linear(dense_neurons1, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier_head(x)
        return x
