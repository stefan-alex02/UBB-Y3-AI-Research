# server/ml/architectures/paper_xception_mobilenet.py
import torch
import torch.nn as nn
# from torchvision import models # Keep for MobileNet if you still want it from torchvision
import timm
from typing import Optional
from ..logger_utils import logger  # Assuming logger is available


class XceptionBasedCloudNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dense_neurons1: int = 128):
        super().__init__()

        # Use timm to create the Xception model
        # List of Xception variants in timm: https://huggingface.co/docs/timm/models?filter=xception
        # 'xception' is a common one, 'xception41', 'xception65', 'xception71' also exist.
        # Let's use the standard 'xception'.
        model_name_timm = 'xception'
        self.base_model_timm = timm.create_model(model_name_timm, pretrained=pretrained, num_classes=0)
        # num_classes=0 or drop_classifier=True removes the original classifier head.
        # The features method will then be the output of the conv backbone.

        num_ftrs = self.base_model_timm.num_features  # timm models usually have this attribute

        # The self.base_model_timm is now the feature extractor
        self.features = self.base_model_timm

        self.classifier_head = nn.Sequential(
            # nn.Flatten(), # timm models with num_classes=0 often output (B, C, H, W) or (B, C) after pooling
            # If it's (B,C) after global pooling inside timm model, Flatten isn't needed.
            # If it's (B,C,H,W), then AdaptiveAvgPool2d + Flatten is needed.
            # Xception in timm usually includes global pooling.
            nn.Linear(num_ftrs, dense_neurons1),
            nn.ReLU(inplace=True),
            nn.Linear(dense_neurons1, num_classes)
        )

    def forward(self, x):
        x = self.features(x)  # This will be (B, num_features) after timm's internal pooling
        # If x is (B, C, H, W) here, you'd need:
        # x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1)).flatten(1)
        x = self.classifier_head(x)
        return x


# MobileNetBasedCloudNet can remain as is if torchvision.models.mobilenet_v2 works for you
# Or you could also switch it to use timm.create_model('mobilenetv2_100', pretrained=True, num_classes=0)
# for consistency.
# ... (MobileNetBasedCloudNet code) ...
from torchvision import models  # For MobileNet


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
