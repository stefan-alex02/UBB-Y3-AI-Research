# server/ml/architectures/standard_cnn_extractors.py
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple

from ..logger_utils import logger


class StandardCNNFeatureExtractor(nn.Module):
    def __init__(self,
                 model_name: str = "efficientnet_b0",
                 pretrained: bool = True,
                 output_channels_target: Optional[int] = None,  # If None, uses model's natural output channels
                 freeze_extractor: bool = False,  # Whether to freeze the weights of the loaded backbone
                 num_frozen_stages: int = 0
                 # 0 means fine-tune all (if not freeze_extractor), 1 means freeze first stage, etc.
                 # Interpretation of "stage" depends on model_name
                 ):
        super().__init__()
        self.model_name = model_name
        self.output_channels_target = output_channels_target

        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            self.features = base_model.features  # This is an nn.Sequential
            # EfficientNet-B0 features output 1280 channels before pooling/classifier
            self._natural_output_channels = 1280
            # EfficientNet-B0 downsamples by 32 (224 -> 7)
            self.downsample_factor = 32
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            # Use layers up to the one before avgpool and fc
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self._natural_output_channels = 512  # ResNet18 output channels before fc
            self.downsample_factor = 32
        elif model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            self.features = base_model.features  # nn.Sequential
            self._natural_output_channels = 1280  # MobileNetV2 output channels before classifier
            self.downsample_factor = 32
        else:
            raise ValueError(f"Unsupported standard CNN model_name: {model_name}")

        logger.info(
            f"Loaded {model_name} as feature extractor. Pretrained: {pretrained}. Natural output channels: {self._natural_output_channels}")

        self.final_projection = None
        if self.output_channels_target and self.output_channels_target != self._natural_output_channels:
            self.final_projection = nn.Conv2d(self._natural_output_channels, self.output_channels_target, kernel_size=1,
                                              bias=True)
            # Initialize projection
            nn.init.kaiming_normal_(self.final_projection.weight, a=0, mode='fan_out', nonlinearity='relu')
            if self.final_projection.bias is not None: nn.init.constant_(self.final_projection.bias, 0)
            self.current_output_channels = self.output_channels_target
            logger.info(
                f"Added final 1x1 projection: {self._natural_output_channels} -> {self.current_output_channels} channels.")
        else:
            self.current_output_channels = self._natural_output_channels  # Expose this for HybridViT

        # Freezing logic
        if freeze_extractor:
            logger.info(f"Freezing all weights of the {model_name} backbone.")
            for param in self.features.parameters():
                param.requires_grad = False
        elif num_frozen_stages > 0:
            self._freeze_stages(num_frozen_stages)

    def _freeze_stages(self, num_frozen_stages: int):
        if not hasattr(self.features, 'children') or not isinstance(self.features, nn.Sequential):
            logger.warning(
                f"Cannot apply num_frozen_stages to {self.model_name} as self.features is not a simple Sequential module.")
            return

        stages = list(self.features.children())
        if num_frozen_stages > len(stages):
            logger.warning(
                f"Requested to freeze {num_frozen_stages} stages, but {self.model_name} features has only {len(stages)}. Freezing all.")
            num_frozen_stages = len(stages)

        for i in range(num_frozen_stages):
            logger.info(f"Freezing stage {i} of {self.model_name} features.")
            for param in stages[i].parameters():
                param.requires_grad = False

    def get_output_spatial_size(self, input_h: int, input_w: int) -> Tuple[int, int]:
        return input_h // self.downsample_factor, input_w // self.downsample_factor

    def forward(self, x):
        x = self.features(x)
        if self.final_projection:
            x = self.final_projection(x)
        return x

    def load_fine_tuned_weights(self, path: str, strict: bool = True):
        """
        Loads weights that were fine-tuned for this specific StandardCNNFeatureExtractor instance
        (e.g., after a standalone pre-training run on cloud data).
        """
        try:
            state_dict = torch.load(path, map_location='cpu')
            actual_state_to_load = state_dict.get('state_dict', state_dict) if isinstance(state_dict,
                                                                                          dict) and 'state_dict' in state_dict else state_dict

            if isinstance(actual_state_to_load, nn.Module):  # If entire model object was saved
                actual_state_to_load = actual_state_to_load.state_dict()

            if not isinstance(actual_state_to_load, dict):
                logger.error(f"Could not extract state_dict from {path}. Found type: {type(actual_state_to_load)}")
                return

            self.load_state_dict(actual_state_to_load, strict=strict)
            logger.info(f"Successfully loaded fine-tuned weights for {self.model_name} from {path}")
        except Exception as e:
            logger.error(f"Error loading fine-tuned weights for {self.model_name} from {path}: {e}")
            # Decide whether to raise or proceed with ImageNet/random weights
            raise e
