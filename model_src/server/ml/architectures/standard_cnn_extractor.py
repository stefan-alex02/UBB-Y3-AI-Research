import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple, Dict

from ..logger_utils import logger


class StandardCNNFeatureExtractor(nn.Module):
    def __init__(self,
                 model_name: str = "efficientnet_b0",
                 pretrained: bool = True,
                 output_channels_target: Optional[int] = None,
                 freeze_extractor: bool = False,
                 num_frozen_stages: int = 0,
                 num_classes: Optional[int] = None
                 ):
        super().__init__()
        self.model_name = model_name
        self.output_channels_target = output_channels_target
        self._num_classes_for_standalone = num_classes

        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)
            self.features = base_model.features
            self._natural_output_channels = 1280
            self.downsample_factor = 32
        elif model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self._natural_output_channels = 512
            self.downsample_factor = 32
        elif model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            self.features = base_model.features
            self._natural_output_channels = 1280
            self.downsample_factor = 32
        else:
            raise ValueError(f"Unsupported standard CNN model_name: {model_name}")

        logger.info(
            f"Loaded {model_name} as feature extractor. ImageNet Pretrained: {pretrained}. Natural output channels: {self._natural_output_channels}")

        self.final_projection = None
        current_channels_before_head = self._natural_output_channels
        if self.output_channels_target and self.output_channels_target != self._natural_output_channels:
            self.final_projection = nn.Conv2d(self._natural_output_channels, self.output_channels_target, kernel_size=1,
                                              bias=True)
            nn.init.kaiming_normal_(self.final_projection.weight, a=0, mode='fan_out', nonlinearity='relu')
            if self.final_projection.bias is not None: nn.init.constant_(self.final_projection.bias, 0)
            current_channels_before_head = self.output_channels_target
            logger.info(
                f"Added final 1x1 projection: {self._natural_output_channels} -> {current_channels_before_head} channels.")

        self.current_output_channels = current_channels_before_head

        self.standalone_pool = None
        self.standalone_head = None
        if self._num_classes_for_standalone is not None and self._num_classes_for_standalone > 0:
            self.standalone_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.standalone_head = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(current_channels_before_head, self._num_classes_for_standalone)
            )
            logger.info(f"Added standalone classification head for {self._num_classes_for_standalone} classes.")

        if freeze_extractor:
            logger.info(f"Freezing all weights of the {model_name} backbone's 'features' part.")
            for param in self.features.parameters():
                param.requires_grad = False
        elif num_frozen_stages > 0:
            self._freeze_stages(num_frozen_stages)

    def get_parameter_counts(self) -> Dict[str, int]:
        """
        Calculates the total and trainable parameters for this module instance.

        Returns:
            A dictionary containing 'total_params' and 'trainable_params'.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_params': total_params, 'trainable_params': trainable_params}

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

        if self.standalone_head is not None:
            x_pooled = self.standalone_pool(x)
            x_flat = torch.flatten(x_pooled, 1)
            return self.standalone_head(x_flat)
        return x

    def load_fine_tuned_weights(self, path: str):
        """
        Loads weights that were fine-tuned for this StandardCNNFeatureExtractor.
        It intelligently handles cases where the standalone training might have had a
        classification head ('standalone_head') that isn't present when used as a
        pure feature extractor, and whether the 'final_projection' layer exists in both.
        """
        logger.info(
            f"Attempting to load fine-tuned weights for {self.model_name} from {path} into current instance structure.")
        try:
            loaded_full_state_dict = torch.load(path, map_location='cpu', weights_only=True)

            if isinstance(loaded_full_state_dict,
                          dict) and 'model_state_dict' in loaded_full_state_dict:
                actual_loaded_state_dict = loaded_full_state_dict['model_state_dict']
                actual_loaded_state_dict = {k.replace("module.", "", 1): v for k, v in actual_loaded_state_dict.items()}
            elif isinstance(loaded_full_state_dict,
                            dict) and 'state_dict' in loaded_full_state_dict:
                actual_loaded_state_dict = loaded_full_state_dict['state_dict']
            elif isinstance(loaded_full_state_dict, dict):
                actual_loaded_state_dict = loaded_full_state_dict
            elif isinstance(loaded_full_state_dict, nn.Module):
                actual_loaded_state_dict = loaded_full_state_dict.state_dict()
            else:
                logger.error(f"Could not extract state_dict from {path}. Found type: {type(loaded_full_state_dict)}")
                return False

            current_instance_state_dict = self.state_dict()
            new_state_dict_to_load = {}
            unexpected_keys = []
            missing_keys = list(current_instance_state_dict.keys())

            keys_loaded_count = 0
            for k_loaded, v_loaded in actual_loaded_state_dict.items():
                if k_loaded in current_instance_state_dict:
                    if current_instance_state_dict[k_loaded].shape == v_loaded.shape:
                        new_state_dict_to_load[k_loaded] = v_loaded
                        if k_loaded in missing_keys: missing_keys.remove(k_loaded)
                        keys_loaded_count += 1
                    else:
                        logger.warning(f"Shape mismatch for key {k_loaded}: "
                                       f"loaded {v_loaded.shape}, current {current_instance_state_dict[k_loaded].shape}. Skipping.")
                else:
                    if not (k_loaded.startswith("standalone_head.") or k_loaded.startswith("standalone_pool.")):
                        unexpected_keys.append(k_loaded)

            final_proj_missing = [k for k in missing_keys if k.startswith("final_projection.")]
            if final_proj_missing:
                logger.warning(f"Weights for 'final_projection' not found in checkpoint (keys: {final_proj_missing}). "
                               f"It will use its random initialization if it exists in the current model.")
                for k_fp in final_proj_missing:
                    if k_fp in missing_keys: missing_keys.remove(k_fp)

            if unexpected_keys:
                logger.warning(f"Unexpected keys in loaded state_dict (will be ignored): {unexpected_keys}")

            final_missing_keys_error = [k for k in missing_keys if
                                        not (k.startswith("standalone_head.") or k.startswith("standalone_pool."))]

            if final_missing_keys_error:
                logger.error(
                    f"CRITICAL Missing keys in current model instance not found in checkpoint (excluding standalone head/pool or handled final_projection): {final_missing_keys_error}")

            if not new_state_dict_to_load:
                logger.error(f"No compatible weights found in {path} to load into current {self.model_name} instance.")
                return False

            self.load_state_dict(new_state_dict_to_load, strict=False)
            logger.info(
                f"Successfully loaded {keys_loaded_count} matching weight tensors for {self.model_name} from {path}.")

            if final_missing_keys_error:
                raise RuntimeError(
                    f"Critical missing keys encountered during weight loading: {final_missing_keys_error}")
            return True

        except Exception as e:
            logger.error(f"Error during load_fine_tuned_weights for {self.model_name} from {path}: {e}")
            raise e
