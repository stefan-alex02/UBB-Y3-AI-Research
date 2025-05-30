import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple

from ..logger_utils import logger


class StandardCNNFeatureExtractor(nn.Module):
    def __init__(self,
                 model_name: str = "efficientnet_b0",
                 pretrained: bool = True,  # ImageNet pretraining for the backbone
                 output_channels_target: Optional[int] = None,
                 freeze_extractor: bool = False,  # For when used *within* hybrid
                 num_frozen_stages: int = 0,  # For when used *within* hybrid
                 num_classes: Optional[int] = None  # For standalone training
                 ):
        super().__init__()
        self.model_name = model_name
        self.output_channels_target = output_channels_target
        self._num_classes_for_standalone = num_classes  # Store it

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

        self.current_output_channels = current_channels_before_head  # This is what HybridViT needs to know

        # --- Standalone Classifier Head (only if num_classes_for_standalone is provided) ---
        self.standalone_pool = None
        self.standalone_head = None
        if self._num_classes_for_standalone is not None and self._num_classes_for_standalone > 0:
            self.standalone_pool = nn.AdaptiveAvgPool2d((1, 1))
            # Head operates on `current_channels_before_head` (after optional projection)
            self.standalone_head = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),  # Add some dropout before the linear layer
                nn.Linear(current_channels_before_head, self._num_classes_for_standalone)
            )
            logger.info(f"Added standalone classification head for {self._num_classes_for_standalone} classes.")

        # Freezing logic (applies to self.features, not self.final_projection or standalone_head by default)
        if freeze_extractor:  # This flag is more for when it's *part* of the hybrid model
            logger.info(f"Freezing all weights of the {model_name} backbone's 'features' part.")
            for param in self.features.parameters():
                param.requires_grad = False
        elif num_frozen_stages > 0:
            self._freeze_stages(num_frozen_stages)
        # If not freeze_extractor and num_frozen_stages is 0, all self.features params are trainable by default.

    def _freeze_stages(self, num_frozen_stages: int):
        # ... (same as before) ...
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

        # If training/evaluating standalone, pass through the standalone head
        if self.standalone_head is not None:
            x_pooled = self.standalone_pool(x)
            x_flat = torch.flatten(x_pooled, 1)
            return self.standalone_head(x_flat)
        return x  # Otherwise, return features for the hybrid model

    def load_fine_tuned_weights(self, path: str):  # Removed strict argument, will handle it internally
        """
        Loads weights that were fine-tuned for this StandardCNNFeatureExtractor.
        It intelligently handles cases where the standalone training might have had a
        classification head ('standalone_head') that isn't present when used as a
        pure feature extractor, and whether the 'final_projection' layer exists in both.
        """
        logger.info(
            f"Attempting to load fine-tuned weights for {self.model_name} from {path} into current instance structure.")
        try:
            # Add weights_only=True for security and to avoid the FutureWarning
            loaded_full_state_dict = torch.load(path, map_location='cpu', weights_only=True)

            # If the .pt file contains a dictionary with 'state_dict' key (e.g., from Skorch checkpoint)
            if isinstance(loaded_full_state_dict,
                          dict) and 'model_state_dict' in loaded_full_state_dict:  # Common for skorch checkpoints
                actual_loaded_state_dict = loaded_full_state_dict['model_state_dict']
                # Skorch prefixes module parameters with "module."
                # We need to strip this prefix to match the keys in a standalone nn.Module
                actual_loaded_state_dict = {k.replace("module.", "", 1): v for k, v in actual_loaded_state_dict.items()}
            elif isinstance(loaded_full_state_dict,
                            dict) and 'state_dict' in loaded_full_state_dict:  # another common pattern
                actual_loaded_state_dict = loaded_full_state_dict['state_dict']
            elif isinstance(loaded_full_state_dict, dict):  # Assumed to be a raw state_dict
                actual_loaded_state_dict = loaded_full_state_dict
            elif isinstance(loaded_full_state_dict, nn.Module):  # If entire model object was saved
                actual_loaded_state_dict = loaded_full_state_dict.state_dict()
            else:
                logger.error(f"Could not extract state_dict from {path}. Found type: {type(loaded_full_state_dict)}")
                return False  # Indicate failure

            # Get state_dict of the current instance
            current_instance_state_dict = self.state_dict()
            new_state_dict_to_load = {}
            unexpected_keys = []
            missing_keys = list(current_instance_state_dict.keys())  # Start with all keys as missing

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
                        # Keep k_loaded in missing_keys if shape mismatches
                else:
                    # This key from the file is not in the current model structure
                    # This is expected for "standalone_head.*" if current instance doesn't have it.
                    if not (k_loaded.startswith("standalone_head.") or k_loaded.startswith("standalone_pool.")):
                        unexpected_keys.append(k_loaded)  # Log other unexpected keys

            # Now, handle missing keys in the current instance
            # Special case: 'final_projection' might be missing from the loaded file
            # if it wasn't used during standalone training. In this case, those weights remain initialized.
            final_proj_missing = [k for k in missing_keys if k.startswith("final_projection.")]
            if final_proj_missing:
                logger.warning(f"Weights for 'final_projection' not found in checkpoint (keys: {final_proj_missing}). "
                               f"It will use its random initialization if it exists in the current model.")
                for k_fp in final_proj_missing:
                    if k_fp in missing_keys: missing_keys.remove(k_fp)  # Don't report as error if we accept this

            if unexpected_keys:
                logger.warning(f"Unexpected keys in loaded state_dict (will be ignored): {unexpected_keys}")

            final_missing_keys_error = [k for k in missing_keys if
                                        not (k.startswith("standalone_head.") or k.startswith("standalone_pool."))]

            if final_missing_keys_error:
                logger.error(
                    f"CRITICAL Missing keys in current model instance not found in checkpoint (excluding standalone head/pool or handled final_projection): {final_missing_keys_error}")
                # This is a more serious issue.

            if not new_state_dict_to_load:
                logger.error(f"No compatible weights found in {path} to load into current {self.model_name} instance.")
                return False

            # Load the prepared state_dict, strict=False allows missing keys (like standalone_head
            # if current instance doesn't have it, or final_projection if loaded doesn't have it)
            # and ignores unexpected keys (like standalone_head if current doesn't have it).
            self.load_state_dict(new_state_dict_to_load, strict=False)
            logger.info(
                f"Successfully loaded {keys_loaded_count} matching weight tensors for {self.model_name} from {path}.")

            if final_missing_keys_error:
                # Re-iterate the critical error if some essential parts were still missing
                raise RuntimeError(
                    f"Critical missing keys encountered during weight loading: {final_missing_keys_error}")
            return True

        except Exception as e:
            logger.error(f"Error during load_fine_tuned_weights for {self.model_name} from {path}: {e}")
            # Propagate the error so the calling code (HybridViT) knows loading failed.
            raise e  # Or return False and let HybridViT handle it
