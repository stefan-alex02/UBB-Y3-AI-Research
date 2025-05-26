import logging
from typing import Optional, List, Callable

import torch
import torch.nn as nn
from torchvision import models

# Use your project's logger
from ..logger_utils import logger


class FeatureMapEmbed(nn.Module):
    """
    Embeds a feature map (B, C_in, H, W) into a sequence (B, H*W, C_out)
    for Swin Transformer, similar to PatchEmbed but with 1x1 conv.
    """
    def __init__(self, in_channels: int, embed_dim: int, norm_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x) # (B, embed_dim, H, W)
        # Swin's PatchEmbed.forward also has: x = self.norm(x) after projection
        # Then: x.flatten(2).transpose(1, 2)  # B C H W -> B C N -> B N C
        x = x.flatten(2).transpose(1, 2) # (B, H*W, embed_dim)
        x = self.norm(x) # Apply norm after transpose like official Swin PatchEmbed if LayerNorm
        return x


class PretrainedSwin(nn.Module):
    def __init__(self,
                 num_classes: int,
                 swin_model_variant: str = 'swin_t',
                 pretrained: bool = True,
                 num_stages_to_unfreeze: int = 1,
                 head_dropout_rate: float = 0.0,
                 # New parameters for feature map input
                 input_is_feature_map: bool = False,
                 feature_map_input_channels: Optional[int] = None
                 ):
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        self.input_is_feature_map = input_is_feature_map
        self.feature_map_input_channels = feature_map_input_channels

        logger.debug(f"Initializing PretrainedSwinTransformer:")
        logger.debug(f"  Model Variant: {swin_model_variant}, Pretrained: {pretrained}")
        logger.debug(f"  Num Stages to Unfreeze: {num_stages_to_unfreeze}")
        logger.debug(f"  Head Dropout: {head_dropout_rate}")
        if self.input_is_feature_map:
            logger.debug(f"  Input Mode: Feature Map (Channels: {self.feature_map_input_channels})")
        else:
            logger.debug(f"  Input Mode: Raw Image (3 Channels assumed)")

        weights_arg = None  # Copied from your original
        if pretrained:
            if swin_model_variant == 'swin_t':
                weights_arg = models.Swin_T_Weights.IMAGENET1K_V1
            elif swin_model_variant == 'swin_s':
                weights_arg = models.Swin_S_Weights.IMAGENET1K_V1
            elif swin_model_variant == 'swin_b':
                weights_arg = models.Swin_B_Weights.IMAGENET1K_V1
            elif swin_model_variant == 'swin_v2_t':
                weights_arg = models.Swin_V2_T_Weights.IMAGENET1K_V1
            else:
                logger.warning(
                    f"No default weights enum for {swin_model_variant}. Using 'DEFAULT'.");
                weights_arg = "DEFAULT"

        try:
            swin_model_fn = getattr(models, swin_model_variant)
            swin_model = swin_model_fn(weights=weights_arg, progress=True, dropout=0.0)
        except AttributeError:
            raise ValueError(f"Unsupported Swin model variant: {swin_model_variant}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Swin '{swin_model_variant}': {e}")
        logger.debug(f"Loaded {swin_model_variant} from torchvision.")

        # --- Adapt input stage if input_is_feature_map ---
        if self.input_is_feature_map:
            if self.feature_map_input_channels is None:
                raise ValueError("feature_map_input_channels must be specified if input_is_feature_map is True.")

            try:
                # Get the original PatchEmbed module (which is an nn.Sequential)
                original_patch_embed_sequential = swin_model.features[0]
                if not isinstance(original_patch_embed_sequential, nn.Sequential) or len(
                        original_patch_embed_sequential) < 1:
                    raise TypeError("Expected swin_model.features[0] to be nn.Sequential for PatchEmbed.")

                # The projection Conv2d is the first child
                original_proj_conv = original_patch_embed_sequential[0]
                if not isinstance(original_proj_conv, nn.Conv2d):
                    raise TypeError(
                        f"Expected first child of PatchEmbed Sequential to be nn.Conv2d, got {type(original_proj_conv)}")

                target_embed_dim = original_proj_conv.out_channels  # This should be 96 for Swin-T

                # Check for LayerNorm in the original PatchEmbed Sequential
                norm_layer_for_fm_embed = None
                # The LayerNorm is typically the last element in this Sequential PatchEmbed
                if len(original_patch_embed_sequential) > 1 and isinstance(original_patch_embed_sequential[-1],
                                                                           nn.LayerNorm):
                    norm_layer_for_fm_embed = nn.LayerNorm
                    logger.debug(f"Original PatchEmbed used LayerNorm. Will include in new FeatureMapEmbed.")

                logger.info(
                    f"Adapting Swin model for feature map input: {self.feature_map_input_channels} channels -> {target_embed_dim} channels for sequence.")

                # Replace the entire original PatchEmbed Sequential (swin_model.features[0])
                # with our custom FeatureMapEmbed module.
                swin_model.features[0] = FeatureMapEmbed(
                    in_channels=self.feature_map_input_channels,
                    embed_dim=target_embed_dim,
                    norm_layer=norm_layer_for_fm_embed
                )
                logger.info(f"Replaced Swin PatchEmbed (Sequential) with custom FeatureMapEmbed.")

            except Exception as e_adapt:
                logger.error(f"Failed to adapt Swin input stage for feature maps: {e_adapt}", exc_info=True)
                raise RuntimeError("Swin input adaptation failed.") from e_adapt

        # Freeze all parameters initially (AFTER potential modification of patch_embed)
        for param in swin_model.parameters():
            param.requires_grad = False

        # --- Unfreezing logic (ensure it uses the potentially modified swin_model) ---
        unfrozen_stages_count = 0
        # ... (your existing unfreezing logic for num_stages_to_unfreeze)
        # This logic should now correctly apply to the modified swin_model.
        # If num_stages_to_unfreeze >= 4 (or unfreeze_up_to_feature_idx == 0),
        # it will unfreeze our new FeatureMapEmbed (swin_model.features[0]) as well.

        if num_stages_to_unfreeze > 0:
            # Unfreeze the final LayerNorm before the head
            if hasattr(swin_model, 'norm') and swin_model.norm is not None:
                for param in swin_model.norm.parameters():
                    param.requires_grad = True
                logger.debug("Unfroze final LayerNorm (swin_model.norm).")

            unfreeze_up_to_feature_idx = -1
            if num_stages_to_unfreeze >= 4:
                unfreeze_up_to_feature_idx = 0
            elif num_stages_to_unfreeze == 3:
                unfreeze_up_to_feature_idx = 2
            elif num_stages_to_unfreeze == 2:
                unfreeze_up_to_feature_idx = 4
            elif num_stages_to_unfreeze == 1:
                unfreeze_up_to_feature_idx = 6

            if unfreeze_up_to_feature_idx != -1:
                # The first element swin_model.features[0] is now our FeatureMapEmbed
                # The stages with blocks are at indices 1, 3, 5, 7
                for i in range(unfreeze_up_to_feature_idx, len(swin_model.features)):
                    for param in swin_model.features[i].parameters():
                        param.requires_grad = True
                # Logging the number of unfrozen stages more directly from user input
                logger.info(f"Unfroze approximately last {num_stages_to_unfreeze} Swin stages "
                            f"(and associated patch merging/embedding layers starting from Swin's internal feature index {unfreeze_up_to_feature_idx}).")

        # Replace the head for the new number of classes
        if hasattr(swin_model, 'head') and isinstance(swin_model.head, nn.Linear):
            original_in_features = swin_model.head.in_features
            if head_dropout_rate > 0.0:
                swin_model.head = nn.Sequential(
                    nn.Dropout(head_dropout_rate),
                    nn.Linear(original_in_features, num_classes)
                )
                logger.debug(
                    f"Replaced Swin head with Dropout ({head_dropout_rate}) + Linear for {num_classes} classes.")
            else:
                swin_model.head = nn.Linear(original_in_features, num_classes)
                logger.debug(f"Replaced Swin head with Linear for {num_classes} classes.")
        else:
            logger.error(
                "Could not find or replace the head of the Swin Transformer. Model may not work correctly.")

        for param in swin_model.head.parameters():  # Ensure new head is trainable
            param.requires_grad = True

        self.model = swin_model
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"PretrainedSwin: Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
