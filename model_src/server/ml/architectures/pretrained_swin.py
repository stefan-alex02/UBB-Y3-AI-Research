from typing import Optional, Callable

import torch
import torch.nn as nn
from torchvision import models

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
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PretrainedSwin(nn.Module):
    def __init__(self,
                 num_classes: int,
                 swin_model_variant: str = 'swin_t',
                 pretrained: bool = True,
                 num_stages_to_unfreeze: int = 1,
                 head_dropout_rate: float = 0.0,
                 input_is_feature_map: bool = False,
                 is_hybrid_input: bool = False,
                 hybrid_in_channels: Optional[int] = None,
                 feature_map_input_channels: Optional[int] = None,
                 hybrid_target_stage0_channels: int = 96
                 ):
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        self.input_is_feature_map = input_is_feature_map
        self.feature_map_input_channels = feature_map_input_channels
        self.is_hybrid_input = is_hybrid_input
        self.hybrid_in_channels = hybrid_in_channels

        logger.debug(f"Initializing PretrainedSwinTransformer:")
        logger.debug(f"  Model Variant: {swin_model_variant}, Pretrained: {pretrained}")
        logger.debug(f"  Num Stages to Unfreeze: {num_stages_to_unfreeze}")
        logger.debug(f"  Head Dropout: {head_dropout_rate}")
        if self.input_is_feature_map:
            logger.debug(f"  Input Mode: Feature Map (Channels: {self.feature_map_input_channels})")
        else:
            logger.debug(f"  Input Mode: Raw Image (3 Channels assumed)")

        weights_arg = None
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

        if self.is_hybrid_input:
            if self.hybrid_in_channels is None:
                raise ValueError("hybrid_in_channels must be specified for hybrid Swin.")

            logger.info(f"Adapting Swin '{swin_model_variant}' for hybrid input.")

            hybrid_projection_layer = nn.Conv2d(
                self.hybrid_in_channels,
                hybrid_target_stage0_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )
            nn.init.kaiming_normal_(hybrid_projection_layer.weight, mode='fan_out', nonlinearity='relu')
            if hybrid_projection_layer.bias is not None:
                nn.init.zeros_(hybrid_projection_layer.bias)

            new_features_list = [hybrid_projection_layer] + list(self.model.features.children())[1:]
            self.model.features = nn.Sequential(*new_features_list)

            logger.info(
                f"Replaced Swin PatchEmbed with hybrid projection: {self.hybrid_in_channels} -> {hybrid_target_stage0_channels} channels.")

        if self.is_hybrid_input:
            if self.hybrid_in_channels is None:
                raise ValueError("hybrid_in_channels must be specified for hybrid Swin.")

            self.hybrid_input_projection = nn.Conv2d(self.hybrid_in_channels, 96, kernel_size=1, stride=1)
            logger.info(
                f"Swin configured for hybrid input: Replacing PatchEmbed. Projecting {self.hybrid_in_channels} -> 96 channels.")

        if self.input_is_feature_map:
            if self.feature_map_input_channels is None:
                raise ValueError("feature_map_input_channels must be specified if input_is_feature_map is True.")

            try:
                original_patch_embed_sequential = swin_model.features[0]
                if not isinstance(original_patch_embed_sequential, nn.Sequential) or len(
                        original_patch_embed_sequential) < 1:
                    raise TypeError("Expected swin_model.features[0] to be nn.Sequential for PatchEmbed.")

                original_proj_conv = original_patch_embed_sequential[0]
                if not isinstance(original_proj_conv, nn.Conv2d):
                    raise TypeError(
                        f"Expected first child of PatchEmbed Sequential to be nn.Conv2d, got {type(original_proj_conv)}")

                target_embed_dim = original_proj_conv.out_channels

                norm_layer_for_fm_embed = None
                if len(original_patch_embed_sequential) > 1 and isinstance(original_patch_embed_sequential[-1],
                                                                           nn.LayerNorm):
                    norm_layer_for_fm_embed = nn.LayerNorm
                    logger.debug(f"Original PatchEmbed used LayerNorm. Will include in new FeatureMapEmbed.")

                logger.info(
                    f"Adapting Swin model for feature map input: {self.feature_map_input_channels} channels -> {target_embed_dim} channels for sequence.")

                swin_model.features[0] = FeatureMapEmbed(
                    in_channels=self.feature_map_input_channels,
                    embed_dim=target_embed_dim,
                    norm_layer=norm_layer_for_fm_embed
                )
                logger.info(f"Replaced Swin PatchEmbed (Sequential) with custom FeatureMapEmbed.")

            except Exception as e_adapt:
                logger.error(f"Failed to adapt Swin input stage for feature maps: {e_adapt}", exc_info=True)
                raise RuntimeError("Swin input adaptation failed.") from e_adapt

        for param in swin_model.parameters():
            param.requires_grad = False

        unfrozen_stages_count = 0

        # Unfreeze hybrid projection layer
        if self.is_hybrid_input and hasattr(self.model.features[0], 'parameters'):
            for param in self.model.features[0].parameters():  # Unfreeze our custom projection
                param.requires_grad = True
            logger.debug("Unfroze custom hybrid input projection layer.")

        if num_stages_to_unfreeze > 0:
            if hasattr(self.model, 'norm') and self.model.norm is not None:
                for param in self.model.norm.parameters():
                    param.requires_grad = True
                logger.debug("Unfroze final LayerNorm (swin_model.norm).")

            num_block_stages_in_features = sum(1 for i, layer in enumerate(self.model.features) if
                                               i > 0 and i % 2 == 1)

            effective_num_stages_to_unfreeze = min(num_stages_to_unfreeze, num_block_stages_in_features)

            unfreeze_from_feature_idx = len(self.model.features)

            if effective_num_stages_to_unfreeze > 0:
                block_stage_indices = [i for i, layer in enumerate(self.model.features) if i > 0 and i % 2 == 1]

                if effective_num_stages_to_unfreeze <= len(block_stage_indices):
                    first_block_stage_to_unfreeze_idx_in_all_stages = len(
                        block_stage_indices) - effective_num_stages_to_unfreeze
                    actual_feature_index_of_first_block_stage = block_stage_indices[
                        first_block_stage_to_unfreeze_idx_in_all_stages]

                    if actual_feature_index_of_first_block_stage > 1:
                        unfreeze_from_feature_idx = actual_feature_index_of_first_block_stage - 1
                    else:
                        unfreeze_from_feature_idx = actual_feature_index_of_first_block_stage

            if num_stages_to_unfreeze >= num_block_stages_in_features:
                unfreeze_from_feature_idx = 0 if not self.is_hybrid_input else 1
                if self.is_hybrid_input: unfreeze_from_feature_idx = 1

            if unfreeze_from_feature_idx < len(self.model.features):
                for i in range(unfreeze_from_feature_idx, len(self.model.features)):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
                    if i > 0 and i % 2 == 1: unfrozen_stages_count += 1
                logger.info(f"Unfroze last {unfrozen_stages_count} Swin block stages (and associated patch merging).")

        # Replace the head
        if hasattr(self.model, 'head') and isinstance(self.model.head, nn.Linear):
            original_in_features = self.model.head.in_features
            new_head_layers = []
            if head_dropout_rate > 0.0:
                new_head_layers.append(nn.Dropout(head_dropout_rate))
            new_head_layers.append(nn.Linear(original_in_features, num_classes))
            self.model.head = nn.Sequential(*new_head_layers)
            logger.debug(f"Replaced Swin head for {num_classes} classes (dropout: {head_dropout_rate}).")
        else:
            logger.error("Could not find or replace the head of the Swin Transformer.")

        for param in self.model.head.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"PretrainedSwin ({'hybrid' if self.is_hybrid_input else 'standard'}): "
            f"Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
