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
                 is_hybrid_input: bool = False,
                 hybrid_in_channels: Optional[int] = None,  # e.g., 48
                 feature_map_input_channels: Optional[int] = None,
                 hybrid_target_stage0_channels: int = 96  # Swin-T's first stage expects 96 channels
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

        if self.is_hybrid_input:
            if self.hybrid_in_channels is None:
                raise ValueError("hybrid_in_channels must be specified for hybrid Swin.")

            logger.info(f"Adapting Swin '{swin_model_variant}' for hybrid input.")

            # 1. Create the new input projection layer
            # This layer takes (B, hybrid_in_channels, H_feat, W_feat)
            # and outputs (B, hybrid_target_stage0_channels, H_feat, W_feat)
            hybrid_projection_layer = nn.Conv2d(
                self.hybrid_in_channels,
                hybrid_target_stage0_channels,  # e.g., 96 for Swin-T
                kernel_size=1,
                stride=1,
                bias=True  # Can add bias
            )
            # Initialize this new layer (optional, but good practice)
            nn.init.kaiming_normal_(hybrid_projection_layer.weight, mode='fan_out', nonlinearity='relu')
            if hybrid_projection_layer.bias is not None:
                nn.init.zeros_(hybrid_projection_layer.bias)

            # 2. Reconstruct the 'features' sequential block
            # The original self.model.features is an nn.Sequential
            # self.model.features[0] is the PatchEmbed
            # self.model.features[1:] are the stages and patch merging layers

            # Keep all layers from the original 'features' except the first one (PatchEmbed)
            # and prepend our new hybrid_projection_layer.
            # IMPORTANT: The output of hybrid_projection_layer must spatially match what
            # self.model.features[1] (first Swin stage) expects.
            # For Swin-T, if input is 224x224, patch_embed (stride 4) -> 56x56.
            # So, your CNN feature extractor must output H_feat=56, W_feat=56.

            new_features_list = [hybrid_projection_layer] + list(self.model.features.children())[1:]
            self.model.features = nn.Sequential(*new_features_list)

            logger.info(
                f"Replaced Swin PatchEmbed with hybrid projection: {self.hybrid_in_channels} -> {hybrid_target_stage0_channels} channels.")

        if self.is_hybrid_input:
            if self.hybrid_in_channels is None:
                raise ValueError("hybrid_in_channels must be specified for hybrid Swin.")

            # The paper's Swin receives 56x56x48. Swin-T's first stage expects 96 channels.
            # The paper's Table 1: "Linear Embedding and Block (4x)" implies the first part
            # of their Swin setup projects 48 -> 96 channels and then has 2 blocks.
            # Standard Swin-T: features[0] is PatchEmbed (e.g., Conv2d(3, 96, kernel_size=4, stride=4))
            #                     features[1] is Stage 1 (2 blocks, operates on 56x56x96 if input is 224x224)

            # We need to replace swin_model.features[0] (PatchEmbed)
            # with a layer that takes (B, 48, 56, 56) and outputs (B, 96, 56, 56)
            # This is effectively a 1x1 convolution for channel projection.
            self.hybrid_input_projection = nn.Conv2d(self.hybrid_in_channels, 96, kernel_size=1, stride=1)
            logger.info(
                f"Swin configured for hybrid input: Replacing PatchEmbed. Projecting {self.hybrid_in_channels} -> 96 channels.")

            # The rest of the Swin stages (self.model.features[1:]) should work as is,
            # as they expect the 56x56x96 input.
            # Also, self.model.norm and self.model.avgpool, self.model.head will operate on the output of the stages.

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

        # Unfreeze the hybrid projection layer if it was added
        if self.is_hybrid_input and hasattr(self.model.features[0], 'parameters'):
            for param in self.model.features[0].parameters():  # Unfreeze our custom projection
                param.requires_grad = True
            logger.debug("Unfroze custom hybrid input projection layer.")

        if num_stages_to_unfreeze > 0:
            if hasattr(self.model, 'norm') and self.model.norm is not None:
                for param in self.model.norm.parameters():
                    param.requires_grad = True
                logger.debug("Unfroze final LayerNorm (swin_model.norm).")

            # Stage indices in self.model.features (AFTER potential modification):
            # If hybrid: features[0] = hybrid_projection
            #            features[1] = Stage 1 blocks (orig features[1])
            #            features[2] = PatchMerge for S1 (orig features[2])
            #            features[3] = Stage 2 blocks (orig features[3])
            #            ...
            # If not hybrid: features[0] = PatchEmbed
            #                features[1] = Stage 1 blocks
            #                ...

            # Map num_stages_to_unfreeze to feature indices in the *current* self.model.features
            # Stage 1 blocks are at self.model.features[1]
            # Stage 2 blocks are at self.model.features[3]
            # Stage 3 blocks are at self.model.features[5]
            # Stage 4 blocks are at self.model.features[7]
            # Associated PatchMerging layers are at indices 2, 4, 6.

            # Determine the starting index in self.model.features to unfreeze
            # based on how many *block stages* we want to unfreeze from the end.
            num_block_stages_in_features = sum(1 for i, layer in enumerate(self.model.features) if
                                               i > 0 and i % 2 == 1)  # Count layers at odd indices > 0

            effective_num_stages_to_unfreeze = min(num_stages_to_unfreeze, num_block_stages_in_features)

            unfreeze_from_feature_idx = len(self.model.features)  # Default: unfreeze nothing from features

            if effective_num_stages_to_unfreeze > 0:
                # Calculate how many actual layers (blocks + patchmerging) this corresponds to from the end
                # Unfreezing 1 stage (e.g., Stage 4) means unfreezing features[7] (blocks) and features[6] (its patch merge)
                # Unfreezing 2 stages (Stage 3,4) means features[5,6,7]

                # Find the starting index of the Nth block stage from the end
                block_stage_indices = [i for i, layer in enumerate(self.model.features) if i > 0 and i % 2 == 1]

                if effective_num_stages_to_unfreeze <= len(block_stage_indices):
                    # Start unfreezing from the patch merging layer *before* the first block stage we want to unfreeze,
                    # or from the block stage itself if it's Stage 1 (which has no preceding patch merge in the main sequence)
                    first_block_stage_to_unfreeze_idx_in_all_stages = len(
                        block_stage_indices) - effective_num_stages_to_unfreeze
                    actual_feature_index_of_first_block_stage = block_stage_indices[
                        first_block_stage_to_unfreeze_idx_in_all_stages]

                    # If this block stage is not the very first block stage (i.e., not self.model.features[1]),
                    # also unfreeze its preceding PatchMerging layer.
                    if actual_feature_index_of_first_block_stage > 1:
                        unfreeze_from_feature_idx = actual_feature_index_of_first_block_stage - 1  # Start from PatchMerge
                    else:
                        unfreeze_from_feature_idx = actual_feature_index_of_first_block_stage  # Start from Stage 1 blocks

            # If unfreezing all stages (e.g., num_stages_to_unfreeze >= 4 for Swin-T)
            # For Swin-T (4 block stages), if unfreeze >= 4, means unfreeze from original PatchEmbed (or our hybrid proj)
            if num_stages_to_unfreeze >= num_block_stages_in_features:
                unfreeze_from_feature_idx = 0 if not self.is_hybrid_input else 1  # 0 for original patch embed, 1 for first actual stage if hybrid_proj is at 0
                if self.is_hybrid_input: unfreeze_from_feature_idx = 1  # Because features[0] is already handled (hybrid_projection)

            if unfreeze_from_feature_idx < len(self.model.features):
                for i in range(unfreeze_from_feature_idx, len(self.model.features)):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
                    if i > 0 and i % 2 == 1: unfrozen_stages_count += 1  # Count actual block stages
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

        for param in self.model.head.parameters():  # Ensure new head is trainable
            param.requires_grad = True

        # Log trainable params (moved from constructor of HybridSwin to here)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"PretrainedSwin ({'hybrid' if self.is_hybrid_input else 'standard'}): "
            f"Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The forward pass now directly uses the modified self.model
        # self.model.features will use the hybrid_projection_layer if is_hybrid_input was true
        return self.model(x)
