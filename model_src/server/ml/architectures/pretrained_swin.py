import logging
from typing import Optional, List

import torch
import torch.nn as nn
from torchvision import models

# Use your project's logger
try:
    from ...logger_utils import logger  # Adjust path based on your structure
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)


class PretrainedSwinTransf(nn.Module):
    def __init__(self,
                 num_classes: int,
                 swin_model_variant: str = 'swin_t',  # e.g., swin_t, swin_s, swin_b, swin_v2_t etc.
                 pretrained: bool = True,
                 # Unfreezing strategy for Swin typically means unfreezing later STAGES
                 # A "stage" in Swin consists of patch merging + multiple SwinTransformerBlocks
                 # Swin models usually have 4 stages.
                 num_stages_to_unfreeze: int = 1,  # Unfreeze last N stages (0 means only head)
                 # Stage 0 is patch_embed, Stage 1-4 are Swin blocks
                 head_dropout_rate: float = 0.0
                 ):
        """
        Wrapper for Pretrained Swin Transformers from torchvision.

        Args:
            num_classes (int): Number of output classes.
            swin_model_variant (str): Variant like 'swin_t', 'swin_s', 'swin_b',
                                      'swin_v2_t', 'swin_v2_s', 'swin_v2_b'.
            pretrained (bool): If True, loads ImageNet pre-trained weights.
            num_stages_to_unfreeze (int): Number of final stages to unfreeze.
                                       0: Only head is trainable.
                                       1: Last stage + head.
                                       2: Last two stages + head, etc.
                                       Max is usually 4 for main stages.
            head_dropout_rate (float): Dropout rate for the classification head.
        """
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        logger.debug(f"Initializing PretrainedSwinTransformer:")
        logger.debug(f"  Model Variant: {swin_model_variant}, Pretrained: {pretrained}")
        logger.debug(f"  Num Stages to Unfreeze: {num_stages_to_unfreeze}")
        logger.debug(f"  Head Dropout: {head_dropout_rate}")

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
            # Add more Swin V1/V2 variants as needed from torchvision.models
            else:
                logger.warning(
                    f"No default weights enum for {swin_model_variant}. Using 'DEFAULT'."); weights_arg = "DEFAULT"

        try:
            swin_model_fn = getattr(models, swin_model_variant)
            swin_model = swin_model_fn(weights=weights_arg, progress=True,
                                       dropout=head_dropout_rate)  # torchvision Swin takes dropout for head
        except AttributeError:
            raise ValueError(f"Unsupported Swin model variant: {swin_model_variant}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Swin '{swin_model_variant}': {e}")
        logger.debug(f"Loaded {swin_model_variant} from torchvision.")

        # Freeze all parameters initially
        for param in swin_model.parameters():
            param.requires_grad = False

        # Unfreeze layers based on num_stages_to_unfreeze
        # Swin structure: features (patch_embed + stages), norm (final layernorm), head
        # features is nn.Sequential:
        #   features[0] = PatchEmbed
        #   features[1] = Stage 1 (Sequence of SwinTransformerBlocks)
        #   features[2] = PatchMerging for Stage 1 output
        #   features[3] = Stage 2
        #   features[4] = PatchMerging for Stage 2 output
        #   features[5] = Stage 3
        #   features[6] = PatchMerging for Stage 3 output
        #   features[7] = Stage 4
        # Typical Swin has 4 main stages of blocks.
        # A "stage" of blocks is at indices 1, 3, 5, 7 in swin_model.features.
        # The patch merging layers are at 2, 4, 6.

        unfrozen_stages_count = 0
        total_block_stages = 4  # Swin-T/S/B have 4 stages with blocks

        if num_stages_to_unfreeze > 0:
            # Unfreeze the final LayerNorm before the head
            if hasattr(swin_model, 'norm') and swin_model.norm is not None:
                for param in swin_model.norm.parameters():
                    param.requires_grad = True
                logger.debug("Unfroze final LayerNorm (swin_model.norm).")

            # Unfreeze specified number of final block stages (and their preceding patch merging if applicable)
            # Stage indices in swin_model.features: Stage 1 (idx 1), Stage 2 (idx 3), Stage 3 (idx 5), Stage 4 (idx 7)
            # Patch Merging indices: idx 2 (after Stage 1), idx 4 (after Stage 2), idx 6 (after Stage 3)

            # Map num_stages_to_unfreeze to feature indices
            # Unfreezing 1 stage means unfreezing Stage 4 (features[7])
            # Unfreezing 2 stages means unfreezing Stage 3 (features[5]), its PatchMerging (features[6]), and Stage 4 (features[7])

            # Correct mapping of stages and their associated patch merging layers to unfreeze
            # If unfreezing stage X, also unfreeze its preceding patch merging layer (if not stage 1)
            # Stage 1: features[0] (PatchEmbed), features[1] (Blocks)
            # Stage 2: features[2] (PatchMerge), features[3] (Blocks)
            # Stage 3: features[4] (PatchMerge), features[5] (Blocks)
            # Stage 4: features[6] (PatchMerge), features[7] (Blocks)

            unfreeze_up_to_feature_idx = -1  # Default to only head

            if num_stages_to_unfreeze >= 4:  # Unfreeze all stages + patch_embed
                unfreeze_up_to_feature_idx = 0  # Start from patch_embed
            elif num_stages_to_unfreeze == 3:  # Unfreeze stages 2,3,4 + their patch merges
                unfreeze_up_to_feature_idx = 2  # Start from patch_merge before stage 2
            elif num_stages_to_unfreeze == 2:  # Unfreeze stages 3,4 + their patch merges
                unfreeze_up_to_feature_idx = 4  # Start from patch_merge before stage 3
            elif num_stages_to_unfreeze == 1:  # Unfreeze stage 4 + its patch merge
                unfreeze_up_to_feature_idx = 6  # Start from patch_merge before stage 4

            if unfreeze_up_to_feature_idx != -1:
                for i in range(unfreeze_up_to_feature_idx, len(swin_model.features)):
                    for param in swin_model.features[i].parameters():
                        param.requires_grad = True
                    if i % 2 == 1: unfrozen_stages_count += 1  # Count actual block stages (at odd indices)
                logger.info(
                    f"Unfroze last {unfrozen_stages_count} Swin stages (and associated patch merging/embedding layers).")

        # Replace the head for the new number of classes
        if hasattr(swin_model, 'head') and isinstance(swin_model.head, nn.Linear):
            original_in_features = swin_model.head.in_features
            swin_model.head = nn.Linear(original_in_features, num_classes)
            logger.debug(f"Replaced Swin head for {num_classes} classes. Original in_features: {original_in_features}")
        else:
            logger.error("Could not find or replace the head of the Swin Transformer. Model may not work correctly.")
            # Fallback: try to add a new head if 'head' attribute doesn't exist as expected
            # This requires knowing the output dimension of swin_model.norm
            # For simplicity, we rely on the standard structure.

        # Ensure the new head is trainable
        for param in swin_model.head.parameters():
            param.requires_grad = True

        self.model = swin_model
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"PretrainedSwinTransformer: Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
