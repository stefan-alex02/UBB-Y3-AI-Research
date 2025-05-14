from typing import Optional, List

import torch
import torch.nn as nn
from torchvision import models

from ..logger_utils import logger


class PretrainedViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 vit_model_variant: str = 'vit_b_16',
                 pretrained: bool = True,
                 unfreeze_strategy: str = 'encoder_tail', # More descriptive common name
                 num_transformer_blocks_to_unfreeze: int = 2, # Renamed for clarity
                 unfreeze_cls_token: bool = True,
                 unfreeze_pos_embedding: bool = True,
                 unfreeze_patch_embedding: bool = False, # Usually kept frozen
                 unfreeze_encoder_layernorm: bool = True, # The final LN of the encoder
                 custom_head_hidden_dims: Optional[List[int]] = None,
                 head_dropout_rate: float = 0.0
                 ):
        """
        Flexible Vision Transformer (ViT) for image classification.

        Allows selection of ViT variants, customization of the classification head,
        and fine-grained control over unfreezing pre-trained layers.

        Args:
            num_classes (int): Number of output classes.
            vit_model_variant (str): ViT variant from `torchvision.models`
                (e.g., `vit_b_16` or `vit_l_16` or `vit_h_14`).
            pretrained (bool): If True, loads ImageNet pre-trained weights.
            unfreeze_strategy (str): Strategy for unfreezing encoder blocks.
                Options:
                - 'none': Freeze all encoder blocks. Only elements specified by other
                          `unfreeze_*` flags (like CLS, PosEmb, Head) might be trainable.
                - 'encoder_tail': Unfreeze the last `num_transformer_blocks_to_unfreeze`
                                  Transformer encoder blocks.
                - 'full_encoder': Unfreeze all Transformer encoder blocks.
                (The head, and optionally CLS token, positional embeddings, and final encoder
                 LayerNorm are handled by separate flags below for more explicit control).
            num_transformer_blocks_to_unfreeze (int): Number of final Transformer encoder
                blocks to unfreeze. Used if `unfreeze_strategy` is 'encoder_tail'.
            unfreeze_cls_token (bool): If True and model has a CLS token, make it trainable.
            unfreeze_pos_embedding (bool): If True, make positional embeddings trainable.
            unfreeze_patch_embedding (bool): If True, make the initial patch embedding layer
                (conv_proj) trainable. Usually kept False for fine-tuning.
            unfreeze_encoder_layernorm (bool): If True, unfreeze the final LayerNorm of the
                Transformer encoder (if it exists).
            custom_head_hidden_dims (Optional[List[int]]): Hidden layer dimensions for a custom MLP head.
                If None, a simple linear head is used.
            head_dropout_rate (float): Dropout rate in the classification head.
        """
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        logger.debug(f"Initializing FlexibleViT:")
        logger.debug(f"  Model Variant: {vit_model_variant}, Pretrained: {pretrained}")
        logger.debug(f"  Unfreeze Strategy (Encoder Blocks): {unfreeze_strategy}")
        if unfreeze_strategy == 'encoder_tail':
            logger.debug(f"  Num Transformer Blocks to Unfreeze: {num_transformer_blocks_to_unfreeze}")
        logger.debug(f"  Unfreeze CLS: {unfreeze_cls_token}, PosEmb: {unfreeze_pos_embedding}, PatchEmb: {unfreeze_patch_embedding}, EncoderLN: {unfreeze_encoder_layernorm}")
        logger.debug(f"  Custom Head Hidden Dims: {custom_head_hidden_dims}, Head Dropout: {head_dropout_rate}")

        # --- Load Model ---
        weights_arg = None
        if pretrained:
            if vit_model_variant == 'vit_b_16': weights_arg = models.ViT_B_16_Weights.IMAGENET1K_V1
            elif vit_model_variant == 'vit_l_16': weights_arg = models.ViT_L_16_Weights.IMAGENET1K_V1
            elif vit_model_variant == 'vit_h_14': weights_arg = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
            else: logger.warning(f"No default weights enum for {vit_model_variant}. Using 'DEFAULT'."); weights_arg = "DEFAULT"
        try:
            vit_model_fn = getattr(models, vit_model_variant)
            vit_model = vit_model_fn(weights=weights_arg)
        except AttributeError: raise ValueError(f"Unsupported ViT model variant: {vit_model_variant}.")
        except Exception as e: raise RuntimeError(f"Failed to load ViT '{vit_model_variant}': {e}")
        logger.debug(f"Loaded {vit_model_variant} from torchvision.")

        # --- Freeze all parameters initially ---
        for param in vit_model.parameters():
            param.requires_grad = False

        # --- Unfreezing Logic (more granular) ---
        unfrozen_parts_log = []

        # 1. Patch Embedding (conv_proj)
        if unfreeze_patch_embedding and hasattr(vit_model, 'conv_proj'):
            for param in vit_model.conv_proj.parameters():
                param.requires_grad = True
            # CLS token is part of conv_proj in newer ViTs, handle it with its own flag
            if hasattr(vit_model.conv_proj, 'class_token') and vit_model.conv_proj.class_token is not None:
                 vit_model.conv_proj.class_token.requires_grad = unfreeze_cls_token
            unfrozen_parts_log.append("Patch Embedding")

        # 2. CLS Token (if not part of conv_proj or if conv_proj not unfrozen but cls_token is)
        if unfreeze_cls_token:
            if hasattr(vit_model, 'class_token') and vit_model.class_token is not None: # Older style
                vit_model.class_token.requires_grad = True
                if "CLS Token" not in unfrozen_parts_log: unfrozen_parts_log.append("CLS Token")
            elif hasattr(vit_model, 'conv_proj') and hasattr(vit_model.conv_proj, 'class_token') and \
                 vit_model.conv_proj.class_token is not None and not (unfreeze_patch_embedding and vit_model.conv_proj.class_token.requires_grad):
                vit_model.conv_proj.class_token.requires_grad = True # Ensure it's true if flag is true
                if "CLS Token" not in unfrozen_parts_log: unfrozen_parts_log.append("CLS Token (in conv_proj)")


        # 3. Positional Embeddings
        if unfreeze_pos_embedding and hasattr(vit_model.encoder, 'pos_embedding'):
            vit_model.encoder.pos_embedding.requires_grad = True
            unfrozen_parts_log.append("Positional Embeddings")

        # 4. Transformer Encoder Blocks
        num_total_encoder_layers = len(vit_model.encoder.layers)
        unfrozen_encoder_block_count = 0
        if unfreeze_strategy == 'encoder_tail':
            actual_num_to_unfreeze = min(num_transformer_blocks_to_unfreeze, num_total_encoder_layers)
            if num_transformer_blocks_to_unfreeze > num_total_encoder_layers:
                logger.warning(f"Requested to unfreeze {num_transformer_blocks_to_unfreeze} end encoder blocks, "
                               f"but model only has {num_total_encoder_layers}. Unfreezing all {num_total_encoder_layers}.")
            start_idx = num_total_encoder_layers - actual_num_to_unfreeze
            for i in range(start_idx, num_total_encoder_layers):
                for param in vit_model.encoder.layers[i].parameters():
                    param.requires_grad = True
                unfrozen_encoder_block_count += 1
        elif unfreeze_strategy == 'full_encoder':
            for param in vit_model.encoder.layers.parameters(): # Unfreeze all blocks
                param.requires_grad = True
            unfrozen_encoder_block_count = num_total_encoder_layers
        elif unfreeze_strategy == 'none':
            pass # No encoder blocks unfrozen by this strategy
        else:
            logger.warning(f"Unknown encoder block unfreeze_strategy: {unfreeze_strategy}. No encoder blocks will be unfrozen by strategy.")

        if unfrozen_encoder_block_count > 0:
            unfrozen_parts_log.append(f"{unfrozen_encoder_block_count} Encoder Blocks")

        # 5. Encoder's final LayerNorm
        if unfreeze_encoder_layernorm and hasattr(vit_model.encoder, 'ln'):
            for param in vit_model.encoder.ln.parameters():
                param.requires_grad = True
            unfrozen_parts_log.append("Encoder LayerNorm")

        # --- Replace or customize the classification head ---
        # (Head in_features detection logic - unchanged from previous version)
        original_head_in_features: int
        if hasattr(vit_model.heads, 'head') and isinstance(vit_model.heads.head, nn.Linear): original_head_in_features = vit_model.heads.head.in_features
        elif isinstance(vit_model.heads, nn.Linear): original_head_in_features = vit_model.heads.in_features
        else:
            try:
                final_linear_layer = None
                if isinstance(vit_model.heads, nn.Sequential):
                    for layer in reversed(list(vit_model.heads.children())):
                        if isinstance(layer, nn.Linear): final_linear_layer = layer; break
                if final_linear_layer: original_head_in_features = final_linear_layer.in_features
                elif hasattr(vit_model, 'hidden_dim'): original_head_in_features = vit_model.hidden_dim
                else: logger.warning("Could not reliably determine head in_features, defaulting based on vit_b_16."); original_head_in_features = 768
            except Exception as e_head: logger.error(f"Error determining ViT head in_features: {e_head}. Defaulting to 768."); original_head_in_features = 768

        # (Custom head creation logic - unchanged from previous version)
        if custom_head_hidden_dims and len(custom_head_hidden_dims) > 0:
            head_layers: List[nn.Module] = []; current_in_features = original_head_in_features
            for i, hidden_dim in enumerate(custom_head_hidden_dims):
                head_layers.append(nn.Linear(current_in_features, hidden_dim))
                head_layers.append(nn.ReLU(inplace=True))
                if head_dropout_rate > 0: head_layers.append(nn.Dropout(head_dropout_rate))
                current_in_features = hidden_dim
                # TODO: add BatchNorm
            head_layers.append(nn.Linear(current_in_features, num_classes))
            final_head = nn.Sequential(*head_layers)
            logger.debug(f"Created custom MLP head: hidden_dims={custom_head_hidden_dims}, in_feat={original_head_in_features}")
        else:
            layers_for_simple_head: List[nn.Module] = []
            if head_dropout_rate > 0.0: layers_for_simple_head.append(nn.Dropout(head_dropout_rate))
            layers_for_simple_head.append(nn.Linear(original_head_in_features, num_classes))
            final_head = nn.Sequential(*layers_for_simple_head)
            logger.debug(f"Created simple linear head (dropout: {head_dropout_rate}), in_feat={original_head_in_features}.")

        vit_model.heads = final_head # Replace the entire 'heads' attribute

        # 6. Ensure all parameters of the new head are always trainable
        for param in vit_model.heads.parameters():
            param.requires_grad = True
        unfrozen_parts_log.append("Classification Head") # Always unfrozen

        if unfrozen_parts_log: logger.info(f"FlexibleViT - Trainable components: {', '.join(unfrozen_parts_log)}.")
        else: logger.info("FlexibleViT - No components explicitly unfrozen (only new head is trainable).")


        self.model = vit_model
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"FlexibleViT: Trainable params: {trainable_params/1e6:.2f}M / Total params: {total_params/1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
