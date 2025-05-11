from typing import Optional, List

import torch
import torch.nn as nn
from torchvision import models

from ..logger_utils import logger


class FlexibleViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 vit_model_variant: str = 'vit_b_16',  # e.g., 'vit_b_16', 'vit_l_16', 'vit_h_14'
                 pretrained: bool = True,
                 # Unfreezing strategy:
                 # 'end_encoder_layers': Unfreeze the last N encoder layers + pos_emb + cls_token + head.
                 # 'all_encoder_layers': Unfreeze all encoder layers + pos_emb + cls_token + head.
                 # 'head_only': Unfreeze only the classification head + pos_emb + cls_token.
                 # 'specific_layers': For advanced use, pass a list of layer names/indices to unfreeze.
                 # 'none': Freeze everything except the randomly initialized new head (not generally recommended for fine-tuning).
                 unfreeze_strategy: str = 'end_encoder_layers',
                 num_end_encoder_layers_to_unfreeze: int = 2,  # Used if strategy is 'end_encoder_layers'
                 # For SimpleViT compatibility: unfreeze_strategy='simple_vit_compat'
                 # will try to mimic the old SimpleViT behavior.
                 # Custom head configuration:
                 custom_head_hidden_dims: Optional[List[int]] = None,  # e.g., [512] for one hidden layer in head
                 head_dropout_rate: float = 0.0,  # Default for torchvision's linear head is 0.0
                 ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes

        logger.debug(f"Initializing FlexibleViT:")
        logger.debug(f"  Model Variant: {vit_model_variant}, Pretrained: {pretrained}")
        logger.debug(f"  Unfreeze Strategy: {unfreeze_strategy}")
        if unfreeze_strategy == 'end_encoder_layers':
            logger.debug(f"  Num End Encoder Layers to Unfreeze: {num_end_encoder_layers_to_unfreeze}")
        logger.debug(f"  Custom Head Hidden Dims: {custom_head_hidden_dims}, Head Dropout: {head_dropout_rate}")

        # --- Load Model ---
        weights_arg = None
        if pretrained:
            if vit_model_variant == 'vit_b_16':
                weights_arg = models.ViT_B_16_Weights.IMAGENET1K_V1
            elif vit_model_variant == 'vit_l_16':
                weights_arg = models.ViT_L_16_Weights.IMAGENET1K_V1
            elif vit_model_variant == 'vit_h_14':
                weights_arg = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1  # Example, pick appropriate
            # Add more variants and their default pre-trained weights here
            else:
                logger.warning(
                    f"No default pretrained weights specified for {vit_model_variant}. Loading without specific weights enum.")

        try:
            vit_model_fn = getattr(models, vit_model_variant)
            vit_model = vit_model_fn(weights=weights_arg)
        except AttributeError:
            raise ValueError(f"Unsupported ViT model variant: {vit_model_variant}. Check torchvision.models.")
        logger.debug(f"Loaded {vit_model_variant} from torchvision.")

        # --- Freeze all parameters initially ---
        for param in vit_model.parameters():
            param.requires_grad = False

        # --- Unfreezing Logic ---
        # Always unfreeze classification head (as it's replaced/modified)
        # Positional embeddings and CLS token are often good to fine-tune.

        # Unfreeze CLS token and Positional Embeddings (handle different torchvision versions)
        if hasattr(vit_model, 'class_token') and vit_model.class_token is not None:  # torchvision < 0.12 style
            vit_model.class_token.requires_grad = True
            logger.debug("Unfroze model.class_token")
        elif hasattr(vit_model, 'conv_proj') and hasattr(vit_model.conv_proj,
                                                         'class_token') and vit_model.conv_proj.class_token is not None:  # For vit_b_16 in newer torchvision
            vit_model.conv_proj.class_token.requires_grad = True
            logger.debug("Unfroze model.conv_proj.class_token")
        # Some ViT variants might not have an explicit class token (e.g., if using global average pooling on patch tokens)
        # This example assumes a CLS token exists as per standard ViT.

        if hasattr(vit_model.encoder, 'pos_embedding'):
            vit_model.encoder.pos_embedding.requires_grad = True
            logger.debug("Unfroze model.encoder.pos_embedding")

        unfrozen_encoder_block_count = 0
        if unfreeze_strategy == 'simple_vit_compat':
            # Mimic SimpleViT: Unfreeze the head (done below) and the last few parameter groups.
            # This is an approximation because parameter groups don't map 1:1 to "layers" easily.
            # SimpleViT used `total_params - num_layers_to_unfreeze`.
            # A more robust way for compatibility is to unfreeze the encoder's norm and last N blocks.
            logger.info(
                "Using 'simple_vit_compat' unfreeze strategy: unfreezing head, pos_emb, cls_token, encoder norm, and last 2 encoder blocks.")
            if hasattr(vit_model.encoder, 'ln'):  # Encoder's final LayerNorm
                for param in vit_model.encoder.ln.parameters():
                    param.requires_grad = True

            num_total_encoder_layers = len(vit_model.encoder.layers)
            num_blocks_to_unfreeze_compat = min(2, num_total_encoder_layers)  # Unfreeze last 2 blocks for compatibility
            for i in range(num_total_encoder_layers - num_blocks_to_unfreeze_compat, num_total_encoder_layers):
                for param in vit_model.encoder.layers[i].parameters():
                    param.requires_grad = True
                unfrozen_encoder_block_count += 1

        elif unfreeze_strategy == 'end_encoder_layers':
            if hasattr(vit_model.encoder, 'ln'):  # Encoder's final LayerNorm, often good to unfreeze with last blocks
                for param in vit_model.encoder.ln.parameters():
                    param.requires_grad = True

            num_total_encoder_layers = len(vit_model.encoder.layers)
            actual_num_to_unfreeze = min(num_end_encoder_layers_to_unfreeze, num_total_encoder_layers)
            if num_end_encoder_layers_to_unfreeze > num_total_encoder_layers:
                logger.warning(f"Requested to unfreeze {num_end_encoder_layers_to_unfreeze} end encoder layers, "
                               f"but model only has {num_total_encoder_layers}. Unfreezing all {num_total_encoder_layers} encoder layers.")

            for i in range(num_total_encoder_layers - actual_num_to_unfreeze, num_total_encoder_layers):
                for param in vit_model.encoder.layers[i].parameters():
                    param.requires_grad = True
                unfrozen_encoder_block_count += 1

        elif unfreeze_strategy == 'all_encoder_layers':
            for param in vit_model.encoder.parameters():  # Unfreeze all encoder parameters
                param.requires_grad = True
            unfrozen_encoder_block_count = len(vit_model.encoder.layers)

        elif unfreeze_strategy == 'head_only':
            # CLS, Positional Embeddings already unfrozen. Head will be unfrozen below.
            pass  # No additional encoder layers unfrozen

        elif unfreeze_strategy == 'none':
            # Only the new head will be trainable. CLS and Positional embeddings remain frozen.
            # This means we need to re-freeze CLS and Positional if they were unfrozen above.
            if hasattr(vit_model, 'class_token') and vit_model.class_token is not None:
                vit_model.class_token.requires_grad = False
            elif hasattr(vit_model, 'conv_proj') and hasattr(vit_model.conv_proj, 'class_token'):
                vit_model.conv_proj.class_token.requires_grad = False
            if hasattr(vit_model.encoder, 'pos_embedding'):
                vit_model.encoder.pos_embedding.requires_grad = False
            logger.info("Unfreeze strategy 'none': Only the new classification head will be trainable.")
        else:
            logger.warning(f"Unknown unfreeze_strategy: {unfreeze_strategy}. Defaulting to 'head_only'.")
            # Fallback to head_only implicitly

        if unfrozen_encoder_block_count > 0:
            logger.info(f"Unfroze last {unfrozen_encoder_block_count} Transformer encoder blocks.")
        elif unfreeze_strategy not in ['head_only', 'none', 'simple_vit_compat']:
            logger.info("No Transformer encoder blocks were unfrozen based on strategy.")

        # --- Replace or customize the classification head ---
        original_head_in_features: int
        # torchvision.models.vit_b_16().heads is an nn.Sequential(OrderedDict([('head', nn.Linear(...))]))
        # We need to access the .in_features of that nn.Linear layer.
        if hasattr(vit_model.heads, 'head') and isinstance(vit_model.heads.head, nn.Linear):
            original_head_in_features = vit_model.heads.head.in_features
        elif isinstance(vit_model.heads, nn.Linear):  # Some models might have a simpler head
            original_head_in_features = vit_model.heads.in_features
        else:
            # Fallback or error if head structure is unexpected
            # For ViT, vit_model.hidden_dim might be an alternative if heads structure is too complex/changed
            # or use a probe with a dummy input if really necessary (but try to avoid)
            try:
                # Attempt to find the last linear layer's input features if standard access fails
                # This is a bit heuristic
                last_layer = None
                for layer in reversed(list(vit_model.heads.children())):
                    if isinstance(layer, nn.Linear):
                        last_layer = layer
                        break
                if last_layer:
                    original_head_in_features = last_layer.in_features
                else:
                    raise AttributeError("Could not determine in_features for ViT head.")
            except Exception as e:
                logger.error(
                    f"Could not automatically determine head in_features: {e}. Defaulting to 768 for vit_b_16 like models.")
                original_head_in_features = 768  # Common for vit_b_16

        if custom_head_hidden_dims and len(custom_head_hidden_dims) > 0:
            head_layers: List[nn.Module] = []
            current_in_features = original_head_in_features
            for i, hidden_dim in enumerate(custom_head_hidden_dims):
                head_layers.append(nn.Linear(current_in_features, hidden_dim))
                head_layers.append(nn.ReLU(inplace=True))
                if head_dropout_rate > 0:
                    head_layers.append(nn.Dropout(head_dropout_rate))
                current_in_features = hidden_dim
            head_layers.append(nn.Linear(current_in_features, num_classes))
            final_head = nn.Sequential(*head_layers)
            logger.debug(f"Created custom MLP head with hidden dims: {custom_head_hidden_dims}")
        else:
            # Default: single linear layer, possibly with dropout before it
            layers_for_simple_head: List[nn.Module] = []
            if head_dropout_rate > 0.0:  # Add dropout only if specified, before the linear layer
                layers_for_simple_head.append(nn.Dropout(head_dropout_rate))
            layers_for_simple_head.append(nn.Linear(original_head_in_features, num_classes))
            final_head = nn.Sequential(*layers_for_simple_head)
            logger.debug(f"Created simple linear head (dropout: {head_dropout_rate}).")

        vit_model.heads = final_head  # Replace the entire 'heads' sequential block

        # Ensure all parameters of the new head are trainable
        for param in vit_model.heads.parameters():
            param.requires_grad = True
        logger.debug(
            f"Replaced ViT heads with a new head for {num_classes} classes. All new head params are trainable.")

        self.model = vit_model

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"FlexibleViT: Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M "
            f"({100 * trainable_params / total_params:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
