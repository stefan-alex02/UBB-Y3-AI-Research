import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision import models

from ..logger_utils import logger


class PretrainedViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 vit_model_variant: str = 'vit_b_16',
                 pretrained: bool = True,
                 unfreeze_strategy: str = 'encoder_tail',
                 num_transformer_blocks_to_unfreeze: int = 2,
                 unfreeze_cls_token: bool = True,
                 unfreeze_pos_embedding: bool = True,
                 unfreeze_patch_embedding: bool = False,
                 unfreeze_encoder_layernorm: bool = True,
                 custom_head_hidden_dims: Optional[List[int]] = None,
                 is_hybrid_input: bool = False,
                 hybrid_in_channels: Optional[int] = None,
                 hybrid_cnn_output_h: Optional[int] = None,
                 hybrid_cnn_output_w: Optional[int] = None,
                 head_dropout_rate: float = 0.0
                 ):
        """
        Initializes a pre-trained Vision Transformer (ViT) model with flexible fine-tuning options.

        This class provides an adaptable implementation of ViT that supports transfer learning
        with fine-grained control over which components remain frozen. It can be used in standard
        mode (direct image input) or hybrid mode (taking feature maps from a CNN backbone).

        Args:
            num_classes: Number of output classes for classification.
            vit_model_variant: Specific ViT architecture to use. Options include 'vit_b_16',
                              'vit_l_16', and 'vit_h_14'.
            pretrained: Whether to initialize with ImageNet pre-trained weights.
            unfreeze_strategy: Strategy for unfreezing transformer blocks:
                              'encoder_tail' (unfreeze last N blocks) or 'full_encoder' (unfreeze all).
            num_transformer_blocks_to_unfreeze: Number of transformer blocks to unfreeze from the end
                                               when using 'encoder_tail' strategy.
            unfreeze_cls_token: Whether to make the classification token trainable.
            unfreeze_pos_embedding: Whether to make positional embeddings trainable.
            unfreeze_patch_embedding: Whether to make patch embedding layer trainable.
            unfreeze_encoder_layernorm: Whether to make encoder layer normalization trainable.
            custom_head_hidden_dims: Optional list of hidden layer dimensions for a multi-layer
                                    classification head. If None, uses a single linear layer.
            is_hybrid_input: Whether to operate in hybrid mode (taking CNN feature maps as input
                            instead of raw images).
            hybrid_in_channels: Number of input channels in CNN feature maps for hybrid mode.
            hybrid_cnn_output_h: Height of CNN feature maps in hybrid mode.
            hybrid_cnn_output_w: Width of CNN feature maps in hybrid mode.
            head_dropout_rate: Dropout rate to apply in the classification head.

        Raises:
            ValueError: If num_classes is not positive, if hybrid mode parameters are missing,
                      or if the ViT variant is not supported.
            RuntimeError: If model loading fails.

        Note:
            In hybrid mode, the model expects input to be CNN feature maps rather than images.
            The positional embeddings are automatically interpolated to match the feature map size.
            Memory efficiency can be improved during training by using gradient checkpointing.
        """
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        self.is_hybrid_input = is_hybrid_input
        self.hybrid_in_channels = hybrid_in_channels

        logger.debug(f"Initializing PretrainedViT:")
        logger.debug(f"  Model Variant: {vit_model_variant}, Pretrained: {pretrained}")
        logger.debug(f"  Is Hybrid Input: {self.is_hybrid_input}")
        if self.is_hybrid_input:
            logger.debug(f"  Hybrid In Channels: {self.hybrid_in_channels}")
        logger.debug(f"  Unfreeze Strategy (Encoder Blocks): {unfreeze_strategy}")
        if unfreeze_strategy == 'encoder_tail':
            logger.debug(f"  Num Transformer Blocks to Unfreeze: {num_transformer_blocks_to_unfreeze}")
        logger.debug(
            f"  Unfreeze CLS: {unfreeze_cls_token}, PosEmb: {unfreeze_pos_embedding}, PatchEmb: {unfreeze_patch_embedding}, EncoderLN: {unfreeze_encoder_layernorm}")
        logger.debug(f"  Custom Head Hidden Dims: {custom_head_hidden_dims}, Head Dropout: {head_dropout_rate}")

        # Load model
        weights_arg = None
        if pretrained:
            if vit_model_variant == 'vit_b_16':
                weights_arg = models.ViT_B_16_Weights.IMAGENET1K_V1
            elif vit_model_variant == 'vit_l_16':
                weights_arg = models.ViT_L_16_Weights.IMAGENET1K_V1
            elif vit_model_variant == 'vit_h_14':
                weights_arg = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
            else:
                logger.warning(
                    f"No default weights enum for {vit_model_variant}. Using 'DEFAULT'."); weights_arg = "DEFAULT"
        try:
            vit_model_fn = getattr(models, vit_model_variant)
            self.model = vit_model_fn(weights=weights_arg)
        except AttributeError:
            raise ValueError(f"Unsupported ViT model variant: {vit_model_variant}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT '{vit_model_variant}': {e}")
        logger.debug(f"Loaded {vit_model_variant} from torchvision.")

        try:
            self.target_embed_dim = self.model.hidden_dim
        except AttributeError:
            logger.error(
                f"ViT model '{vit_model_variant}' does not have 'hidden_dim' attribute. Attempting to infer from conv_proj.")
            try:
                if hasattr(self.model, 'conv_proj') and self.model.conv_proj is not None:
                    self.target_embed_dim = self.model.conv_proj.out_channels
                elif hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed,
                                                                    'proj') and self.model.patch_embed.proj is not None:
                    self.target_embed_dim = self.model.patch_embed.proj.out_channels
                else:
                    raise AttributeError("Could not determine target_embed_dim from known attributes.")
            except AttributeError as e_infer:
                logger.critical(
                    f"CRITICAL: Could not determine target_embed_dim for ViT {vit_model_variant}: {e_infer}. Defaulting to 768, but this may be incorrect.")
                self.target_embed_dim = 768

        # Hybrid input adaptations
        if self.is_hybrid_input:
            if self.hybrid_in_channels is None:
                raise ValueError("hybrid_in_channels must be specified for hybrid ViT.")
            if hybrid_cnn_output_h is None or hybrid_cnn_output_w is None:
                raise ValueError("hybrid_cnn_output_h and hybrid_cnn_output_w must be provided for hybrid ViT.")

            logger.info(
                f"Adapting ViT '{vit_model_variant}' for hybrid input. Expected CNN feature map: {hybrid_cnn_output_h}x{hybrid_cnn_output_w}.")

            self.hybrid_input_projection = nn.Conv2d(
                self.hybrid_in_channels, self.target_embed_dim, kernel_size=1, stride=1, bias=True
            )
            nn.init.kaiming_normal_(self.hybrid_input_projection.weight, mode='fan_out', nonlinearity='relu')
            if self.hybrid_input_projection.bias is not None:
                nn.init.zeros_(self.hybrid_input_projection.bias)
            logger.info(
                f"Created hybrid input projection: {self.hybrid_in_channels} -> {self.target_embed_dim} channels.")

            if hasattr(self.model.encoder, 'pos_embedding'):
                self.original_pos_embedding = self.model.encoder.pos_embedding
            else:
                logger.warning(
                    "Could not find 'pos_embedding' on ViT encoder for hybrid mode. Positional encoding might be incorrect.")
                self.original_pos_embedding = None

            # Perform interpolation
            if self.original_pos_embedding is not None:
                H_feat_expected = hybrid_cnn_output_h
                W_feat_expected = hybrid_cnn_output_w

                num_patches_expected = H_feat_expected * W_feat_expected
                if self.original_pos_embedding.shape[1] != (num_patches_expected + 1):
                    logger.info(f"Interpolating positional embedding in __init__ for hybrid ViT. "
                                f"Original patches: {self.original_pos_embedding.shape[1] - 1}, "
                                f"Expected feature patches: {num_patches_expected} ({H_feat_expected}x{W_feat_expected})")
                    interpolated_pe = self._interpolate_pos_embedding_static(
                        self.original_pos_embedding,
                        H_feat_expected,
                        W_feat_expected,
                        self.target_embed_dim,
                        device='cpu'
                    )
                    self.interpolated_hybrid_pos_embedding = nn.Parameter(interpolated_pe.squeeze(0),
                                                                          requires_grad=unfreeze_pos_embedding)
                else:
                    self.interpolated_hybrid_pos_embedding = nn.Parameter(
                        self.original_pos_embedding.squeeze(0).clone(), requires_grad=unfreeze_pos_embedding
                    )
                if unfreeze_pos_embedding:
                    self.interpolated_hybrid_pos_embedding.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = False

        unfrozen_parts_log = []
        if hasattr(self, 'hybrid_input_projection'):
            for param in self.hybrid_input_projection.parameters():
                param.requires_grad = True
            unfrozen_parts_log.append("Hybrid Projection")
            logger.debug("Unfroze custom hybrid input projection layer for ViT.")

        # Unfreezing
        if not self.is_hybrid_input and unfreeze_patch_embedding and hasattr(self.model, 'conv_proj'):
            for param in self.model.conv_proj.parameters():
                param.requires_grad = True
            if "Original Patch Embedding" not in unfrozen_parts_log: unfrozen_parts_log.append(
                "Original Patch Embedding")

        # CLS Token
        cls_token_param = None
        if hasattr(self.model, 'class_token') and self.model.class_token is not None:
            cls_token_param = self.model.class_token
        elif hasattr(self.model, 'conv_proj') and hasattr(self.model.conv_proj,
                                                          'class_token') and self.model.conv_proj.class_token is not None:
            cls_token_param = self.model.conv_proj.class_token

        if unfreeze_cls_token and cls_token_param is not None:
            cls_token_param.requires_grad = True
            if "CLS Token" not in unfrozen_parts_log: unfrozen_parts_log.append("CLS Token")

        # Positional Embeddings
        pos_embed_to_consider = self.original_pos_embedding if self.is_hybrid_input and hasattr(self,
                                                                                                'original_pos_embedding') else \
            (self.model.encoder.pos_embedding if hasattr(self.model.encoder, 'pos_embedding') else None)
        if unfreeze_pos_embedding and pos_embed_to_consider is not None:
            pos_embed_to_consider.requires_grad = True
            if "Positional Embeddings" not in unfrozen_parts_log: unfrozen_parts_log.append("Positional Embeddings")

        # Transformer Encoder Blocks
        num_total_encoder_layers = len(self.model.encoder.layers)
        unfrozen_encoder_block_count = 0
        if unfreeze_strategy == 'encoder_tail':
            actual_num_to_unfreeze = min(num_transformer_blocks_to_unfreeze, num_total_encoder_layers)
            if num_transformer_blocks_to_unfreeze > num_total_encoder_layers:
                logger.warning(f"Requested to unfreeze {num_transformer_blocks_to_unfreeze} end encoder blocks, "
                               f"but model only has {num_total_encoder_layers}. Unfreezing all.")
            start_idx = num_total_encoder_layers - actual_num_to_unfreeze
            for i in range(start_idx, num_total_encoder_layers):
                for param in self.model.encoder.layers[i].parameters():
                    param.requires_grad = True
                unfrozen_encoder_block_count += 1
        elif unfreeze_strategy == 'full_encoder':
            for param in self.model.encoder.layers.parameters():
                param.requires_grad = True
            unfrozen_encoder_block_count = num_total_encoder_layers

        if unfrozen_encoder_block_count > 0:
            if f"{unfrozen_encoder_block_count} Encoder Blocks" not in unfrozen_parts_log:
                unfrozen_parts_log.append(f"{unfrozen_encoder_block_count} Encoder Blocks")

        # Encoder LayerNorm
        if unfreeze_encoder_layernorm and hasattr(self.model.encoder, 'ln'):
            for param in self.model.encoder.ln.parameters():
                param.requires_grad = True
            if "Encoder LayerNorm" not in unfrozen_parts_log: unfrozen_parts_log.append("Encoder LayerNorm")

        # Classification head
        original_head_in_features: int
        if hasattr(self.model.heads, 'head') and isinstance(self.model.heads.head, nn.Linear):
            original_head_in_features = self.model.heads.head.in_features
        elif isinstance(self.model.heads, nn.Linear):
            original_head_in_features = self.model.heads.in_features
        else:
            try:
                final_linear_layer = None
                if isinstance(self.model.heads, nn.Sequential):
                    for layer in reversed(list(self.model.heads.children())):
                        if isinstance(layer, nn.Linear): final_linear_layer = layer; break
                if final_linear_layer:
                    original_head_in_features = final_linear_layer.in_features
                elif hasattr(self.model, 'hidden_dim'):
                    original_head_in_features = self.model.hidden_dim
                else:
                    logger.warning(
                        "Could not reliably determine head in_features, defaulting."); original_head_in_features = 768
            except Exception as e_head:
                logger.error(
                    f"Error determining ViT head in_features: {e_head}. Defaulting."); original_head_in_features = 768

        if custom_head_hidden_dims and len(custom_head_hidden_dims) > 0:
            head_layers: List[nn.Module] = []
            current_in_features = original_head_in_features
            for i, hidden_dim in enumerate(custom_head_hidden_dims):
                head_layers.append(nn.Linear(current_in_features, hidden_dim))
                head_layers.append(nn.ReLU(inplace=True))
                if head_dropout_rate > 0: head_layers.append(nn.Dropout(head_dropout_rate))
                current_in_features = hidden_dim
            head_layers.append(nn.Linear(current_in_features, num_classes))
            final_head = nn.Sequential(*head_layers)
            logger.debug(
                f"Created custom MLP head: hidden_dims={custom_head_hidden_dims}, in_feat={original_head_in_features}")
        else:
            layers_for_simple_head: List[nn.Module] = []
            if head_dropout_rate > 0.0: layers_for_simple_head.append(nn.Dropout(head_dropout_rate))
            layers_for_simple_head.append(nn.Linear(original_head_in_features, num_classes))
            final_head = nn.Sequential(*layers_for_simple_head)
            logger.debug(
                f"Created simple linear head (dropout: {head_dropout_rate}), in_feat={original_head_in_features}.")

        self.model.heads = final_head

        for param in self.model.heads.parameters():
            param.requires_grad = True
        if "Classification Head" not in unfrozen_parts_log: unfrozen_parts_log.append("Classification Head")

        if unfrozen_parts_log:
            logger.info(
                f"PretrainedViT ({'hybrid' if self.is_hybrid_input else 'standard'}) - Trainable: {', '.join(unfrozen_parts_log)}.")
        else:
            logger.info(
                f"PretrainedViT ({'hybrid' if self.is_hybrid_input else 'standard'}) - Minimal parts trainable (e.g., only new head).")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        total_params_model = sum(p.numel() for p in self.model.parameters())
        total_params_hybrid_proj = 0
        if hasattr(self, 'hybrid_input_projection'):
            total_params_hybrid_proj = sum(p.numel() for p in self.hybrid_input_projection.parameters())

        total_params = total_params_model + total_params_hybrid_proj \
            if hasattr(self, 'hybrid_input_projection') and self.hybrid_input_projection is not self.model.conv_proj \
            else total_params_model

        # Calculate percentage
        percentage_trainable = (100 * trainable_params / total_params) if total_params > 0 else 0.0

        logger.info(
            f"PretrainedViT ({'hybrid' if self.is_hybrid_input else 'standard'}): "
            f"Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M "
            f"({percentage_trainable:.2f}%)"
        )

    @staticmethod
    def _interpolate_pos_embedding_static(original_pos_embed_param, H_feat_new, W_feat_new, target_embed_dim,
                                          device='cpu'):
        """
        Interpolates the positional embeddings of a Vision Transformer for a new feature map size.

        This method is used when adapting a pre-trained ViT model to a different input resolution
        or when using the hybrid approach where a CNN outputs feature maps of different dimensions
        than the original ViT patch grid. It keeps the class token embedding unchanged while
        resizing the patch embeddings using bicubic interpolation.

        Args:
            original_pos_embed_param: The original positional embedding parameter tensor from the ViT model.
                                     Should have shape [1, num_tokens, embed_dim] where num_tokens = 1 + grid_size^2.
            H_feat_new: Target height for the new feature map grid.
            W_feat_new: Target width for the new feature map grid.
            target_embed_dim: Embedding dimension of the position encodings.
            device: The device to perform computation on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Interpolated positional embeddings with shape [1, 1+H_feat_new*W_feat_new, target_embed_dim].

        Notes:
            - This method assumes the original position embeddings were created for a square grid of patches.
            - The first position embedding corresponding to the class token is preserved unchanged.
            - If the original embeddings don't form a square grid, it returns a zero tensor of the correct shape
              and logs an error.
            - Uses bicubic interpolation for resizing, which generally works better for positional embeddings
              than nearest neighbor or bilinear interpolation.
        """
        original_pos_embed = original_pos_embed_param.data.to(device)

        cls_pos_embed = original_pos_embed[:, :1, :]
        patch_pos_embed = original_pos_embed[:, 1:, :]
        orig_num_patches = patch_pos_embed.shape[1]
        orig_grid_size = int(math.sqrt(orig_num_patches))

        if orig_grid_size * orig_grid_size != orig_num_patches:
            logger.error(f"Static Interpolate: Original pos_embed (patches={orig_num_patches}) not square.")
            num_total_tokens_new = H_feat_new * W_feat_new + 1
            return torch.zeros(1, num_total_tokens_new, target_embed_dim, device=device)

        patch_pos_embed_2d = patch_pos_embed.transpose(1, 2).reshape(1, target_embed_dim, orig_grid_size,
                                                                     orig_grid_size)
        interpolated_patch_pos_embed_2d = nn.functional.interpolate(
            patch_pos_embed_2d, size=(H_feat_new, W_feat_new), mode='bicubic', align_corners=False
        )
        interpolated_patch_pos_embed = interpolated_patch_pos_embed_2d.flatten(2).transpose(1, 2)
        return torch.cat((cls_pos_embed, interpolated_patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass through the Vision Transformer model.

        This method handles both standard ViT processing and hybrid CNN-ViT processing paths.
        In standard mode, the input passes directly through the base ViT model.
        In hybrid mode, it processes CNN feature maps by:
          1. Projecting them to the embedding dimension
          2. Adding class tokens
          3. Adding positional embeddings
          4. Passing through the transformer encoder
          5. Extracting the class token output for classification

        Args:
            x: Input tensor. In standard mode, this should be image tensors of shape
               [batch_size, channels, height, width]. In hybrid mode, this should be CNN
               feature maps of shape [batch_size, hybrid_in_channels, height, width].

        Returns:
            torch.Tensor: Classification logits of shape [batch_size, num_classes].

        Raises:
            RuntimeError: If the class token cannot be found in hybrid mode.
        """
        if self.is_hybrid_input:
            projected_features = self.hybrid_input_projection(x)
            patches_from_cnn = projected_features.flatten(2).transpose(1, 2)

            cls_token_data = None
            if hasattr(self.model, 'class_token') and self.model.class_token is not None:
                cls_token_data = self.model.class_token
            elif hasattr(self.model, 'conv_proj') and hasattr(self.model.conv_proj,
                                                              'class_token') and self.model.conv_proj.class_token is not None:
                cls_token_data = self.model.conv_proj.class_token
            if cls_token_data is None: raise RuntimeError("ViT CLS token not found for hybrid mode.")
            cls_tokens = cls_token_data.expand(patches_from_cnn.shape[0], -1, -1)
            x_tokens = torch.cat((cls_tokens, patches_from_cnn), dim=1)

            # Use pre-interpolated embedding
            if hasattr(self, 'interpolated_hybrid_pos_embedding'):
                current_pos_embed = self.interpolated_hybrid_pos_embedding.unsqueeze(0).expand(x_tokens.shape[0], -1,
                                                                                               -1)
                x_embedded = x_tokens + current_pos_embed.to(x_tokens.device, dtype=x_tokens.dtype)
            else:
                logger.warning(
                    "Hybrid ViT: interpolated_hybrid_pos_embedding not found, skipping positional encoding addition.")
                x_embedded = x_tokens

            x_processed = self.model.encoder.dropout(x_embedded)
            x_processed = checkpoint(self.model.encoder.layers, x_processed, use_reentrant=False)
            x_processed = self.model.encoder.ln(x_processed)

            cls_token_output = x_processed[:, 0]
            output = self.model.heads(cls_token_output)
            return output
        else:
            return self.model(x)
