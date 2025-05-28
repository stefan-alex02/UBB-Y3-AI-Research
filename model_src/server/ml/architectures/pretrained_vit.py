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
                 # ---- New parameters for hybrid mode ----
                 hybrid_cnn_output_h: Optional[int] = None,
                 hybrid_cnn_output_w: Optional[int] = None,
                 # ----
                 head_dropout_rate: float = 0.0
                 ):
        super().__init__()
        if num_classes <= 0: raise ValueError("Number of classes must be positive.")
        self.num_classes = num_classes
        self.is_hybrid_input = is_hybrid_input
        self.hybrid_in_channels = hybrid_in_channels

        logger.debug(f"Initializing PretrainedViT (not FlexibleViT anymore):")  # Corrected class name in log
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

        # --- Load Model ---
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
            self.model = vit_model_fn(weights=weights_arg)  # Assign to self.model
        except AttributeError:
            raise ValueError(f"Unsupported ViT model variant: {vit_model_variant}.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT '{vit_model_variant}': {e}")
        logger.debug(f"Loaded {vit_model_variant} from torchvision.")

        # --- Define target_embed_dim AFTER self.model is loaded ---
        try:
            self.target_embed_dim = self.model.hidden_dim
        except AttributeError:
            logger.error(
                f"ViT model '{vit_model_variant}' does not have 'hidden_dim' attribute. Attempting to infer from conv_proj.")
            try:
                # For older torchvision ViTs, conv_proj might be directly on model
                if hasattr(self.model, 'conv_proj') and self.model.conv_proj is not None:
                    self.target_embed_dim = self.model.conv_proj.out_channels
                # For some ViT structures, it might be nested
                elif hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed,
                                                                    'proj') and self.model.patch_embed.proj is not None:
                    self.target_embed_dim = self.model.patch_embed.proj.out_channels
                else:
                    raise AttributeError("Could not determine target_embed_dim from known attributes.")
            except AttributeError as e_infer:
                logger.critical(
                    f"CRITICAL: Could not determine target_embed_dim for ViT {vit_model_variant}: {e_infer}. Defaulting to 768, but this may be incorrect.")
                self.target_embed_dim = 768  # Fallback, e.g., for vit_b_16, but could be wrong for others

        # --- Hybrid Input Adaptations ---
        if self.is_hybrid_input:
            if self.hybrid_in_channels is None:
                raise ValueError("hybrid_in_channels must be specified for hybrid ViT.")
            # ---- Check for new H, W params ----
            if hybrid_cnn_output_h is None or hybrid_cnn_output_w is None:
                raise ValueError("hybrid_cnn_output_h and hybrid_cnn_output_w must be provided for hybrid ViT.")
            # ----

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

            # ASSUME a fixed output size from CNN for a given input size to the hybrid model
            # Example: If hybrid model input is 224x224, and CNN always gives 56x56 features
            # This needs to be known or calculated based on your CNN's strides.
            # For PaperCNNFeatureExtractor with 448x448 input, it outputs 56x56.
            # If your main pipeline `img_size` is (224,224), PaperCNN will output 28x28.
            # Let's assume you know these:

            # You need to determine H_feat_expected, W_feat_expected based on your
            # pipeline's img_size and the CNN's downsampling factor.
            # For example, if pipeline.img_size = (224,224) and PaperCNN downsamples by 8 (224/8 = 28)
            # H_feat_expected = self.model.config.image_size // self.feature_extractor.downsample_factor
            # This requires PaperCNNFeatureExtractor to expose its downsample_factor or calculate output size.

            # Let's make a placeholder calculation for H_feat, W_feat based on a common scenario
            # If your overall pipeline img_size is (224, 224), and PaperCNNFeatureExtractor
            # consistently downsamples by a factor of 8 (common for ViT-like inputs from CNNs),
            # then H_feat and W_feat would be 224/8 = 28.
            # The paper's CNN outputs 56x56 for a 448x448 input.
            # If your input to HybridViT is 224x224, your PaperCNNFeatureExtractor outputs 28x28.

            # Get the expected feature map size from the CNN part
            # This is a bit of a hack, we should ideally pass this or calculate it based on CNN structure
            # For PaperCNNFeatureExtractor, it downsamples by 2 three times (conv2, mbconv3, mbconv4) -> total 8x downsample
            # So, if input to HybridViT is (say) 224x224, H_feat/W_feat will be 224/8 = 28
            # THIS NEEDS TO BE ACCURATE for your setup.
            # Let's assume your pipeline `img_size` is (224,224) for this example.
            # The PaperCNNFeatureExtractor was designed based on the paper which used 448x448.
            # If your pipeline img_size is (224,224), the actual H_feat, W_feat from PaperCNNFeatureExtractor
            # will be 224 / (stride_mbconv2 * stride_mbconv3 * stride_mbconv4) = 224 / (2*2*2) = 28x28.

            # It's better if PaperCNNFeatureExtractor can report its output spatial size for a given input size.
            # For now, let's hardcode an example assumption.
            # THIS IS A CRITICAL PART TO GET RIGHT.
            # Let's assume your self.pipeline.img_size = (224,224)
            # The PaperCNNFeatureExtractor has 3 stride-2 layers in its main path
            # So, H_feat_expected = 224 // 8 = 28, W_feat_expected = 224 // 8 = 28

            # To be more robust, you could do a dummy forward pass through the CNN extractor
            # with a dummy input of the expected pipeline image size to get H_feat, W_feat.
            # This is a common pattern.
            dummy_input_cnn = torch.randn(1, self.hybrid_in_channels if self.is_hybrid_input else 3, 224,
                                          224)  # Use a representative input size
            # Or better, get this from your pipeline config
            # Create a temporary instance of PaperCNNFeatureExtractor if it's not part of self.model
            # In HybridViT, self.feature_extractor is available.
            # In PretrainedViT standalone, you'd need to know the CNN's properties.
            # Let's assume this PretrainedViT is ONLY used inside HybridViT for this interpolation logic
            # or that these H_feat_expected, W_feat_expected are passed in if used standalone hybrid.

            # This calculation should ideally be done in HybridViT __init__ and the
            # resulting interpolated pos_embed passed to PretrainedViT or PretrainedViT does it once.

            # If PretrainedViT is doing it:
            if self.original_pos_embedding is not None:
                # Use the passed H_feat, W_feat for interpolation
                H_feat_expected = hybrid_cnn_output_h
                W_feat_expected = hybrid_cnn_output_w

                num_patches_expected = H_feat_expected * W_feat_expected
                if self.original_pos_embedding.shape[1] != (num_patches_expected + 1):
                    logger.info(f"Interpolating positional embedding in __init__ for hybrid ViT. "
                                f"Original patches: {self.original_pos_embedding.shape[1] - 1}, "
                                f"Expected feature patches: {num_patches_expected} ({H_feat_expected}x{W_feat_expected})")
                    # Call a static/helper version of _interpolate_pos_embedding
                    # that takes original_pos_embed, target_H, target_W, target_embed_dim
                    interpolated_pe = self._interpolate_pos_embedding_static(
                        self.original_pos_embedding,
                        H_feat_expected,
                        W_feat_expected,
                        self.target_embed_dim,
                        device='cpu'  # Interpolate on CPU then move to device
                    )
                    self.interpolated_hybrid_pos_embedding = nn.Parameter(interpolated_pe.squeeze(0),
                                                                          requires_grad=unfreeze_pos_embedding)
                    # Squeeze(0) if _interpolate_pos_embedding_static returns (1, N, D)
                    # nn.Parameter will handle device movement with the model.
                else:
                    # If sizes match, just use the original (potentially making it a parameter if unfreezing)
                    self.interpolated_hybrid_pos_embedding = nn.Parameter(
                        self.original_pos_embedding.squeeze(0).clone(), requires_grad=unfreeze_pos_embedding
                    )
                if unfreeze_pos_embedding:  # Ensure requires_grad is set if needed
                    self.interpolated_hybrid_pos_embedding.requires_grad = True

        # --- Freeze all parameters initially (of self.model) ---
        for param in self.model.parameters():
            param.requires_grad = False

        # If hybrid_input_projection was created, make it trainable (it's not part of self.model yet for freezing)
        unfrozen_parts_log = []
        if hasattr(self, 'hybrid_input_projection'):  # Check if the attribute exists
            for param in self.hybrid_input_projection.parameters():
                param.requires_grad = True
            unfrozen_parts_log.append("Hybrid Projection")
            logger.debug("Unfroze custom hybrid input projection layer for ViT.")

        # --- Unfreezing Logic ---
        # Original Patch Embedding (conv_proj) - only unfreeze if NOT hybrid and flag is True
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
            pos_embed_to_consider.requires_grad = True  # This makes the nn.Parameter trainable
            if "Positional Embeddings" not in unfrozen_parts_log: unfrozen_parts_log.append("Positional Embeddings")

        # Transformer Encoder Blocks
        num_total_encoder_layers = len(self.model.encoder.layers)  # Use self.model here
        unfrozen_encoder_block_count = 0
        if unfreeze_strategy == 'encoder_tail':
            actual_num_to_unfreeze = min(num_transformer_blocks_to_unfreeze, num_total_encoder_layers)
            if num_transformer_blocks_to_unfreeze > num_total_encoder_layers:
                logger.warning(f"Requested to unfreeze {num_transformer_blocks_to_unfreeze} end encoder blocks, "
                               f"but model only has {num_total_encoder_layers}. Unfreezing all.")
            start_idx = num_total_encoder_layers - actual_num_to_unfreeze
            for i in range(start_idx, num_total_encoder_layers):
                for param in self.model.encoder.layers[i].parameters():  # Use self.model
                    param.requires_grad = True
                unfrozen_encoder_block_count += 1
        elif unfreeze_strategy == 'full_encoder':
            for param in self.model.encoder.layers.parameters():  # Use self.model
                param.requires_grad = True
            unfrozen_encoder_block_count = num_total_encoder_layers

        if unfrozen_encoder_block_count > 0:
            if f"{unfrozen_encoder_block_count} Encoder Blocks" not in unfrozen_parts_log:
                unfrozen_parts_log.append(f"{unfrozen_encoder_block_count} Encoder Blocks")

        # Encoder's final LayerNorm
        if unfreeze_encoder_layernorm and hasattr(self.model.encoder, 'ln'):
            for param in self.model.encoder.ln.parameters():  # Use self.model
                param.requires_grad = True
            if "Encoder LayerNorm" not in unfrozen_parts_log: unfrozen_parts_log.append("Encoder LayerNorm")

        # --- Replace or customize the classification head ---
        original_head_in_features: int
        if hasattr(self.model.heads, 'head') and isinstance(self.model.heads.head, nn.Linear):  # Use self.model
            original_head_in_features = self.model.heads.head.in_features
        elif isinstance(self.model.heads, nn.Linear):  # Use self.model
            original_head_in_features = self.model.heads.in_features
        else:  # Fallback logic for head_in_features (as before)
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
            # ... (custom head creation logic as before) ...
            head_layers: List[nn.Module] = [];
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

        self.model.heads = final_head  # Replace the head on self.model

        for param in self.model.heads.parameters():  # Ensure new head is trainable
            param.requires_grad = True
        if "Classification Head" not in unfrozen_parts_log: unfrozen_parts_log.append("Classification Head")

        if unfrozen_parts_log:
            logger.info(
                f"PretrainedViT ({'hybrid' if self.is_hybrid_input else 'standard'}) - Trainable: {', '.join(unfrozen_parts_log)}.")
        else:  # Should at least have Head and Hybrid Projection if hybrid
            logger.info(
                f"PretrainedViT ({'hybrid' if self.is_hybrid_input else 'standard'}) - Minimal parts trainable (e.g., only new head).")

        # Log trainable params
        # self.parameters() will include hybrid_input_projection if it exists, and all TRAINABLE params from self.model
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # For total params, sum params from self.model and self.hybrid_input_projection (if it exists and is distinct)
        total_params_model = sum(p.numel() for p in self.model.parameters())
        total_params_hybrid_proj = 0
        if hasattr(self, 'hybrid_input_projection'):
            # Ensure we don't double count if hybrid_input_projection was somehow made part of self.model
            # (it shouldn't be with current structure, it's a separate member)
            total_params_hybrid_proj = sum(p.numel() for p in self.hybrid_input_projection.parameters())

        total_params = total_params_model + total_params_hybrid_proj \
            if hasattr(self, 'hybrid_input_projection') and self.hybrid_input_projection is not self.model.conv_proj \
            else total_params_model

        # Calculate percentage safely
        percentage_trainable = (100 * trainable_params / total_params) if total_params > 0 else 0.0

        logger.info(
            f"PretrainedViT ({'hybrid' if self.is_hybrid_input else 'standard'}): "
            f"Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M "
            f"({percentage_trainable:.2f}%)" # <<< CORRECTED
        )
        # Removed the redundant self.model = vit_model and subsequent logging of params for "FlexibleViT"

    # Add a static method for interpolation to be called from __init__
    @staticmethod
    def _interpolate_pos_embedding_static(original_pos_embed_param, H_feat_new, W_feat_new, target_embed_dim,
                                          device='cpu'):
        # original_pos_embed_param is the nn.Parameter from the loaded ViT
        original_pos_embed = original_pos_embed_param.data.to(device)  # Work with the tensor data

        cls_pos_embed = original_pos_embed[:, :1, :]
        patch_pos_embed = original_pos_embed[:, 1:, :]
        orig_num_patches = patch_pos_embed.shape[1]
        orig_grid_size = int(math.sqrt(orig_num_patches))

        if orig_grid_size * orig_grid_size != orig_num_patches:
            logger.error(f"Static Interpolate: Original pos_embed (patches={orig_num_patches}) not square.")
            # Fallback or error
            num_total_tokens_new = H_feat_new * W_feat_new + 1
            return torch.zeros(1, num_total_tokens_new, target_embed_dim, device=device)

        patch_pos_embed_2d = patch_pos_embed.transpose(1, 2).reshape(1, target_embed_dim, orig_grid_size,
                                                                     orig_grid_size)
        interpolated_patch_pos_embed_2d = nn.functional.interpolate(
            patch_pos_embed_2d, size=(H_feat_new, W_feat_new), mode='bicubic', align_corners=False
        )
        interpolated_patch_pos_embed = interpolated_patch_pos_embed_2d.flatten(2).transpose(1, 2)
        return torch.cat((cls_pos_embed, interpolated_patch_pos_embed), dim=1)

    # In PretrainedViT forward method for hybrid:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_hybrid_input:
            projected_features = self.hybrid_input_projection(x)
            patches_from_cnn = projected_features.flatten(2).transpose(1, 2)

            cls_token_data = None  # Find CLS token (same as before)
            # ...
            if hasattr(self.model, 'class_token') and self.model.class_token is not None:
                cls_token_data = self.model.class_token
            elif hasattr(self.model, 'conv_proj') and hasattr(self.model.conv_proj,
                                                              'class_token') and self.model.conv_proj.class_token is not None:
                cls_token_data = self.model.conv_proj.class_token
            if cls_token_data is None: raise RuntimeError("ViT CLS token not found for hybrid mode.")
            cls_tokens = cls_token_data.expand(patches_from_cnn.shape[0], -1, -1)
            x_tokens = torch.cat((cls_tokens, patches_from_cnn), dim=1)

            # Use the pre-interpolated embedding
            if hasattr(self, 'interpolated_hybrid_pos_embedding'):
                # Ensure it's expanded for batch and on the correct device
                current_pos_embed = self.interpolated_hybrid_pos_embedding.unsqueeze(0).expand(x_tokens.shape[0], -1,
                                                                                               -1)
                x_embedded = x_tokens + current_pos_embed.to(x_tokens.device, dtype=x_tokens.dtype)
            else:
                # This case should ideally not be hit if __init__ is correct
                logger.warning(
                    "Hybrid ViT: interpolated_hybrid_pos_embedding not found, skipping positional encoding addition.")
                x_embedded = x_tokens

            x_processed = self.model.encoder.dropout(x_embedded)
            x_processed = checkpoint(self.model.encoder.layers, x_processed, use_reentrant=False)  # With checkpointing
            x_processed = self.model.encoder.ln(x_processed)

            cls_token_output = x_processed[:, 0]
            output = self.model.heads(cls_token_output)
            return output
        else:
            return self.model(x)
