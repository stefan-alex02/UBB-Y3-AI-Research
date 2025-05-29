import torch
import torch.nn as nn
from typing import Optional, List

from .feature_extractors import PaperCNNFeatureExtractor  # Or other extractors
from .pretrained_vit import PretrainedViT
from .pretrained_swin import PretrainedSwin
from ..logger_utils import logger


class HybridViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 cnn_extractor_name: str = "paper_cnn",
                 cnn_out_channels: int = 48,
                 pretrained_cnn_path: Optional[str] = None,
                 # --- Parameters for the PretrainedViT backend ---
                 vit_model_variant: str = 'vit_b_16',
                 pretrained: bool = True,
                 unfreeze_strategy: str = 'encoder_tail',
                 num_transformer_blocks_to_unfreeze: int = 2,
                 unfreeze_cls_token: bool = True,
                 unfreeze_pos_embedding: bool = True,
                 unfreeze_patch_embedding: bool = False,
                 unfreeze_encoder_layernorm: bool = True,
                 freeze_cnn_extractor: bool = False,
                 custom_head_hidden_dims: Optional[List[int]] = None,
                 head_dropout_rate: float = 0.0,
                 # ---- Add parameter for pipeline_img_size ----
                 pipeline_img_h: int = 224, # Default or get from config
                 pipeline_img_w: int = 224  # Default or get from config
                 ):
        super().__init__()
        if cnn_extractor_name == "paper_cnn":
            self.feature_extractor = PaperCNNFeatureExtractor(in_channels=3)
            # Ensure cnn_out_channels matches the actual output of PaperCNNFeatureExtractor
            if self.feature_extractor.output_channels != cnn_out_channels:
                logger.warning(
                    f"Provided cnn_out_channels {cnn_out_channels} but PaperCNN outputs {self.feature_extractor.output_channels}. Using actual output channels.")
                cnn_out_channels_actual = self.feature_extractor.output_channels
            else:
                cnn_out_channels_actual = cnn_out_channels

        else:
            raise ValueError(f"Unknown CNN extractor name: {cnn_extractor_name}")

        if pretrained_cnn_path:
            logger.info(f"Loading pretrained weights for CNN extractor from: {pretrained_cnn_path}")
            self.feature_extractor.load_pretrained_weights(pretrained_cnn_path)
        else:
            logger.info("CNN feature extractor initialized with random weights.")

        # --- Freeze CNN extractor if specified ---
        if freeze_cnn_extractor:
            logger.info("Freezing CNN feature extractor parameters.")
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # --- Determine CNN output H, W ---
        # This assumes the HybridViT module will always receive images of size (pipeline_img_h, pipeline_img_w)
        # You might want to get this pipeline_img_h/w from a global config or pass it via Skorch params.
        # For now, let's assume it's passed or known.
        try:
            with torch.no_grad():
                # Create a dummy input matching what the HybridViT expects
                dummy_input_for_hybrid = torch.randn(1, 3, pipeline_img_h, pipeline_img_w)
                dummy_cnn_output = self.feature_extractor(dummy_input_for_hybrid)
            cnn_output_h_feat = dummy_cnn_output.shape[2]
            cnn_output_w_feat = dummy_cnn_output.shape[3]
            logger.info(f"Determined CNN feature extractor output size: {cnn_output_h_feat}x{cnn_output_w_feat} for input {pipeline_img_h}x{pipeline_img_w}")
        except Exception as e:
            logger.error(f"Could not determine CNN output H, W dynamically: {e}. "
                           f"Falling back to hardcoded 14x14 for 224px input assumption.")
            # Fallback based on your current CNN structure for 224px input
            if pipeline_img_h == 224 and pipeline_img_w == 224:
                cnn_output_h_feat = 14
                cnn_output_w_feat = 14
            else: # If pipeline size changes, this fallback is risky
                raise ValueError("Cannot determine CNN output size for non-224px input without dynamic check or better config.")

        self.transformer_backend = PretrainedViT(
            num_classes=num_classes,
            vit_model_variant=vit_model_variant,
            pretrained=pretrained,  # Pass through
            is_hybrid_input=True,
            hybrid_in_channels=cnn_out_channels_actual,  # From the CNN
            # ---- Pass the determined H_feat, W_feat ----
            hybrid_cnn_output_h=cnn_output_h_feat,
            hybrid_cnn_output_w=cnn_output_w_feat,
            # ----
            unfreeze_strategy=unfreeze_strategy,
            num_transformer_blocks_to_unfreeze=num_transformer_blocks_to_unfreeze,
            unfreeze_cls_token=unfreeze_cls_token,
            unfreeze_pos_embedding=unfreeze_pos_embedding,
            unfreeze_patch_embedding=unfreeze_patch_embedding,  # Will be ignored by ViT if is_hybrid_input=True
            unfreeze_encoder_layernorm=unfreeze_encoder_layernorm,
            custom_head_hidden_dims=custom_head_hidden_dims,
            head_dropout_rate=head_dropout_rate
        )

    def forward(self, x):
        features = self.feature_extractor(x)  # (B, C_cnn, H_feat, W_feat)
        output = self.transformer_backend(features)
        return output


class HybridSwin(nn.Module):
    def __init__(self,
                 num_classes: int,
                 cnn_extractor_name: str = "paper_cnn",
                 cnn_out_channels: int = 48,  # Expected by PretrainedSwin's hybrid_input_projection
                 swin_model_variant: str = 'swin_t',
                 # Pass other Swin params
                 pretrained_cnn_path: Optional[str] = None,
                 **swin_kwargs  # For num_stages_to_unfreeze etc.
                 ):
        super().__init__()
        if cnn_extractor_name == "paper_cnn":
            self.feature_extractor = PaperCNNFeatureExtractor(in_channels=3)
            if self.feature_extractor.output_channels != cnn_out_channels:
                logger.warning(
                    f"Provided cnn_out_channels {cnn_out_channels} but PaperCNN outputs {self.feature_extractor.output_channels}. Using actual output channels.")
                cnn_out_channels_actual = self.feature_extractor.output_channels
            else:
                cnn_out_channels_actual = cnn_out_channels
        else:
            raise ValueError(f"Unknown CNN extractor name: {cnn_extractor_name}")

        if pretrained_cnn_path:
            logger.info(f"Loading pretrained weights for CNN extractor from: {pretrained_cnn_path}")
            self.feature_extractor.load_pretrained_weights(pretrained_cnn_path)
        else:
            logger.info("CNN feature extractor initialized with random weights.")

        self.transformer_backend = PretrainedSwin(
            num_classes=num_classes,
            swin_model_variant=swin_model_variant,
            is_hybrid_input=True,
            hybrid_in_channels=cnn_out_channels_actual,  # Pass actual from CNN
            **swin_kwargs
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.transformer_backend(features)
        return output
