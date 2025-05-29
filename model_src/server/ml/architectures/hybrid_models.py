import torch
import torch.nn as nn
from typing import Optional, List

from .feature_extractors import PaperCNNFeatureExtractor  # Or other extractors
from .pretrained_vit import PretrainedViT
from .pretrained_swin import PretrainedSwin
from ..logger_utils import logger
from .standard_cnn_extractor import StandardCNNFeatureExtractor # New import

class HybridViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 # --- CNN Feature Extractor Configuration ---
                 cnn_extractor_type: str = "standard_cnn", # "standard_cnn" or "paper_cnn"
                 # Params for StandardCNNFeatureExtractor
                 cnn_model_name: str = "efficientnet_b0",
                 cnn_pretrained_imagenet: bool = True, # Use ImageNet weights for standard CNNs
                 cnn_output_channels_target: Optional[int] = 128, # Target channels for ViT input
                 cnn_freeze_extractor: bool = False, # Freeze the whole standard CNN
                 cnn_num_frozen_stages: int = 0, # Freeze early stages of standard CNN
                 # Path for CNN weights fine-tuned on cloud data (applies to standard or paper_cnn)
                 cnn_fine_tuned_weights_path: Optional[str] = None,

                 # --- Parameters for the PretrainedViT backend ---
                 vit_model_variant: str = 'vit_b_16',
                 vit_pretrained_imagenet: bool = True, # For the ViT backend itself
                 # ... (all other PretrainedViT params: unfreeze_strategy, head_dropout_rate, etc.)
                 pipeline_img_h: int = 224,
                 pipeline_img_w: int = 224,
                 **vit_kwargs # Catches other PretrainedViT specific params
                 ):
        super().__init__()

        if cnn_extractor_type == "standard_cnn":
            self.feature_extractor = StandardCNNFeatureExtractor(
                model_name=cnn_model_name,
                pretrained=cnn_pretrained_imagenet,
                output_channels_target=cnn_output_channels_target,
                freeze_extractor=cnn_freeze_extractor,
                num_frozen_stages=cnn_num_frozen_stages
            )
            if cnn_fine_tuned_weights_path:
                logger.info(f"Loading fine-tuned cloud weights for StandardCNNFeatureExtractor from: {cnn_fine_tuned_weights_path}")
                self.feature_extractor.load_fine_tuned_weights(cnn_fine_tuned_weights_path)
            # Else it uses ImageNet weights if cnn_pretrained_imagenet=True, or random if False

        elif cnn_extractor_type == "paper_cnn":
            self.feature_extractor = PaperCNNFeatureExtractor(in_channels=3)
            if cnn_fine_tuned_weights_path: # Renamed from pretrained_cnn_path for clarity
                logger.info(f"Loading fine-tuned cloud weights for PaperCNNFeatureExtractor from: {cnn_fine_tuned_weights_path}")
                self.feature_extractor.load_pretrained_weights(cnn_fine_tuned_weights_path) # Uses its own loading method
            else:
                logger.info("PaperCNNFeatureExtractor initialized with random weights (no cloud pretraining path provided).")
        else:
            raise ValueError(f"Unknown cnn_extractor_type: {cnn_extractor_type}")

        cnn_out_channels_actual = self.feature_extractor.current_output_channels

        # Determine CNN output H, W (as before)
        try:
            with torch.no_grad():
                dummy_input_for_hybrid = torch.randn(1, 3, pipeline_img_h, pipeline_img_w)
                dummy_cnn_output = self.feature_extractor(dummy_input_for_hybrid)
            cnn_output_h_feat = dummy_cnn_output.shape[2]
            cnn_output_w_feat = dummy_cnn_output.shape[3]
            logger.info(f"Determined CNN feature extractor output size: {cnn_output_h_feat}x{cnn_output_w_feat} for input {pipeline_img_h}x{pipeline_img_w}")
        except Exception as e:
            # ... (fallback logic for cnn_output_h_feat, cnn_output_w_feat) ...
            # For EfficientNet-B0 or ResNet18 with 224px input, this should be 7x7
            if pipeline_img_h == 224 and pipeline_img_w == 224 and \
               (cnn_model_name.startswith("efficientnet") or cnn_model_name.startswith("resnet") or cnn_model_name.startswith("mobilenet")):
                cnn_output_h_feat = 7
                cnn_output_w_feat = 7
            else:
                 logger.error(f"Could not determine CNN output H, W dynamically: {e}. You might need to hardcode or improve logic.")
                 # A default based on common downsampling, adjust if necessary
                 cnn_output_h_feat = pipeline_img_h // self.feature_extractor.downsample_factor
                 cnn_output_w_feat = pipeline_img_w // self.feature_extractor.downsample_factor


        self.transformer_backend = PretrainedViT(
            num_classes=num_classes,
            vit_model_variant=vit_model_variant,
            pretrained=vit_pretrained_imagenet, # Pretrained status of ViT itself
            is_hybrid_input=True,
            hybrid_in_channels=cnn_out_channels_actual,
            hybrid_cnn_output_h=cnn_output_h_feat,
            hybrid_cnn_output_w=cnn_output_w_feat,
            **vit_kwargs # Passes unfreeze_strategy, num_blocks, head_dropout, etc.
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.transformer_backend(features)
        return output

