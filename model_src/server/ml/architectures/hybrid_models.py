from typing import Optional

import torch
import torch.nn as nn

from .feature_extractors import PaperCNNFeatureExtractor
from .pretrained_vit import PretrainedViT
from .standard_cnn_extractor import StandardCNNFeatureExtractor
from ..logger_utils import logger


class HybridViT(nn.Module):
    def __init__(self,
                 num_classes: int,
                 cnn_extractor_type: str = "standard_cnn",
                 cnn_model_name: str = "efficientnet_b0",
                 cnn_pretrained_imagenet: bool = True,
                 cnn_output_channels_target: Optional[int] = 128,
                 cnn_freeze_extractor: bool = False,
                 cnn_num_frozen_stages: int = 0,
                 cnn_fine_tuned_weights_path: Optional[str] = None,

                 vit_model_variant: str = 'vit_b_16',
                 vit_pretrained_imagenet: bool = True,
                 pipeline_img_h: int = 224,
                 pipeline_img_w: int = 224,

                 **vit_kwargs
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

        elif cnn_extractor_type == "paper_cnn":
            self.feature_extractor = PaperCNNFeatureExtractor(in_channels=3)
            if cnn_fine_tuned_weights_path:
                logger.info(f"Loading fine-tuned cloud weights for PaperCNNFeatureExtractor from: {cnn_fine_tuned_weights_path}")
                self.feature_extractor.load_pretrained_weights(cnn_fine_tuned_weights_path)
            else:
                logger.info("PaperCNNFeatureExtractor initialized with random weights (no cloud pretraining path provided).")
        else:
            raise ValueError(f"Unknown cnn_extractor_type: {cnn_extractor_type}")

        cnn_out_channels_actual = self.feature_extractor.current_output_channels

        cnn_params = self.feature_extractor.get_parameter_counts()
        cnn_total_params = cnn_params['total_params']
        cnn_trainable_params = cnn_params['trainable_params']
        logger.info(f"CNN Feature Extractor ({self.feature_extractor.model_name}) part: "
                    f"Trainable params: {cnn_trainable_params / 1e6:.2f}M / Total params: {cnn_total_params / 1e6:.2f}M")

        # Determine CNN output H, W
        try:
            with torch.no_grad():
                dummy_input_for_hybrid = torch.randn(1, 3, pipeline_img_h, pipeline_img_w)
                dummy_cnn_output = self.feature_extractor(dummy_input_for_hybrid)
            cnn_output_h_feat = dummy_cnn_output.shape[2]
            cnn_output_w_feat = dummy_cnn_output.shape[3]
            logger.info(f"Determined CNN feature extractor output size: {cnn_output_h_feat}x{cnn_output_w_feat} for input {pipeline_img_h}x{pipeline_img_w}")
        except Exception as e:
            if pipeline_img_h == 224 and pipeline_img_w == 224 and \
               (cnn_model_name.startswith("efficientnet") or cnn_model_name.startswith("resnet") or cnn_model_name.startswith("mobilenet")):
                cnn_output_h_feat = 7
                cnn_output_w_feat = 7
            else:
                 logger.error(f"Could not determine CNN output H, W dynamically: {e}. You might need to hardcode or improve logic.")
                 cnn_output_h_feat = pipeline_img_h // self.feature_extractor.downsample_factor
                 cnn_output_w_feat = pipeline_img_w // self.feature_extractor.downsample_factor


        self.transformer_backend = PretrainedViT(
            num_classes=num_classes,
            vit_model_variant=vit_model_variant,
            pretrained=vit_pretrained_imagenet,
            is_hybrid_input=True,
            hybrid_in_channels=cnn_out_channels_actual,
            hybrid_cnn_output_h=cnn_output_h_feat,
            hybrid_cnn_output_w=cnn_output_w_feat,
            **vit_kwargs
        )

        vit_backend_total_params = sum(p.numel() for p in self.transformer_backend.parameters())
        vit_backend_trainable_params = sum(p.numel() for p in self.transformer_backend.parameters() if p.requires_grad)
        logger.info(f"ViT Backend part: "
                    f"Trainable params: {vit_backend_trainable_params / 1e6:.2f}M / Total params: {vit_backend_total_params / 1e6:.2f}M")

        total_params = cnn_total_params + vit_backend_total_params
        trainable_params = cnn_trainable_params + vit_backend_trainable_params
        percentage_trainable = (100 * trainable_params / total_params) if total_params > 0 else 0.0

        logger.info(f"HybridViT Total: "
                    f"Trainable params: {trainable_params / 1e6:.2f}M / Total params: {total_params / 1e6:.2f}M "
                    f"({percentage_trainable:.2f}%)")

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.transformer_backend(features)
        return output

