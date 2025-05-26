import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

from .pretrained_swin import PretrainedSwin  # Your existing PretrainedSwin
from ..logger_utils import logger


class MBConvConfig:
    # Simplified config for MBConv block based on paper's usage
    def __init__(self, input_channels, output_channels, kernel_size, stride, expand_ratio, se_ratio=0.25,
                 dropout_rate=0.0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.dropout_rate = dropout_rate


class MBConv(nn.Module):
    def __init__(self, cfg: MBConvConfig, norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU):  # SiLU is Swish
        super().__init__()
        self.cfg = cfg
        self.use_residual = (cfg.stride == 1 and cfg.input_channels == cfg.output_channels)

        expanded_channels = cfg.input_channels * cfg.expand_ratio

        layers = []
        # Expansion phase (if expand_ratio > 1)
        if cfg.expand_ratio != 1:
            layers.extend([
                nn.Conv2d(cfg.input_channels, expanded_channels, kernel_size=1, bias=False),
                norm_layer(expanded_channels),
                act_layer(inplace=True)
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=cfg.kernel_size,
                      stride=cfg.stride, padding=cfg.kernel_size // 2, groups=expanded_channels, bias=False),
            norm_layer(expanded_channels),
            act_layer(inplace=True)
        ])

        # Squeeze-and-excitation
        if cfg.se_ratio > 0:
            squeeze_channels = max(1, int(cfg.input_channels * cfg.se_ratio))  # SE on input_channels of block
            layers.append(SqueezeExcitation(input_channels=expanded_channels, squeeze_channels=squeeze_channels,
                                            activation=act_layer))

        # Projection phase
        layers.extend([
            nn.Conv2d(expanded_channels, cfg.output_channels, kernel_size=1, bias=False),
            norm_layer(cfg.output_channels)
        ])

        if cfg.dropout_rate > 0:  # Paper doesn't mention dropout in MBConv, but common
            layers.append(nn.Dropout(cfg.dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_residual:
            result += x
        return result


class PaperCNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()

        # Layer details from Table 1 of the paper
        # 1. Conv (3x3), out 48, size 224x224
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(48),
            act_layer(inplace=True)
        )
        current_channels = 48

        # 2. MBConv1 (3x3) x3, out 24, size 224x224
        mbconv1_layers = []
        mbconv1_cfg = MBConvConfig(current_channels, 24, 3, 1, 1)  # MBConv1 -> expand_ratio = 1
        mbconv1_layers.append(MBConv(mbconv1_cfg, norm_layer, act_layer))
        current_channels = 24
        for _ in range(2):  # 2 more, total 3
            mbconv1_cfg = MBConvConfig(current_channels, 24, 3, 1, 1)
            mbconv1_layers.append(MBConv(mbconv1_cfg, norm_layer, act_layer))
        self.mbconv1_stack = nn.Sequential(*mbconv1_layers)

        # 3. MBConv6 (3x3) x5, out 40, size 112x112
        mbconv6_s2_layers = []
        # First block has stride 2 to get to 112x112
        mbconv6_cfg_s2 = MBConvConfig(current_channels, 40, 3, 2, 6)  # MBConv6 -> expand_ratio = 6
        mbconv6_s2_layers.append(MBConv(mbconv6_cfg_s2, norm_layer, act_layer))
        current_channels = 40
        for _ in range(4):  # 4 more, total 5
            mbconv6_cfg = MBConvConfig(current_channels, 40, 3, 1, 6)
            mbconv6_s2_layers.append(MBConv(mbconv6_cfg, norm_layer, act_layer))
        self.mbconv6_stack_s2 = nn.Sequential(*mbconv6_s2_layers)

        # 4. MBConv6 (5x5) x5, out 64, size 56x56
        mbconv6_s4_layers = []
        # First block has stride 2 to get to 56x56
        mbconv6_cfg_s4 = MBConvConfig(current_channels, 64, 5, 2, 6)
        mbconv6_s4_layers.append(MBConv(mbconv6_cfg_s4, norm_layer, act_layer))
        current_channels = 64
        for _ in range(4):  # 4 more, total 5
            mbconv6_cfg = MBConvConfig(current_channels, 64, 5, 1, 6)
            mbconv6_s4_layers.append(MBConv(mbconv6_cfg, norm_layer, act_layer))
        self.mbconv6_stack_s4 = nn.Sequential(*mbconv6_s4_layers)

        # 5. Conv (1x1), out 48, size 56x56
        self.conv_final_features = nn.Sequential(
            nn.Conv2d(current_channels, 48, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(48),
            act_layer(inplace=True)
        )
        self.output_channels = 48  # Final output channels for Swin

        logger.info(f"PaperCNNFeatureExtractor initialized. Output channels: {self.output_channels}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.mbconv1_stack(x)
        x = self.mbconv6_stack_s2(x)
        x = self.mbconv6_stack_s4(x)
        x = self.conv_final_features(x)
        # Expected output shape: (B, 48, 56, 56) for a 224x224 input
        return x


class HybridCNNRSModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 cnn_in_channels: int = 3,
                 # Swin specific params (passed to PretrainedSwin)
                 swin_model_variant: str = 'swin_t',
                 swin_pretrained: bool = True,
                 swin_num_stages_to_unfreeze: int = 1,
                 swin_head_dropout_rate: float = 0.0
                 ):
        super().__init__()
        self.cnn_feature_extractor = PaperCNNFeatureExtractor(in_channels=cnn_in_channels)

        # The Swin transformer part
        # We need to tell PretrainedSwin it's receiving feature maps
        self.swin_transformer = PretrainedSwin(
            num_classes=num_classes,  # Swin's head will be the final classifier
            swin_model_variant=swin_model_variant,
            pretrained=swin_pretrained,
            num_stages_to_unfreeze=swin_num_stages_to_unfreeze,
            head_dropout_rate=swin_head_dropout_rate,
            # New parameters for feature map input
            input_is_feature_map=True,
            feature_map_input_channels=self.cnn_feature_extractor.output_channels
        )
        logger.info(f"HybridCNNRSModel initialized with PaperCNNFeatureExtractor and Swin-{swin_model_variant}.")

    def forward(self, x):
        # x is raw image, e.g. (B, 3, 448, 448)
        features = self.cnn_feature_extractor(
            x)  # (B, 48, H_feat, W_feat) -> e.g. (B, 48, 56, 56) if input 224, or (B, 48, 112, 112) if input 448
        # The PretrainedSwin will handle the feature maps
        # Note: The paper implies their CNN output for a 448x448 input would still be 56x56.
        # This means their CNN has an effective stride of 8 (448/56 = 8).
        # My PaperCNNFeatureExtractor has effective stride of 4 (224/56 = 4).
        # If input is 448x448 to my CNN, output will be (48, 112, 112).
        # The Swin-T patch_embed replacement should correctly handle this if input to Swin is (B, 48, 112, 112).
        # Swin-T internally operates on sequences of patches. The number of "patches" for Swin will be H_feat * W_feat.
        # The positional embeddings in Swin might need to be interpolated if H_feat, W_feat is not what Swin was pre-trained on (typically 7x7 for final output of a 224 input ViT/Swin).
        # However, Swin-T first stage produces 56x56 feature maps from a 224x224 image.
        # The paper's Swin gets 56x56 features. This implies the input *image* to their CNN->Swin system might be 224x224 for the Swin part to work as expected without large pos_embed interpolation, OR their "Linear Embedding and Block (4x)" means the features entering Swin are 56x56.
        # The paper states: "all images were uniformly adjusted to 448 × 448 pixels."
        # Then, their CNN (Table 1) outputs 56x56. This implies their CNN has total stride 8.
        # My CNN: conv1 (s1), mbconv1 (s1), mbconv6_s2 (first block s2, rest s1), mbconv6_s4 (first block s2, rest s1). Total stride = 1*1*2*2 = 4.
        # So for 448x448 input, my CNN output is 48x112x112.
        # For 224x224 input, my CNN output is 48x56x56. This matches the paper's Swin input size.
        # Let's assume the pipeline `img_size` will be set to (224,224) when using this hybrid model.

        output = self.swin_transformer(features)
        return output

    # Expose parameters for Skorch to tune (if desired for the hybrid model)
    # These would be prefixed with 'module__cnn_feature_extractor__<param>' or 'module__swin_transformer__<param>'
    # For example, if you want to tune swin_head_dropout_rate: 'module__swin_transformer__head_dropout_rate'
