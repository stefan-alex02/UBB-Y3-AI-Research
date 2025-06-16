from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=1, se_ratio=0.25, use_se=True):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))

        # Depthwise convolution
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size // 2, groups=hidden_dim,
                      bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU(inplace=True))

        # Squeeze and Excitation layer
        if use_se:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            layers.append(SELayer(hidden_dim, squeeze_channels))

        # Projection phase
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class PaperCNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor based on Table 1 from Li et al. 2022
    "A Novel Method for Ground-Based Cloud Image Classification Using Transformer"
    Outputs feature maps of size H/8 x W/8 x 48 (e.g., 56x56x48 for 448x448 input)
    """

    def __init__(self, in_channels=3, num_classes: Optional[int] = None):
        super().__init__()

        # Stride 1 conv to get 48 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=1, padding=1, bias=False),  # Input 448x448x3 -> 448x448x48
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        # MBConv1 (3x3), output 24 channels, no downsampling
        self.mbconv1_blocks = self._make_mbconv_layer(MBConvBlock, 48, 24, blocks=3, kernel_size=3, stride=1,
                                                      expand_ratio=1)  # 448x448x24

        # MBConv6 (3x3), output 40 channels, downsample (stride 2)
        self.mbconv2_blocks = self._make_mbconv_layer(MBConvBlock, 24, 40, blocks=5, kernel_size=3, stride=2,
                                                      expand_ratio=6)  # 224x224x40

        # MBConv6 (5x5), output 64 channels, downsample (stride 2)
        self.mbconv3_blocks = self._make_mbconv_layer(MBConvBlock, 40, 64, blocks=5, kernel_size=5, stride=2,
                                                      expand_ratio=6)  # 112x112x64

        self.mbconv4_blocks = self._make_mbconv_layer(MBConvBlock, 64, 96, blocks=1, kernel_size=5, stride=2,
                                                      expand_ratio=6)  # 56x56x96 (intermediate)

        self.mbconv5_blocks = self._make_mbconv_layer(MBConvBlock, 96, 128, blocks=1, kernel_size=3, stride=2,
                                                      expand_ratio=6)  # Output: 14x14x128

        # Final Conv (1x1) to get 48 output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 48, kernel_size=1, stride=1, padding=0, bias=False),  # 14x14x48
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True)
        )

        self.output_channels = 48

        self.num_classes_for_standalone = num_classes
        if self.num_classes_for_standalone is not None and self.num_classes_for_standalone > 0:
            self.standalone_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.standalone_head = nn.Linear(self.output_channels, self.num_classes_for_standalone)
        else:
            self.standalone_pool = None
            self.standalone_head = None

    def _make_mbconv_layer(self, block, in_channels, out_channels, blocks, kernel_size, stride, expand_ratio):
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size, stride, expand_ratio))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size, 1, expand_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = checkpoint(self.mbconv1_blocks, x,
                       use_reentrant=False)
        x = checkpoint(self.mbconv2_blocks, x, use_reentrant=False)
        x = checkpoint(self.mbconv3_blocks, x, use_reentrant=False)
        x = checkpoint(self.mbconv4_blocks, x, use_reentrant=False)
        x = checkpoint(self.mbconv5_blocks, x, use_reentrant=False)
        features = checkpoint(self.final_conv, x, use_reentrant=False) # Output: (B, self.output_channels, H_feat, W_feat)

        if self.standalone_head is not None:
            pooled_features = checkpoint(self.standalone_pool, features, use_reentrant=False)
            pooled_features = checkpoint(torch.flatten, pooled_features, 1, use_reentrant=False)
            logits = checkpoint(self.standalone_head, pooled_features, use_reentrant=False)
            return logits
        else:
            return features

    def load_pretrained_weights(self, path: str):
        try:
            state_dict_loaded = torch.load(path, map_location='cpu')

            actual_state_to_load = {}
            if isinstance(state_dict_loaded, dict):
                actual_state_to_load = state_dict_loaded
            elif isinstance(state_dict_loaded, nn.Module):
                actual_state_to_load = state_dict_loaded.state_dict()
            else:
                print(f"Warning: Pretrained weights file {path} is of unexpected type: {type(state_dict_loaded)}")
                return

            if self.standalone_head is None:
                filtered_state_dict = {
                    k: v for k, v in actual_state_to_load.items()
                    if not k.startswith('standalone_head.') and not k.startswith('standalone_pool.')
                }

                current_model_state_dict = self.state_dict()
                new_state_dict = {}
                loaded_count = 0
                for k, v in filtered_state_dict.items():
                    if k in current_model_state_dict and current_model_state_dict[k].shape == v.shape:
                        new_state_dict[k] = v
                        loaded_count += 1

                if loaded_count > 0:
                    self.load_state_dict(new_state_dict, strict=False)
                    print(
                        f"Successfully loaded {loaded_count} weight tensors for PaperCNNFeatureExtractor backbone from {path}")
                else:
                    print(
                        f"Warning: No matching weights found or loaded for PaperCNNFeatureExtractor backbone from {path}")

            else:  # Standalone head exists
                self.load_state_dict(actual_state_to_load, strict=True)
                print(
                    f"Successfully loaded weights for PaperCNNFeatureExtractor (including standalone head) from {path}")

        except Exception as e:
            print(f"Error loading pretrained weights for PaperCNNFeatureExtractor from {path}: {e}")
            raise


