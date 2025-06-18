import torch
import torch.nn as nn

from ..logger_utils import logger


class CloudNet(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.5):
        super().__init__()
        self.dropout_p = dropout_p

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), # Input (3, 227, 227) -> (96, 55, 55)
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (96, 27, 27)

            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding='same'), # (256, 27, 27)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (256, 13, 13)

            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding='same'), # (384, 13, 13)
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # No pooling after Conv3

            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding='same'), # (256, 13, 13)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # No pooling after Conv4

            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'), # (256, 13, 13)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # (256, 6, 6)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        logger.info(f"CloudNet initialized for {num_classes} classes with dropout {self.dropout_p}.")

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d): # For FC layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
