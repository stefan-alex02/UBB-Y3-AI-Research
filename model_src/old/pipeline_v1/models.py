import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights # Example ViT

from utils import logger

class DummyCNN(nn.Module):
    """A simple Convolutional Neural Network for demonstration."""
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # H/2, W/2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # H/4, W/4
        # Calculate flattened size dynamically? Or assume fixed input size (e.g., 224)
        # Assuming 224x224 input -> 224/4 = 56x56 output from pool2
        # flattened_size = 32 * 56 * 56 # Needs adjustment based on actual input size after transforms
        self.adapool = nn.AdaptiveAvgPool2d((7, 7)) # Makes it robust to input size variations
        flattened_size = 32 * 7 * 7
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adapool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class VisionTransformerModel(nn.Module):
    """Wrapper for a pre-trained Vision Transformer."""
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Replace the classifier head
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)

class DiffusionClassifierPlaceholder(nn.Module):
    """
    Placeholder for a Diffusion-based Classifier.
    NOTE: A real diffusion classifier is significantly more complex.
    This acts as a standard CNN for structural compatibility.
    """
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        # Using the same structure as DummyCNN for demonstration
        # Replace this with an actual diffusion model architecture if available
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adapool = nn.AdaptiveAvgPool2d((7, 7))
        flattened_size = 32 * 7 * 7
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        print("⚠️ WARNING: Using DiffusionClassifierPlaceholder which is a simple CNN, not a true diffusion model.")


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adapool(x)
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Factory function to get models by name
def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Creates a model instance based on its name."""
    if model_name.lower() == 'dummycnn':
        return DummyCNN(num_classes=num_classes)
    elif model_name.lower() == 'vit':
        return VisionTransformerModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'diffusion_placeholder':
         # Warn user this isn't a real diffusion model
        logger.warning("⚠️ Selected 'diffusion_placeholder'. This uses a simple CNN structure, not a real diffusion model.")
        return DiffusionClassifierPlaceholder(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")