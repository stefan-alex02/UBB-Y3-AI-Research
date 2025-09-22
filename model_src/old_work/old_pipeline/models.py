from torchvision import models
import torch.nn as nn

# Define a basic model (e.g., a simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x