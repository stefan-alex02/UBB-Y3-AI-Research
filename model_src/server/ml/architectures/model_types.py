from enum import Enum

from .paper_xception_mobilenet import XceptionBasedCloudNet, MobileNetBasedCloudNet
from .feature_extractors import PaperCNNFeatureExtractor
from .hybrid_models import HybridViT
from .cnn import SimpleCNN
from .diffusion_classifier import DiffusionClassifier
from .pretrained_swin import PretrainedSwin
from .pretrained_vit import PretrainedViT
from .scratch_vit import ScratchViT

model_mapping = {
    "cnn": SimpleCNN,
    "pvit": PretrainedViT,
    "svit": ScratchViT,

    "swin": PretrainedSwin,

    "diff": DiffusionClassifier,

    "hvit": HybridViT,  # New entry for dedicated hybrid model
    "cnn_feat": PaperCNNFeatureExtractor,  # If you want to train/test CNN extractor alone

    "xcloud": XceptionBasedCloudNet,
    "mcloud": MobileNetBasedCloudNet,
}


class ModelType(str, Enum):  # Inheriting from str makes it directly usable as a string
    CNN = "cnn"
    PRETRAINED_VIT = "pvit"
    SCRATCH_VIT = "svit"
    PRETRAINED_SWIN = "swin"
    DIFFUSION = "diff"

    HYBRID_VIT = "hvit"
    CNN_FEAT = "cnn_feat"

    XCLOUD = "xcloud"
    MCLOUD = "mcloud"

    def get_model_class(self):
        """Returns the model class associated with the model type."""
        return model_mapping.get(self.value, None)

    @classmethod
    def _missing_(cls, value):  # Optional: for case-insensitive matching or aliases
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return super()._missing_(value)

    def __str__(self):  # Ensures that str(ModelType.CNN) is "cnn"
        return self.value
