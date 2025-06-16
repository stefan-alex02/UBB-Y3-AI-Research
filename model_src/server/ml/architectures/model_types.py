from enum import Enum

from .cnn import SimpleCNN
from .feature_extractors import PaperCNNFeatureExtractor
from .hybrid_models import HybridViT
from .paper_xception_mobilenet import XceptionBasedCloudNet, MobileNetBasedCloudNet
from .pretrained_swin import PretrainedSwin
from .pretrained_vit import PretrainedViT
from .resnet import ResNet18BasedCloud
from .scratch_vit import ScratchViT
from .shufflenet import ShuffleNetCloud
from .standard_cnn_extractor import StandardCNNFeatureExtractor

model_mapping = {
    "cnn": SimpleCNN,
    "pvit": PretrainedViT,
    "svit": ScratchViT,

    "hyvit": HybridViT,
    "cnn_feat": PaperCNNFeatureExtractor,
    "stfeat": StandardCNNFeatureExtractor,

    "xcloud": XceptionBasedCloudNet,
    "mcloud": MobileNetBasedCloudNet,
    "resnet": ResNet18BasedCloud,
    "swin": PretrainedSwin,
    "shufflenet": ShuffleNetCloud,
}


class ModelType(str, Enum):
    CNN = "cnn"
    PRETRAINED_VIT = "pvit"
    SCRATCH_VIT = "svit"

    HYBRID_VIT = "hyvit"
    CNN_FEAT = "cnn_feat"
    STANDARD_FEAT = "stfeat"

    XCLOUD = "xcloud"
    MCLOUD = "mcloud"
    PRETRAINED_SWIN = "swin"
    RESNET18_CLOUD = "resnet"
    SHUFFLE_CLOUD = "shufflenet"

    def get_model_class(self):
        """Returns the model class associated with the model type."""
        return model_mapping.get(self.value, None)

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            for member in cls:
                if member.value == value.lower():
                    return member
        return super()._missing_(value)

    def __str__(self):
        return self.value
