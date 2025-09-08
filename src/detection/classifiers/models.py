# src/detection/classifier/models.py
"""
3D CNN classifier models.
Supported:
  - R3D18        (ResNet3D-18 backbone from torchvision)
  - R2Plus1D18   (R(2+1)D-18 backbone from torchvision)
"""

import torch
import torch.nn as nn
import torchvision.models.video as tv_models


class R3D18(nn.Module):
    """Wrapper for torchvision's r3d_18."""
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = tv_models.r3d_18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def from_config(cls, cfg: dict):
        ov = cfg.get("classifier", {}).get("overrides", {})
        num_classes = ov.get("num_classes", 2)
        pretrained  = ov.get("pretrained", True)
        return cls(num_classes=num_classes, pretrained=pretrained)


class R2Plus1D18(nn.Module):
    """Wrapper for torchvision's r2plus1d_18 (R(2+1)D-18)."""
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = tv_models.r2plus1d_18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @classmethod
    def from_config(cls, cfg: dict):
        ov = cfg.get("classifier", {}).get("overrides", {})
        num_classes = ov.get("num_classes", 2)
        pretrained  = ov.get("pretrained", True)
        return cls(num_classes=num_classes, pretrained=pretrained)
