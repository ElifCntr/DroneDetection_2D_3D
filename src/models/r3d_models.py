# src/training/models/r3d_models.py
"""
R3D-18 model for 3D video classification.
Processes (C,T,H,W) tubelets, outputs binary classification logits.
Supports pretrained Kinetics weights and backbone freezing.
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
from typing import Optional


class R3D18(nn.Module):
    """
    R3D-18 model for binary video classification (drone detection).

    This wraps torchvision's R3D-18 and adapts it for drone detection:
    - Changes final layer to 2 classes (drone vs background)
    - Optionally loads pretrained Kinetics-400 weights
    - Supports layer freezing for fine-tuning
    - Adds dropout for regularization
    """

    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False, dropout_rate=0.5):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use Kinetics-400 pretrained weights
            freeze_backbone: Whether to freeze backbone layers for fine-tuning
            dropout_rate: Dropout rate for regularization
        """
        super(R3D18, self).__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Load base R3D-18 model
        self.backbone = video_models.r3d_18(pretrained=pretrained)

        # Replace final classification layer with dropout
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )

        # Freeze backbone if requested (for fine-tuning approach)
        if freeze_backbone:
            self._freeze_backbone()

        self._init_info()

    def _freeze_backbone(self):
        """Freeze all layers except final classifier for feature extraction approach."""
        frozen_params = 0
        total_params = 0

        for name, param in self.backbone.named_parameters():
            total_params += param.numel()
            if 'fc' not in name:  # Freeze everything except final classifier
                param.requires_grad = False
                frozen_params += param.numel()

        print(f"Frozen backbone: {frozen_params:,} parameters out of {total_params:,}")
        print("Frozen all layers except classifier for fine-tuning")

    def _init_info(self):
        """Print model initialization information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"R3D-18 Model initialized:")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Pretrained: {self.pretrained}")
        print(f"  - Frozen backbone: {self.freeze_backbone}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, C, T, H, W)
               - B: batch size
               - C: channels (3 for RGB)
               - T: time frames (3 for tubelets)
               - H, W: spatial dimensions (112x112)

        Returns:
            Output tensor of shape (B, num_classes)
        """
        return self.backbone(x)


def get_r3d18(num_classes=2, pretrained=True, freeze_backbone=False, dropout_rate=0.5, device='cuda'):
    """
    Factory function to create and initialize R3D-18 model.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone for fine-tuning
        dropout_rate: Dropout rate for regularization
        device: Device to move model to ('cuda' or 'cpu')

    Returns:
        R3D18 model ready for training
    """
    model = R3D18(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate
    )

    # Move to device
    device_obj = torch.device(device)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device_obj)
        print(f"Model moved to GPU")
    else:
        model = model.to('cpu')
        print(f"Model running on CPU")

    return model


def test_model():
    """Test function to verify model works with expected input shapes."""
    print("Testing R3D-18 model...")

    # Create test input: batch=2, channels=3, time=3, height=112, width=112
    test_input = torch.randn(2, 3, 3, 112, 112)
    print(f"Test input shape: {test_input.shape}")

    # Create model
    model = get_r3d18(num_classes=2, pretrained=False, device='cpu')

    # Forward pass
    with torch.no_grad():
        output = model(test_input)

    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")

    assert output.shape == (2, 2), f"Expected output shape (2, 2), got {output.shape}"
    print("✓ Model test passed!")


if __name__ == "__main__":
    test_model()