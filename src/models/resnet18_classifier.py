# src/models/resnet18_classifier.py
"""
ResNet-18 model for binary drone detection from single frames.
Supports transfer learning and layer freezing options.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 based binary classifier for drone detection.
    """

    def __init__(self,
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 dropout_rate: float = 0.5):
        """
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone layers (only train classifier)
            dropout_rate: Dropout rate for regularization
        """
        super(ResNet18Classifier, self).__init__()

        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        print(f"ResNet-18 initialized:")
        print(f"  - Pretrained: {pretrained}")
        print(f"  - Freeze backbone: {freeze_backbone}")
        print(f"  - Dropout rate: {dropout_rate}")
        print(f"  - Output classes: {num_classes}")

    def _freeze_backbone(self):
        """Freeze all layers except the final classifier."""
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze only the final classifier
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        print("  - Backbone layers frozen, only training final classifier")

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("  - All layers unfrozen for fine-tuning")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def get_features(self, x):
        """
        Extract features from the backbone (before final classifier).

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Feature tensor of shape (batch_size, 512)
        """
        # Forward through all layers except final classifier
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def predict_probabilities(self, x):
        """
        Get prediction probabilities using softmax.

        Args:
            x: Input tensor

        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_binary(self, x, threshold: float = 0.5):
        """
        Get binary predictions for drone detection.

        Args:
            x: Input tensor
            threshold: Threshold for binary classification

        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_probabilities(x)
        # Use probability of drone class (class 1)
        drone_probs = probabilities[:, 1]
        return (drone_probs >= threshold).long()

    def get_trainable_parameters(self):
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self):
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_info(self):
        """Print information about model parameters."""
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()

        print(f"\nModel Parameters:")
        print(f"  - Trainable: {trainable:,}")
        print(f"  - Total: {total:,}")
        print(f"  - Frozen: {total - trainable:,}")
        print(f"  - Trainable %: {100 * trainable / total:.1f}%")


def create_resnet18_model(config: dict) -> ResNet18Classifier:
    """
    Create ResNet-18 model from configuration.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        Configured ResNet-18 model
    """
    model_config = config.get('CNN', {})

    model = ResNet18Classifier(
        num_classes=model_config.get('num_classes', 2),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', True),
        dropout_rate=model_config.get('dropout_rate', 0.5)
    )

    model.print_parameter_info()

    return model


class ResNet18Trainer:
    """
    Trainer specifically for ResNet-18 model with additional features.
    """

    def __init__(self, model: ResNet18Classifier, config: dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        model_config = self.config.get('CNN', {})

        # Different learning rates for frozen vs unfrozen training
        if self.model.freeze_backbone:
            # Only train classifier when backbone is frozen
            trainable_params = [p for p in self.model.backbone.fc.parameters() if p.requires_grad]
            lr = model_config.get('lr', 1e-3)  # Higher LR for training just classifier
        else:
            # Train all parameters when unfrozen
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            lr = model_config.get('lr', 1e-4)  # Lower LR for fine-tuning

        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=model_config.get('weight_decay', 1e-3)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        print(f"Optimizer setup: LR={lr}, trainable params={len(trainable_params)}")

    def unfreeze_and_finetune(self):
        """
        Unfreeze backbone and setup for fine-tuning with lower learning rate.
        Call this after initial training with frozen backbone.
        """
        print("\nSwitching to fine-tuning mode...")
        self.model.unfreeze_backbone()
        self._setup_optimizer()  # Recreate optimizer with all parameters
        print("Ready for fine-tuning with unfrozen backbone")


# Utility functions
def load_resnet18_checkpoint(checkpoint_path: str,
                             num_classes: int = 2) -> ResNet18Classifier:
    """Load ResNet-18 model from checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model configuration from checkpoint if available
    config = checkpoint.get('config', {})
    model_config = config.get('CNN', {})

    # Create model
    model = ResNet18Classifier(
        num_classes=num_classes,
        pretrained=False,  # Don't load ImageNet weights when loading checkpoint
        freeze_backbone=model_config.get('freeze_backbone', True),
        dropout_rate=model_config.get('dropout_rate', 0.5)
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded ResNet-18 from checkpoint: {checkpoint_path}")
    model.print_parameter_info()

    return model


def save_resnet18_model(model: ResNet18Classifier,
                        filepath: str,
                        epoch: int = 0,
                        optimizer_state: Optional[dict] = None,
                        metrics: Optional[dict] = None):
    """Save ResNet-18 model with metadata."""

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': model.num_classes,
            'freeze_backbone': model.freeze_backbone,
            'dropout_rate': model.dropout_rate
        },
        'epoch': epoch,
        'trainable_params': model.get_trainable_parameters(),
        'total_params': model.get_total_parameters()
    }

    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state

    if metrics:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, filepath)
    print(f"Saved ResNet-18 model to: {filepath}")