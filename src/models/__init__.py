"""
Model module for drone detection.
Provides factory functions for creating 2D and 3D models.
"""
from .resnet18_classifier import (
    ResNet18Classifier,
    create_resnet18_model,
    load_resnet18_checkpoint,
    save_resnet18_model
)
from .r3d_models import (
    R3D18,
    get_r3d18
)

# Model registry
MODEL_REGISTRY = {
    'resnet18': ResNet18Classifier,
    'r3d18': R3D18,
    'r2plus1d': None,  # To be implemented
    'mc3': None,  # To be implemented
}


def create_model(model_name, config):
    """
    Factory function to create models.

    Args:
        model_name: Name of the model ('resnet18', 'r3d18', etc.)
        config: Model configuration dictionary

    Returns:
        Model instance

    Example:
        >>> config = {'num_classes': 2, 'pretrained': True}
        >>> model = create_model('resnet18', config)
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join([k for k, v in MODEL_REGISTRY.items() if v is not None])
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    if model_class is None:
        raise NotImplementedError(f"Model {model_name} not yet implemented")

    # Create model based on name
    if model_name == 'resnet18':
        return model_class(
            num_classes=config.get('num_classes', 2),
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', True),
            dropout_rate=config.get('dropout_rate', 0.5)
        )

    elif model_name == 'r3d18':
        return model_class(
            num_classes=config.get('num_classes', 2),
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', False),
            dropout_rate=config.get('dropout_rate', 0.5)
        )

    else:
        # Generic creation for future models
        return model_class(**config)


def list_models():
    """List all available models."""
    return [k for k, v in MODEL_REGISTRY.items() if v is not None]


def register_model(name):
    """
    Decorator to register a new model.

    Example:
        >>> @register_model('my_model')
        >>> class MyModel(nn.Module):
        >>>     pass
    """

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


__all__ = [
    'ResNet18Classifier',
    'R3D18',
    'create_model',
    'create_resnet18_model',
    'load_resnet18_checkpoint',
    'save_resnet18_model',
    'get_r3d18',
    'list_models',
    'register_model',
    'MODEL_REGISTRY'
]