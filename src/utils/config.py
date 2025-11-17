"""
Configuration loading and validation utilities.
Loads YAML configs and provides default values.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"[INFO] Loaded config from: {config_path}")
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.

    Args:
        *configs: Variable number of config dictionaries

    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        _deep_update(merged, config)
    return merged


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a dictionary.

    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates

    Returns:
        Updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def get_training_config(exp_config: dict, clf_config: dict) -> Dict[str, Any]:
    """
    Extract training configuration from experiment and classifier configs.

    Args:
        exp_config: Main experiment configuration
        clf_config: Classifier-specific configuration

    Returns:
        dict: Training configuration
    """
    # Get R3D18 specific config
    r3d_config = clf_config.get('R3D18', {})

    # Extract paths
    paths = exp_config.get('paths', {})

    training_config = {
        # Model parameters
        'num_classes': r3d_config.get('num_classes', 2),
        'pretrained': r3d_config.get('pretrained', True),
        'freeze_backbone': r3d_config.get('freeze_backbone', False),
        'dropout_rate': r3d_config.get('dropout_rate', 0.5),

        # Training parameters
        'epochs': r3d_config.get('epochs', 30),
        'batch_size': r3d_config.get('batch_size', 32),
        'lr': r3d_config.get('lr', 1e-4),
        'weight_decay': r3d_config.get('weight_decay', 1e-3),

        # Regularization
        'patience': r3d_config.get('patience', 5),
        'min_delta': r3d_config.get('min_delta', 0.001),
        'save_every': r3d_config.get('save_every', 5),

        # Data paths
        'train_csv': paths.get('tubelet_indexes', {}).get('train_csv'),
        'val_csv': paths.get('tubelet_indexes', {}).get('val_csv'),
        'tubelets_root': paths.get('tubelets_dir'),
        'save_dir': paths.get('checkpoints_dir'),

        # Data loading
        'num_workers': 4,
    }

    # Validate required paths
    required_paths = ['train_csv', 'val_csv', 'tubelets_root', 'save_dir']
    for path_name in required_paths:
        if not training_config.get(path_name):
            print(f"[WARN] Missing path: {path_name}")

    return training_config


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that a config has all required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required key paths (e.g., ['model.num_classes'])

    Returns:
        True if valid, raises ValueError if not
    """
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config

        for key in keys:
            if key not in current:
                raise ValueError(f"Missing required config key: {key_path}")
            current = current[key]

    return True


def print_config(config: Dict[str, Any], title: str = "Configuration", indent: int = 0):
    """
    Print configuration in a readable format.

    Args:
        config: Configuration dictionary
        title: Title to print
        indent: Indentation level
    """
    if indent == 0:
        print(f"\n{title}:")
        print("=" * 50)

    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, title="", indent=indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

    if indent == 0:
        print("=" * 50)


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"[INFO] Config saved to: {save_path}")


__all__ = [
    'load_config',
    'merge_configs',
    'get_training_config',
    'validate_config',
    'print_config',
    'save_config'
]