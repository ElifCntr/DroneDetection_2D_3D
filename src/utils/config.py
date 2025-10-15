# src/training/utils/config.py
"""
Configuration loading and validation utilities.
Loads YAML configs and provides default values.
"""

import yaml
import os
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from: {config_path}")
    return config


def get_training_config(exp_config, clf_config):
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
        'lr': r3d_config.get('lr', 1e-4),  # Lower default LR
        'weight_decay': r3d_config.get('weight_decay', 1e-3),  # Stronger regularization

        # Regularization
        'patience': r3d_config.get('patience', 5),
        'min_delta': r3d_config.get('min_delta', 0.001),
        'save_every': r3d_config.get('save_every', 5),  # Save every N epochs

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
            raise ValueError(f"Missing required path: {path_name}")

    return training_config


def print_config(config, title="Configuration"):
    """Print configuration in a readable format."""
    print(f"\n{title}:")
    for key, value in config.items():
        print(f"  {key}: {value}")