# train.py
"""
Main training script for 3D CNN drone detection.
Coordinates dataset loading, model creation, and training loop.
"""

import argparse
import torch
import sys

# Add src to path for imports
sys.path.append('..')

from datasets.tubelet_dataset import create_dataloaders
from models import get_r3d18
from training.trainer import Trainer
from utils.config import load_config, get_training_config, print_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train 3D CNN for drone detection')
    parser.add_argument('--exp_config', default='configs/experiment.yaml',
                        help='Path to experiment config')
    parser.add_argument('--clf_config', default='configs/classifier.yaml',
                        help='Path to classifier config')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    return parser.parse_args()


def setup_device(gpu_id):
    """Setup training device."""
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(gpu_id)
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def main():
    # Parse arguments
    args = parse_args()

    print("=== 3D CNN Drone Detection Training ===")

    # Load configurations
    exp_config = load_config(args.exp_config)
    clf_config = load_config(args.clf_config)
    training_config = get_training_config(exp_config, clf_config)

    print_config(training_config, "Training Configuration")

    # Setup device
    device = setup_device(args.gpu)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(
        train_csv=training_config['train_csv'],
        val_csv=training_config['val_csv'],
        root_dir=training_config['tubelets_root'],
        batch_size=training_config['batch_size'],
        num_workers=training_config['num_workers']
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating R3D-18 model...")
    model = get_r3d18(
        num_classes=training_config['num_classes'],
        pretrained=training_config['pretrained'],
        freeze_backbone=training_config.get('freeze_backbone', False),
        dropout_rate=training_config.get('dropout_rate', 0.5),
        device=device.type  # Pass device type string ('cuda' or 'cpu')
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, training_config)

    # Start training
    results = trainer.fit()

    print(f"\nTraining Results:")
    print(f"  Best F1 Score: {results['best_f1']:.4f}")
    print(f"  Final Train Loss: {results['train_losses'][-1]:.4f}")
    print(f"  Final Val Loss: {results['val_losses'][-1]:.4f}")


if __name__ == "__main__":
    main()