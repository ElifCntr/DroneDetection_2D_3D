#!/usr/bin/env python3
"""
Training script for 2D models (ResNet-18) using region datasets.
Trains on pre-extracted 2D regions from center frames.
"""
import argparse
import sys
from pathlib import Path
import torch
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import create_model
from training.trainer import Trainer
from utils.config import load_config
from utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train 2D drone detection models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--train-csv',
        type=str,
        required=True,
        help='Path to training region index CSV file'
    )

    parser.add_argument(
        '--val-csv',
        type=str,
        required=True,
        help='Path to validation region index CSV file'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Model architecture name'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Base directory to save models'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (default: model_name_timestamp)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{args.model_name}_{timestamp}"

    print("=" * 70)
    print("2D MODEL TRAINING")
    print("=" * 70)
    print(f"Experiment: {exp_name}")
    print(f"Model: {args.model_name}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Val CSV: {args.val_csv}")
    print("=" * 70)

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"\n[INFO] Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("\n[INFO] Using CPU")

    # Load configuration
    print(f"\n[INFO] Loading config: {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Create output directory
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name=exp_name,
        log_file=str(log_dir / 'training.log')
    )

    logger.info(f"Starting training: {exp_name}")
    logger.info(f"Configuration: {config}")

    # Create datasets using 2D-specific loader
    print("\n[INFO] Loading datasets...")
    from datasets.region_dataset import create_2d_dataloaders

    train_loader, val_loader = create_2d_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4)
    )

    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples: {len(val_loader.dataset)}")

    # Create model
    print(f"\n[INFO] Creating model: {args.model_name}")
    model = create_model(args.model_name, config['model'])
    model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    # Create trainer
    print("\n[INFO] Initializing trainer...")

    trainer_config = {
        'epochs': config['training']['epochs'],
        'lr': config['training']['learning_rate'],
        'weight_decay': config['training'].get('weight_decay', 1e-4),
        'save_dir': str(output_dir / 'checkpoints'),
        'patience': config['training'].get('patience', 10),
        'min_delta': config['training'].get('min_delta', 0.001),
        'save_every': config['training'].get('save_every', 5)
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n[INFO] Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    results = trainer.fit()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best F1-Score: {results['best_f1']:.4f}")
    print(f"Final Train Loss: {results['train_losses'][-1]:.4f}")
    print(f"Final Val Loss: {results['val_losses'][-1]:.4f}")
    print(f"Model saved to: {output_dir / 'checkpoints'}")
    print("=" * 70)

    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()