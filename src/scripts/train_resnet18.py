# scripts/train_resnet18.py
"""
Training script for ResNet-18 drone detection model.
Trains on video frames extracted from drone detection dataset.
"""

import os
import sys
import argparse
import yaml
import torch
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import create_data_loaders
from models import create_resnet18_model, save_resnet18_model


def load_video_lists(splits_dir: str):
    """Load train/val/test video lists from split files."""

    def read_video_list(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []

    train_videos = read_video_list(os.path.join(splits_dir, 'train_videos.txt'))
    val_videos = read_video_list(os.path.join(splits_dir, 'val_videos.txt'))
    test_videos = read_video_list(os.path.join(splits_dir, 'test_videos.txt'))

    print(f"Loaded video splits:")
    print(f"  - Train: {len(train_videos)} videos")
    print(f"  - Val: {len(val_videos)} videos")
    print(f"  - Test: {len(test_videos)} videos")

    return train_videos, val_videos, test_videos


def setup_training_directories(base_dir: str = "experiments/resnet18"):
    """Create directories for saving training outputs."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"{base_dir}_{timestamp}"

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


def train_resnet18(config_path: str,
                   video_dir: str = "data/raw",
                   annotations_dir: str = "data/annotations",
                   splits_dir: str = "data/splits",
                   gpu_id: int = 0):
    """
    Main training function for ResNet-18.
    """

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Starting ResNet-18 training...")
    print(f"Config: {config_path}")

    # Setup device
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load video splits
    train_videos, val_videos, test_videos = load_video_lists(splits_dir)

    if not train_videos:
        raise ValueError("No training videos found!")

    # Create data loaders
    print("\nCreating data loaders...")
    cnn_config = config.get('CNN', {})

    train_loader, val_loader = create_data_loaders(
        train_videos=train_videos,
        val_videos=val_videos,
        video_dir=video_dir,
        annotations_dir=annotations_dir,
        batch_size=cnn_config.get('batch_size', 32),
        input_size=cnn_config.get('input_size', 112),
        num_workers=4
    )

    print(f"Data loaders created:")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating ResNet-18 model...")
    model = create_resnet18_model(config)
    model.to(device)

    # Setup training directories
    exp_dir = setup_training_directories()

    # Save configuration
    config_save_path = os.path.join(exp_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Prepare trainer config
    trainer_config = {
        'epochs': cnn_config.get('epochs', 25),
        'lr': cnn_config.get('lr', 1e-4),
        'weight_decay': cnn_config.get('weight_decay', 1e-3),
        'save_dir': os.path.join(exp_dir, "checkpoints"),
        'patience': cnn_config.get('patience', 20),
        'min_delta': cnn_config.get('min_delta', 0.001),
        'save_every': 5
    }

    # Create trainer (reuse your existing trainer class)
    print("\nInitializing trainer...")
    trainer = Trainer(model, train_loader, val_loader, trainer_config)

    # Start training
    print("\n" + "=" * 50)
    print("STARTING RESNET-18 TRAINING")
    print("=" * 50)

    results = trainer.fit()

    # Save final results
    results_path = os.path.join(exp_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best F1-score: {results['best_f1']:.4f}")
    print(f"Results saved to: {exp_dir}")

    # Save final model
    final_model_path = os.path.join(exp_dir, "resnet18_final.pth")
    save_resnet18_model(
        model=model,
        filepath=final_model_path,
        epoch=trainer_config['epochs'],
        metrics={'best_f1': results['best_f1']}
    )

    return exp_dir, results


def fine_tune_resnet18(initial_checkpoint: str,
                       config_path: str,
                       video_dir: str = "data/raw",
                       annotations_dir: str = "data/annotations",
                       splits_dir: str = "data/splits"):
    """
    Fine-tune ResNet-18 with unfrozen backbone after initial training.
    """

    print("Starting ResNet-18 fine-tuning...")

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load video splits
    train_videos, val_videos, _ = load_video_lists(splits_dir)

    # Create data loaders
    cnn_config = config.get('CNN', {})
    train_loader, val_loader = create_data_loaders(
        train_videos, val_videos, video_dir, annotations_dir,
        batch_size=cnn_config.get('batch_size', 32),
        input_size=cnn_config.get('input_size', 112)
    )

    # Load pre-trained model
    print(f"Loading checkpoint: {initial_checkpoint}")
    checkpoint = torch.load(initial_checkpoint)
    model = create_resnet18_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Unfreeze for fine-tuning
    model.unfreeze_backbone()

    # Setup directories
    exp_dir = setup_training_directories("experiments/resnet18_finetune")

    # Fine-tuning config (lower learning rate, fewer epochs)
    trainer_config = {
        'epochs': 10,  # Fewer epochs for fine-tuning
        'lr': 1e-5,  # Much lower learning rate
        'weight_decay': 1e-4,
        'save_dir': os.path.join(exp_dir, "checkpoints"),
        'patience': 5,
        'min_delta': 0.0001,
        'save_every': 2
    }

    # Train
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    results = trainer.fit()

    print(f"Fine-tuning completed! Best F1: {results['best_f1']:.4f}")
    return exp_dir, results


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-18 for drone detection')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--video_dir', default='data/raw', help='Directory containing videos')
    parser.add_argument('--annotations_dir', default='data/annotations', help='Directory containing annotations')
    parser.add_argument('--splits_dir', default='data/splits', help='Directory containing train/val/test splits')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--finetune', help='Path to checkpoint for fine-tuning')

    args = parser.parse_args()

    try:
        if args.finetune:
            # Fine-tuning mode
            exp_dir, results = fine_tune_resnet18(
                initial_checkpoint=args.finetune,
                config_path=args.config,
                video_dir=args.video_dir,
                annotations_dir=args.annotations_dir,
                splits_dir=args.splits_dir
            )
        else:
            # Initial training mode
            exp_dir, results = train_resnet18(
                config_path=args.config,
                video_dir=args.video_dir,
                annotations_dir=args.annotations_dir,
                splits_dir=args.splits_dir,
                gpu_id=args.gpu
            )

        print(f"\nTraining completed successfully!")
        print(f"Experiment directory: {exp_dir}")
        print(f"Best F1-score: {results['best_f1']:.4f}")

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()