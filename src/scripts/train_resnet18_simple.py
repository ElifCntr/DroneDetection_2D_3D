# scripts/train_resnet18_simple.py
"""
Simple ResNet-18 training using pre-extracted 2D regions.
Reuses your existing trainer infrastructure.
"""

import sys
import os
import torch
import yaml

# Add project root to path
sys.path.append('.')

from src.training.datasets.region_dataset import create_2d_dataloaders
from src.training.models.resnet18_classifier import create_resnet18_model
from src.training.trainer import Trainer


def print_model_details(model, config):
    """Print detailed information about model architecture and training setup."""
    print("\n" + "=" * 60)
    print("RESNET-18 MODEL CONFIGURATION")
    print("=" * 60)

    cnn_config = config.get('CNN', {})

    print(f"Architecture: ResNet-18")
    print(f"Pretrained Weights: ImageNet")
    print(f"Input Size: {cnn_config.get('input_size', 112)}x{cnn_config.get('input_size', 112)}")
    print(f"Output Classes: {cnn_config.get('num_classes', 2)} (Background, Drone)")
    print(f"Backbone Frozen: {cnn_config.get('freeze_backbone', True)}")
    print(f"Dropout Rate: {cnn_config.get('dropout_rate', 0.5)}")

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nPARAMETER COUNT:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Frozen Parameters: {frozen_params:,}")
    print(f"  Trainable Percentage: {100 * trainable_params / total_params:.1f}%")

    print(f"\nFROZEN LAYERS (ImageNet Pretrained):")
    if cnn_config.get('freeze_backbone', True):
        frozen_layers = [
            "  - conv1 (7x7, 64 filters)",
            "  - layer1 (2 ResNet blocks, 64 channels)",
            "  - layer2 (2 ResNet blocks, 128 channels)",
            "  - layer3 (2 ResNet blocks, 256 channels)",
            "  - layer4 (2 ResNet blocks, 512 channels)",
            "  - All BatchNorm layers",
            "  - All ReLU activations"
        ]
        for layer in frozen_layers:
            print(layer)
        print(f"\nTRAINABLE LAYERS (Random Init):")
        print(f"  - Final Classifier: Linear(512 → 2)")
        print(f"  - Dropout layer (rate: {cnn_config.get('dropout_rate', 0.5)})")
    else:
        print("  All layers trainable (fine-tuning mode)")


def print_training_config(trainer_config, cnn_config):
    """Print training hyperparameters."""
    print(f"\nTRAINING HYPERPARAMETERS:")
    print(f"  Epochs: {trainer_config['epochs']}")
    print(f"  Batch Size: {cnn_config.get('batch_size', 32)}")
    print(f"  Learning Rate: {trainer_config['lr']}")
    print(f"  Weight Decay: {trainer_config['weight_decay']}")
    print(f"  Early Stopping Patience: {trainer_config['patience']}")
    print(f"  Min Delta: {trainer_config['min_delta']}")

    print(f"\nOPTIMIZER & LOSS:")
    print(f"  Optimizer: Adam")
    print(f"  Loss Function: CrossEntropyLoss (with class weights)")
    print(f"  LR Scheduler: ReduceLROnPlateau")

    print(f"\nDATA AUGMENTATION (Training):")
    print(f"  - Random Horizontal Flip (50%)")
    print(f"  - ImageNet Normalization")
    print(f"  - Resize to 112x112")
    print("=" * 60)


def main():
    print("=" * 60)
    print("RESNET-18 DRONE DETECTION TRAINING")
    print("=" * 60)

    # Load config
    print("Loading configuration...")
    with open('D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\configs/classifier.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Updated paths for your system
    train_csv = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\2d_regions\train\train_2d_index.csv"
    val_csv = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\2d_regions\val\val_2d_index.csv"

    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found!")
        print("Run: python scripts/extract_2d_from_tubelets.py first")
        return

    print(f"Training Data: {train_csv}")
    print(f"Validation Data: {val_csv}")

    # Create data loaders
    print("\nCreating data loaders...")
    cnn_config = config.get('CNN', {})
    train_loader, val_loader = create_2d_dataloaders(
        train_csv, val_csv,
        batch_size=cnn_config.get('batch_size', 32)
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating ResNet-18 model...")
    model = create_resnet18_model(config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Print detailed model info
    print_model_details(model, config)

    # Create timestamped save directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = rf'D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\models\resnet18_{timestamp}'

    # Training config
    trainer_config = {
        'epochs': cnn_config.get('epochs', 25),
        'lr': cnn_config.get('lr', 1e-4),
        'weight_decay': cnn_config.get('weight_decay', 1e-3),
        'save_dir': save_dir,
        'patience': cnn_config.get('patience', 20),
        'min_delta': cnn_config.get('min_delta', 0.001),
        'save_every': 5
    }

    print_training_config(trainer_config, cnn_config)

    # Train using your existing trainer
    print("\nINITIALIZING TRAINING...")
    trainer = Trainer(model, train_loader, val_loader, trainer_config)

    print("\nSTARTING RESNET-18 TRAINING...")
    print("=" * 60)
    results = trainer.fit()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best Validation F1-score: {results['best_f1']:.4f}")
    print(f"Final Training Loss: {results['train_losses'][-1]:.4f}")
    print(f"Final Validation Loss: {results['val_losses'][-1]:.4f}")
    print(f"Model saved to: {trainer_config['save_dir']}")

    # Training summary
    print(f"\nTRAINING SUMMARY:")
    print(f"  Total Epochs: {len(results['train_losses'])}")
    print(f"  Best F1 achieved at epoch: {results['val_f1s'].index(results['best_f1']) + 1}")
    print(f"  Final Learning Rate: Check trainer logs")
    print("=" * 60)


if __name__ == "__main__":
    main()