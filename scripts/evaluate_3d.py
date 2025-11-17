#!/usr/bin/env python3
"""
Evaluation script for 3D models (R3D-18)
Evaluates trained models on test set.
"""
import argparse
import sys
from pathlib import Path
import torch
import yaml
import numpy as np
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from utils.config import load_config
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate 3D drone detection model')

    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                        help='Output directory for results')
    parser.add_argument('--model-name', type=str, default='r3d18',
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')

    return parser.parse_args()


def load_model(model_name, config, checkpoint_path, device):
    """Load model from checkpoint."""
    # Create model
    model = create_model(model_name, config['model'])

    # Load checkpoint
    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"[INFO] Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_f1' in checkpoint:
        print(f"[INFO] Checkpoint best F1: {checkpoint['best_f1']:.4f}")

    return model


def evaluate_model(model, dataloader, device):
    """Run evaluation on dataset."""
    all_preds = []
    all_labels = []
    all_probs = []

    print("\n[INFO] Running inference...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx}/{len(dataloader)} batches")

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['per_class'] = {
        'precision': precision_per_class.tolist(),
        'recall': recall_per_class.tolist(),
        'f1_score': f1_per_class.tolist()
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def print_results(metrics):
    """Print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    print("\nPer-Class Metrics:")
    print("  Class 0 (Background):")
    print(f"    Precision: {metrics['per_class']['precision'][0]:.4f}")
    print(f"    Recall:    {metrics['per_class']['recall'][0]:.4f}")
    print(f"    F1-Score:  {metrics['per_class']['f1_score'][0]:.4f}")

    print("  Class 1 (Drone):")
    print(f"    Precision: {metrics['per_class']['precision'][1]:.4f}")
    print(f"    Recall:    {metrics['per_class']['recall'][1]:.4f}")
    print(f"    F1-Score:  {metrics['per_class']['f1_score'][1]:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print("              Predicted")
    print("             Neg    Pos")
    print(f"Actual Neg  {cm[0, 0]:5d}  {cm[0, 1]:5d}")
    print(f"       Pos  {cm[1, 0]:5d}  {cm[1, 1]:5d}")
    print("=" * 70)


def save_results(output_dir, metrics, y_true, y_pred, y_probs, args):
    """Save evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to: {metrics_file}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_file = output_dir / 'predictions.npz'
        np.savez(
            predictions_file,
            labels=y_true,
            predictions=y_pred,
            probabilities=y_probs
        )
        print(f"[INFO] Predictions saved to: {predictions_file}")

    # Save summary text file
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(y_true)}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write("\n" + "=" * 70 + "\n")
    print(f"[INFO] Summary saved to: {summary_file}")


def main():
    """Main evaluation function."""
    args = parse_args()

    print("=" * 70)
    print("3D MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"\n[INFO] Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("\n[INFO] Using CPU")

    # Load config
    print(f"\n[INFO] Loading config: {args.config}")
    config = load_config(args.config)

    # Update config with test CSV
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['test_csv'] = args.test_csv

    # Load model
    print(f"\n[INFO] Loading model: {args.model_name}")
    model = load_model(args.model_name, config, args.checkpoint, device)

    # Load test dataset
    print(f"\n[INFO] Loading test dataset...")
    test_dataset = create_dataset(
        'tubelet',
        config['dataset'],
        split='test',
        transform=None
    )
    print(f"[INFO] Test samples: {len(test_dataset)}")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Run evaluation
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)

    # Compute metrics
    print("\n[INFO] Computing metrics...")
    metrics = compute_metrics(y_true, y_pred)

    # Print results
    print_results(metrics)

    # Save results
    save_results(args.output_dir, metrics, y_true, y_pred, y_probs, args)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()