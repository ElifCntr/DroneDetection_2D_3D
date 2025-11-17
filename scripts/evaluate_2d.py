#!/usr/bin/env python3
"""
Evaluation script for 2D models (ResNet-18) on region test set.
"""
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import json
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import create_model
from utils.config import load_config
from utils.metrics import compute_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate 2D drone detection model')

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--save-predictions', action='store_true', help='Save predictions')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("2D MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: resnet18")
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

    # Load model
    print("[INFO] Loading model: resnet18")
    model = create_model('resnet18', config['model'])

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load state dict (handle key mismatch with strict=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Checkpoint best F1: {checkpoint.get('best_f1', 'unknown'):.4f}")

    # Load test dataset
    print("\n[INFO] Loading test dataset...")
    from datasets.region_dataset import RegionDataset
    import torchvision.transforms as transforms

    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = RegionDataset(args.test_csv, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Test samples: {len(test_dataset)}")

    # Run inference
    print("\n[INFO] Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    print("\n[INFO] Computing metrics...")

    # Compute basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist()
        }
    }

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")

    print("\nPer-Class Metrics:")
    print(f"  Class 0 (Background):")
    print(f"    Precision: {metrics['per_class']['precision'][0]:.4f}")
    print(f"    Recall:    {metrics['per_class']['recall'][0]:.4f}")
    print(f"    F1-Score:  {metrics['per_class']['f1_score'][0]:.4f}")
    print(f"  Class 1 (Drone):")
    print(f"    Precision: {metrics['per_class']['precision'][1]:.4f}")
    print(f"    Recall:    {metrics['per_class']['recall'][1]:.4f}")
    print(f"    F1-Score:  {metrics['per_class']['f1_score'][1]:.4f}")

    # Confusion matrix
    cm = metrics['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"             Neg    Pos")
    print(f"Actual Neg  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos  {cm[1][0]:5d}  {cm[1][1]:5d}")
    print("=" * 70)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to: {metrics_file}")

    # Save predictions
    if args.save_predictions:
        predictions_file = output_dir / 'predictions.npz'
        np.savez(
            predictions_file,
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs
        )
        print(f"[INFO] Predictions saved to: {predictions_file}")

    # Save summary
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: resnet18\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n\n")
        f.write("=" * 70 + "\n")
    print(f"[INFO] Summary saved to: {summary_file}")

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()