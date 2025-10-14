# scripts/evaluate_resnet18.py
"""
Evaluation script for ResNet-18 drone detection model.
Tests on video frames and computes comprehensive metrics.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.frame_dataset import DroneFrameDataset
from src.models.resnet18_classifier import load_resnet18_checkpoint
from src.evaluation.frame_metrics import FrameLevelMetrics


def load_test_videos(splits_dir: str):
    """Load test video list."""
    test_file = os.path.join(splits_dir, 'test_videos.txt')
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test dataset.
    Returns predictions, probabilities, and ground truth labels.
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("Running evaluation...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            # Get predictions and probabilities
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches")

    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)


def compute_detailed_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive evaluation metrics."""

    # Basic metrics using FrameLevelMetrics
    metrics = FrameLevelMetrics.calculate_all_metrics(y_true, y_pred)

    # Drone class probabilities (class 1)
    drone_probabilities = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob

    # Threshold analysis
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = FrameLevelMetrics.evaluate_multiple_thresholds(
        y_true, drone_probabilities, thresholds.tolist()
    )

    # Find optimal threshold
    optimal_threshold, best_f1 = FrameLevelMetrics.find_optimal_threshold(
        threshold_results, metric='f1_score'
    )

    return metrics, threshold_results, optimal_threshold, best_f1


def create_evaluation_plots(y_true, y_pred, y_prob, save_dir):
    """Create evaluation plots and save them."""

    os.makedirs(save_dir, exist_ok=True)

    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ResNet-18')
    plt.colorbar()

    classes = ['No Drone', 'Drone']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Probability Distribution
    plt.figure(figsize=(12, 5))

    drone_probs = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob

    plt.subplot(1, 2, 1)
    no_drone_probs = drone_probs[y_true == 0]
    drone_class_probs = drone_probs[y_true == 1]

    plt.hist(no_drone_probs, bins=30, alpha=0.7, label='No Drone', color='blue')
    plt.hist(drone_class_probs, bins=30, alpha=0.7, label='Drone', color='red')
    plt.xlabel('Drone Probability')
    plt.ylabel('Count')
    plt.title('Probability Distribution by True Class')
    plt.legend()

    # 3. Threshold Analysis
    plt.subplot(1, 2, 2)
    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = FrameLevelMetrics.evaluate_multiple_thresholds(
        y_true, drone_probs, thresholds.tolist()
    )

    plt.plot(threshold_results['threshold'], threshold_results['precision'], 'o-', label='Precision')
    plt.plot(threshold_results['threshold'], threshold_results['recall'], 's-', label='Recall')
    plt.plot(threshold_results['threshold'], threshold_results['f1_score'], '^-', label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {save_dir}")


def print_evaluation_summary(metrics, optimal_threshold, best_f1):
    """Print formatted evaluation summary."""

    print("\n" + "=" * 60)
    print("RESNET-18 EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nPERFORMANCE METRICS (Default threshold=0.5):")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")

    print(f"\nCONFUSION MATRIX:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")

    print(f"\nOPTIMIZED PERFORMANCE:")
    print(f"  Optimal Threshold: {optimal_threshold:.2f}")
    print(f"  Best F1-Score:     {best_f1:.4f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 drone detection model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--video_dir', default='data/raw', help='Directory containing videos')
    parser.add_argument('--annotations_dir', default='data/annotations', help='Directory containing annotations')
    parser.add_argument('--splits_dir', default='data/splits', help='Directory containing train/val/test splits')
    parser.add_argument('--output_dir', default='evaluation_results', help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load test videos
    test_videos = load_test_videos(args.splits_dir)
    if not test_videos:
        raise ValueError("No test videos found!")

    print(f"Found {len(test_videos)} test videos")

    # Create test dataset
    cnn_config = config.get('CNN', {})
    test_dataset = DroneFrameDataset(
        video_list=test_videos,
        video_dir=args.video_dir,
        annotations_dir=args.annotations_dir,
        input_size=cnn_config.get('input_size', 112),
        balance_classes=False,  # Don't balance for evaluation
        frames_per_video=200  # More frames for thorough evaluation
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    print(f"Test dataset: {len(test_dataset)} frames")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_resnet18_checkpoint(args.checkpoint)
    model.to(device)

    # Run evaluation
    y_pred, y_prob, y_true = evaluate_model(model, test_loader, device)

    # Compute metrics
    metrics, threshold_results, optimal_threshold, best_f1 = compute_detailed_metrics(
        y_true, y_pred, y_prob
    )

    # Print results
    print_evaluation_summary(metrics, optimal_threshold, best_f1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save detailed results
    results = {
        'default_metrics': metrics,
        'optimal_threshold': float(optimal_threshold),
        'best_f1_score': float(best_f1),
        'test_samples': len(y_true),
        'model_checkpoint': args.checkpoint
    }

    import json
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save threshold analysis
    threshold_results.to_csv(os.path.join(args.output_dir, 'threshold_analysis.csv'), index=False)

    # Create plots
    create_evaluation_plots(y_true, y_pred, y_prob, args.output_dir)

    # Generate classification report
    target_names = ['No Drone', 'Drone']
    report = classification_report(y_true, y_pred, target_names=target_names)

    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write("ResNet-18 Drone Detection - Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write(f"\n\nOptimal Threshold: {optimal_threshold:.3f}\n")
        f.write(f"Best F1-Score: {best_f1:.4f}\n")

    print(f"\nDetailed results saved to: {args.output_dir}")
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()