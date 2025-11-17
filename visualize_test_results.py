"""
Visualize test set evaluation results using existing visualization utilities.
This complements the threshold analysis in utils.visualization with single-point test results.

Usage:
    python visualize_test_results.py
"""

import sys

sys.path.insert(0, 'src')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Import existing visualization module
from utils.visualization import EvaluationPlotter


def plot_test_confusion_matrix(metrics, output_path):
    """Plot confusion matrix from test results."""
    cm = np.array(metrics['confusion_matrix'])

    plt.figure(figsize=(8, 6))
    labels = ['Background', 'Drone']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})

    plt.title('Test Set Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_test_roc_curve(y_true, y_probs, output_path):
    """Plot ROC curve for test results."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Test Set ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_test_pr_curve(y_true, y_probs, output_path):
    """Plot Precision-Recall curve for test results."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR (AP = {avg_precision:.3f})')

    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2,
             linestyle='--', label=f'Baseline ({baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Test Set Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_test_metrics_bars(metrics, output_path):
    """Plot bar chart of test metrics."""
    plt.figure(figsize=(10, 6))

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Test Set Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_test_per_class(metrics, output_path):
    """Plot per-class metrics."""
    plt.figure(figsize=(10, 6))

    classes = ['Background', 'Drone']
    precision = metrics['per_class']['precision']
    recall = metrics['per_class']['recall']
    f1 = metrics['per_class']['f1_score']

    x = np.arange(len(classes))
    width = 0.25

    plt.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', color='#f39c12', alpha=0.8)

    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Test Set Per-Class Metrics', fontsize=14, fontweight='bold')
    plt.xticks(x, classes)
    plt.ylim([0, 1.1])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Main visualization function."""

    # Paths
    results_dir = Path('outputs/TEST_evaluation_2d') #outputs/TEST_evaluation_2d
    output_dir = results_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TEST RESULTS VISUALIZATION")
    print("=" * 70)
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load metrics
    with open(results_dir / 'metrics.json', 'r') as f:
        metrics = json.load(f)

    # Load predictions
    data = np.load(results_dir / 'predictions.npz')
    y_true = data['labels']
    y_pred = data['predictions']
    y_probs = data['probabilities'][:, 1]  # Positive class probability

    print("\n[INFO] Creating visualizations...")

    # 1. Confusion Matrix
    plot_test_confusion_matrix(metrics, output_dir / 'confusion_matrix.png')

    # 2. ROC Curve
    plot_test_roc_curve(y_true, y_probs, output_dir / 'roc_curve.png')

    # 3. PR Curve
    plot_test_pr_curve(y_true, y_probs, output_dir / 'pr_curve.png')

    # 4. Metrics Bars
    plot_test_metrics_bars(metrics, output_dir / 'metrics_bars.png')

    # 5. Per-Class Metrics
    plot_test_per_class(metrics, output_dir / 'per_class_metrics.png')

    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated {output_dir}:")
    print(f"  1. confusion_matrix.png")
    print(f"  2. roc_curve.png")
    print(f"  3. pr_curve.png")
    print(f"  4. metrics_bars.png")
    print(f"  5. per_class_metrics.png")
    print("=" * 70)


if __name__ == '__main__':
    main()