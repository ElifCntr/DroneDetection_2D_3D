"""
Unified evaluator for 2D frame-based models (ResNet18, etc.)
Works with any 2D CNN architecture.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import pandas as pd
import json


class Evaluator2D:
    """Unified evaluator for 2D frame-based models."""

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            output_dir: Path,
            model_name: str = "model_2d"
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            output_dir: Directory to save results
            model_name: Name of the model (for file naming)
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.model_name = model_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(device)
        self.model.eval()

    def evaluate(
            self,
            dataloader: DataLoader,
            save_predictions: bool = True
    ) -> Dict:
        """
        Run evaluation on dataset.

        Args:
            dataloader: DataLoader for evaluation dataset
            save_predictions: Whether to save predictions to file

        Returns:
            Dictionary containing all metrics and results
        """
        print(f"[INFO] Starting evaluation on {len(dataloader.dataset)} samples...")

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle different batch formats
                if isinstance(batch, (tuple, list)):
                    data, target = batch[0], batch[1]
                else:
                    data = batch['input']
                    target = batch['label']

                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if batch_idx % 50 == 0:
                    print(f"  Processed {batch_idx}/{len(dataloader)} batches")

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)

        print("[INFO] Computing metrics...")

        # Compute all metrics
        metrics = self._compute_metrics(y_true, y_pred, y_prob)

        # Save results
        self._save_results(metrics, y_true, y_pred, y_prob, save_predictions)

        return {
            'metrics': metrics,
            'predictions': y_pred,
            'labels': y_true,
            'probabilities': y_prob
        }

    def _compute_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: np.ndarray
    ) -> Dict:
        """Compute comprehensive evaluation metrics."""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # ROC curve (for binary classification)
        if y_prob.shape[1] == 2:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, thresholds, roc_auc = None, None, None, None

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))

        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc) if roc_auc is not None else None
            },
            'per_class': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist()
            },
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_distribution,
            'total_samples': len(y_true),
            'roc_curve': {
                'fpr': fpr.tolist() if fpr is not None else None,
                'tpr': tpr.tolist() if tpr is not None else None,
                'thresholds': thresholds.tolist() if thresholds is not None else None
            }
        }

        return metrics

    def _save_results(
            self,
            metrics: Dict,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            y_prob: np.ndarray,
            save_predictions: bool
    ):
        """Save evaluation results to files."""

        # Save metrics as JSON
        metrics_file = self.output_dir / f'{self.model_name}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Metrics saved to: {metrics_file}")

        # Save classification report
        report = classification_report(
            y_true, y_pred,
            target_names=['Background', 'Drone'],
            digits=4
        )
        report_file = self.output_dir / f'{self.model_name}_classification_report.txt'
        with open(report_file, 'w') as f:
            f.write(f"Classification Report - {self.model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
        print(f"[INFO] Report saved to: {report_file}")

        # Save predictions if requested
        if save_predictions:
            pred_file = self.output_dir / f'{self.model_name}_predictions.npz'
            np.savez(
                pred_file,
                predictions=y_pred,
                labels=y_true,
                probabilities=y_prob
            )
            print(f"[INFO] Predictions saved to: {pred_file}")

        # Print summary to console
        self._print_summary(metrics)

    def _print_summary(self, metrics: Dict):
        """Print evaluation summary to console."""
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS - {self.model_name}")
        print("=" * 60)

        overall = metrics['overall']
        print(f"\nOVERALL METRICS:")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall:    {overall['recall']:.4f}")
        print(f"  F1-Score:  {overall['f1_score']:.4f}")
        if overall['roc_auc'] is not None:
            print(f"  ROC AUC:   {overall['roc_auc']:.4f}")

        per_class = metrics['per_class']
        print(f"\nPER-CLASS METRICS:")
        class_names = ['Background', 'Drone']
        for i, name in enumerate(class_names):
            print(f"  {name}:")
            print(f"    Precision: {per_class['precision'][i]:.4f}")
            print(f"    Recall:    {per_class['recall'][i]:.4f}")
            print(f"    F1-Score:  {per_class['f1_score'][i]:.4f}")

        cm = np.array(metrics['confusion_matrix'])
        print(f"\nCONFUSION MATRIX:")
        print(f"              Predicted")
        print(f"              Bg    Drone")
        print(f"  Actual Bg   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
        print(f"        Drone {cm[1, 0]:4d}  {cm[1, 1]:4d}")

        dist = metrics['class_distribution']
        print(f"\nCLASS DISTRIBUTION:")
        for label, count in dist.items():
            class_name = 'Background' if label == 0 else 'Drone'
            percentage = count / metrics['total_samples'] * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")

        print("=" * 60 + "\n")

    def evaluate_with_thresholds(
            self,
            dataloader: DataLoader,
            thresholds: List[float] = None
    ) -> pd.DataFrame:
        """
        Evaluate model performance at different confidence thresholds.

        Args:
            dataloader: DataLoader for evaluation
            thresholds: List of thresholds to test (default: 0.1 to 0.9)

        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1).tolist()

        print(f"[INFO] Evaluating {len(thresholds)} thresholds...")

        # Get predictions once
        results = self.evaluate(dataloader, save_predictions=False)
        y_true = results['labels']
        y_prob = results['probabilities']

        # Test each threshold
        threshold_results = []

        for threshold in thresholds:
            # Apply threshold to get predictions
            y_pred_thresh = (y_prob[:, 1] >= threshold).astype(int)

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred_thresh)
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)

            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

        df = pd.DataFrame(threshold_results)

        # Save to CSV
        csv_file = self.output_dir / f'{self.model_name}_threshold_analysis.csv'
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Threshold analysis saved to: {csv_file}")

        # Find optimal threshold
        optimal_idx = df['f1_score'].idxmax()
        optimal = df.iloc[optimal_idx]

        print(f"\n[INFO] Optimal Threshold: {optimal['threshold']:.2f}")
        print(f"        F1-Score: {optimal['f1_score']:.4f}")

        return df


# Factory function for easy creation
def create_evaluator_2d(
        model: torch.nn.Module,
        device: torch.device,
        output_dir: Path,
        model_name: str = "model_2d"
) -> Evaluator2D:
    """
    Factory function to create 2D evaluator.

    Args:
        model: Trained model
        device: Device to use
        output_dir: Output directory
        model_name: Model name

    Returns:
        Evaluator2D instance
    """
    return Evaluator2D(model, device, output_dir, model_name)