"""
Threshold analyzer for finding optimal classification thresholds.
Works with both 2D and 3D models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class ThresholdAnalyzer:
    """Analyze and optimize classification thresholds."""

    def __init__(self, output_dir: Path):
        """
        Initialize threshold analyzer.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style for plots
        sns.set_style("whitegrid")

    def analyze_thresholds(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            thresholds: List[float] = None,
            save_plots: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate model performance at different thresholds.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities (for positive class)
            thresholds: List of thresholds to test
            save_plots: Whether to save visualization plots

        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05).tolist()

        print(f"[INFO] Analyzing {len(thresholds)} thresholds...")

        results = []

        for threshold in thresholds:
            # Apply threshold
            y_pred = (y_prob >= threshold).astype(int)

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Compute true/false positives/negatives
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })

        df = pd.DataFrame(results)

        # Find optimal thresholds for different metrics
        optimal_f1_idx = df['f1_score'].idxmax()
        optimal_acc_idx = df['accuracy'].idxmax()

        optimal_f1 = df.iloc[optimal_f1_idx]
        optimal_acc = df.iloc[optimal_acc_idx]

        # Save results
        csv_file = self.output_dir / 'threshold_analysis.csv'
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Threshold analysis saved to: {csv_file}")

        # Print optimal thresholds
        print(f"\n[OPTIMAL THRESHOLDS]")
        print(f"  Best F1-Score: threshold={optimal_f1['threshold']:.3f}, F1={optimal_f1['f1_score']:.4f}")
        print(f"  Best Accuracy: threshold={optimal_acc['threshold']:.3f}, Acc={optimal_acc['accuracy']:.4f}")

        # Create plots
        if save_plots:
            self._plot_threshold_analysis(df, y_true, y_prob)

        return df

    def _plot_threshold_analysis(
            self,
            df: pd.DataFrame,
            y_true: np.ndarray,
            y_prob: np.ndarray
    ):
        """Create comprehensive threshold analysis plots."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Metrics vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(df['threshold'], df['precision'], 'o-', label='Precision', linewidth=2)
        ax1.plot(df['threshold'], df['recall'], 's-', label='Recall', linewidth=2)
        ax1.plot(df['threshold'], df['f1_score'], '^-', label='F1-Score', linewidth=2)
        ax1.plot(df['threshold'], df['accuracy'], 'd-', label='Accuracy', linewidth=2)
        ax1.set_xlabel('Threshold', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Mark optimal F1 threshold
        optimal_f1_idx = df['f1_score'].idxmax()
        optimal_threshold = df.iloc[optimal_f1_idx]['threshold']
        ax1.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, label='Optimal F1')

        # Plot 2: Precision-Recall Curve
        ax2 = axes[0, 1]
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(recall_curve, precision_curve, linewidth=2)
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: ROC Curve
        ax3 = axes[1, 0]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax3.set_xlabel('False Positive Rate', fontsize=12)
        ax3.set_ylabel('True Positive Rate', fontsize=12)
        ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Plot 4: TP/FP/TN/FN vs Threshold
        ax4 = axes[1, 1]
        ax4.plot(df['threshold'], df['true_positives'], 'o-', label='True Positives', linewidth=2)
        ax4.plot(df['threshold'], df['false_positives'], 's-', label='False Positives', linewidth=2)
        ax4.plot(df['threshold'], df['false_negatives'], '^-', label='False Negatives', linewidth=2)
        ax4.set_xlabel('Threshold', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.set_title('Prediction Counts vs Threshold', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / 'threshold_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[INFO] Threshold plots saved to: {plot_file}")
        plt.close()

    def find_optimal_threshold(
            self,
            y_true: np.ndarray,
            y_prob: np.ndarray,
            metric: str = 'f1_score'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a specific metric.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'accuracy', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        df = self.analyze_thresholds(y_true, y_prob, save_plots=False)

        optimal_idx = df[metric].idxmax()
        optimal_row = df.iloc[optimal_idx]

        return optimal_row['threshold'], optimal_row[metric]

    def compare_models_thresholds(
            self,
            results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
            save_plots: bool = True
    ) -> pd.DataFrame:
        """
        Compare threshold performance across multiple models.

        Args:
            results_dict: Dict mapping model names to (y_true, y_prob) tuples
            save_plots: Whether to save comparison plots

        Returns:
            DataFrame with optimal thresholds for each model
        """
        comparison_results = []

        for model_name, (y_true, y_prob) in results_dict.items():
            print(f"\n[INFO] Analyzing {model_name}...")
            df = self.analyze_thresholds(y_true, y_prob, save_plots=False)

            optimal_idx = df['f1_score'].idxmax()
            optimal = df.iloc[optimal_idx]

            comparison_results.append({
                'model': model_name,
                'optimal_threshold': optimal['threshold'],
                'f1_score': optimal['f1_score'],
                'accuracy': optimal['accuracy'],
                'precision': optimal['precision'],
                'recall': optimal['recall']
            })

        comparison_df = pd.DataFrame(comparison_results)

        # Save comparison
        csv_file = self.output_dir / 'model_threshold_comparison.csv'
        comparison_df.to_csv(csv_file, index=False)
        print(f"\n[INFO] Model comparison saved to: {csv_file}")

        # Create comparison plot
        if save_plots:
            self._plot_model_comparison(comparison_df)

        return comparison_df

    def _plot_model_comparison(self, df: pd.DataFrame):
        """Create model comparison plot."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Optimal Thresholds
        ax1 = axes[0]
        ax1.bar(df['model'], df['optimal_threshold'])
        ax1.set_ylabel('Optimal Threshold', fontsize=12)
        ax1.set_title('Optimal Thresholds by Model', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Performance Metrics
        ax2 = axes[1]
        x = np.arange(len(df))
        width = 0.2

        ax2.bar(x - width * 1.5, df['accuracy'], width, label='Accuracy')
        ax2.bar(x - width * 0.5, df['precision'], width, label='Precision')
        ax2.bar(x + width * 0.5, df['recall'], width, label='Recall')
        ax2.bar(x + width * 1.5, df['f1_score'], width, label='F1-Score')

        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Performance Metrics at Optimal Threshold', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['model'], rotation=45)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        plot_file = self.output_dir / 'model_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"[INFO] Comparison plot saved to: {plot_file}")
        plt.close()


# Factory function
def create_threshold_analyzer(output_dir: Path) -> ThresholdAnalyzer:
    """
    Create threshold analyzer.

    Args:
        output_dir: Output directory

    Returns:
        ThresholdAnalyzer instance
    """
    return ThresholdAnalyzer(output_dir)