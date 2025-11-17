"""
Qualitative analyzer for finding and visualizing prediction examples.
Identifies TP, FP, FN, TN cases for error analysis.
Works with both 2D and 3D models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json


class QualitativeAnalyzer:
    """Analyze prediction examples for qualitative insights."""

    def __init__(self, output_dir: Path):
        """
        Initialize qualitative analyzer.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different prediction types
        for pred_type in ['TP', 'FP', 'FN', 'TN']:
            (self.output_dir / pred_type).mkdir(exist_ok=True)

    def analyze_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sample_paths: Optional[List[str]] = None,
        top_k: int = 20
    ) -> Dict[str, pd.DataFrame]:
        """
        Categorize and analyze predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            sample_paths: Optional paths to samples (for loading images/tubelets)
            top_k: Number of top examples to save per category

        Returns:
            Dictionary mapping prediction types to DataFrames with examples
        """
        print("[INFO] Analyzing predictions...")

        # Get confidence for predicted class
        confidence = np.array([y_prob[i, y_pred[i]] for i in range(len(y_pred))])

        # Categorize predictions
        tp_mask = (y_true == 1) & (y_pred == 1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        tn_mask = (y_true == 0) & (y_pred == 0)

        categories = {
            'TP': self._extract_examples(
                tp_mask, y_true, y_pred, confidence, sample_paths, top_k
            ),
            'FP': self._extract_examples(
                fp_mask, y_true, y_pred, confidence, sample_paths, top_k
            ),
            'FN': self._extract_examples(
                fn_mask, y_true, y_pred, confidence, sample_paths, top_k
            ),
            'TN': self._extract_examples(
                tn_mask, y_true, y_pred, confidence, sample_paths, top_k
            )
        }

        # Print statistics
        self._print_statistics(categories)

        # Save results
        self._save_results(categories)

        return categories

    def _extract_examples(
        self,
        mask: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: np.ndarray,
        sample_paths: Optional[List[str]],
        top_k: int
    ) -> pd.DataFrame:
        """Extract top examples for a prediction category."""

        indices = np.where(mask)[0]

        if len(indices) == 0:
            return pd.DataFrame()

        examples = []
        for idx in indices:
            example = {
                'index': int(idx),
                'true_label': int(y_true[idx]),
                'predicted_label': int(y_pred[idx]),
                'confidence': float(confidence[idx])
            }

            if sample_paths is not None and idx < len(sample_paths):
                example['sample_path'] = sample_paths[idx]

            examples.append(example)

        df = pd.DataFrame(examples)

        # Sort by confidence (descending for correct predictions, ascending for errors)
        if len(df) > 0:
            df = df.sort_values('confidence', ascending=False).head(top_k)

        return df

    def _print_statistics(self, categories: Dict[str, pd.DataFrame]):
        """Print statistics about prediction categories."""

        print("\n" + "=" * 60)
        print("QUALITATIVE ANALYSIS RESULTS")
        print("=" * 60)

        total = sum(len(df) for df in categories.values())

        for cat_name, df in categories.items():
            count = len(df)
            percentage = (count / total * 100) if total > 0 else 0

            category_names = {
                'TP': 'True Positives (Correct Drone Detection)',
                'FP': 'False Positives (Background → Drone)',
                'FN': 'False Negatives (Missed Drones)',
                'TN': 'True Negatives (Correct Background)'
            }

            print(f"\n{category_names[cat_name]}:")
            print(f"  Count: {count} ({percentage:.1f}%)")

            if len(df) > 0:
                print(f"  Avg Confidence: {df['confidence'].mean():.4f}")
                print(f"  Max Confidence: {df['confidence'].max():.4f}")
                print(f"  Min Confidence: {df['confidence'].min():.4f}")

        print("=" * 60 + "\n")

    def _save_results(self, categories: Dict[str, pd.DataFrame]):
        """Save analysis results to files."""

        # Save each category to CSV
        for cat_name, df in categories.items():
            if len(df) > 0:
                csv_file = self.output_dir / f'{cat_name}_examples.csv'
                df.to_csv(csv_file, index=False)
                print(f"[INFO] {cat_name} examples saved to: {csv_file}")

        # Create summary JSON
        summary = {
            'total_samples': sum(len(df) for df in categories.values()),
            'counts': {name: len(df) for name, df in categories.items()},
            'percentages': {
                name: len(df) / sum(len(d) for d in categories.values()) * 100
                for name, df in categories.items()
            }
        }

        json_file = self.output_dir / 'analysis_summary.json'
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Summary saved to: {json_file}")

    def create_confusion_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_plot: bool = True
    ):
        """Create and save confusion matrix visualization."""

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Background', 'Drone'],
            yticklabels=['Background', 'Drone'],
            ax=ax, cbar_kws={'label': 'Count'}
        )

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        # Add percentage annotations
        cm_percent = cm / cm.sum() * 100
        for i in range(2):
            for j in range(2):
                text = ax.text(
                    j + 0.5, i + 0.7,
                    f'({cm_percent[i, j]:.1f}%)',
                    ha='center', va='center',
                    fontsize=10, color='gray'
                )

        plt.tight_layout()

        if save_plot:
            plot_file = self.output_dir / 'confusion_matrix.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"[INFO] Confusion matrix saved to: {plot_file}")

        plt.close()

    def create_confidence_distribution_plot(
        self,
        categories: Dict[str, pd.DataFrame],
        save_plot: bool = True
    ):
        """Create confidence distribution plots for each category."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        category_names = {
            'TP': 'True Positives',
            'FP': 'False Positives',
            'FN': 'False Negatives',
            'TN': 'True Negatives'
        }

        colors = {
            'TP': 'green',
            'FP': 'red',
            'FN': 'orange',
            'TN': 'blue'
        }

        for idx, (cat, df) in enumerate(categories.items()):
            ax = axes[idx]

            if len(df) > 0:
                ax.hist(
                    df['confidence'], bins=20,
                    color=colors[cat], alpha=0.7, edgecolor='black'
                )
                ax.axvline(
                    df['confidence'].mean(), color='black',
                    linestyle='--', linewidth=2, label=f'Mean: {df["confidence"].mean():.3f}'
                )
                ax.set_xlabel('Confidence', fontsize=11)
                ax.set_ylabel('Count', fontsize=11)
                ax.set_title(f'{category_names[cat]} (n={len(df)})', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5, 0.5, 'No Examples',
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=14
                )
                ax.set_title(category_names[cat], fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_plot:
            plot_file = self.output_dir / 'confidence_distributions.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"[INFO] Confidence distributions saved to: {plot_file}")

        plt.close()

    def generate_error_analysis_report(
        self,
        categories: Dict[str, pd.DataFrame],
        model_name: str = "Model"
    ):
        """Generate a comprehensive error analysis report."""

        report_file = self.output_dir / 'error_analysis_report.txt'

        with open(report_file, 'w') as f:
            f.write(f"ERROR ANALYSIS REPORT - {model_name}\n")
            f.write("=" * 70 + "\n\n")

            # Overall statistics
            total = sum(len(df) for df in categories.values())
            f.write(f"Total Samples: {total}\n\n")

            # Category breakdown
            for cat_name in ['TP', 'FP', 'FN', 'TN']:
                df = categories[cat_name]
                count = len(df)
                percentage = (count / total * 100) if total > 0 else 0

                category_descriptions = {
                    'TP': 'True Positives - Correctly detected drones',
                    'FP': 'False Positives - Background incorrectly classified as drone',
                    'FN': 'False Negatives - Drones that were missed',
                    'TN': 'True Negatives - Correctly classified background'
                }

                f.write(f"{category_descriptions[cat_name]}\n")
                f.write(f"  Count: {count} ({percentage:.1f}%)\n")

                if len(df) > 0:
                    f.write(f"  Mean Confidence: {df['confidence'].mean():.4f}\n")
                    f.write(f"  Std Confidence: {df['confidence'].std():.4f}\n")

                f.write("\n")

            # Error analysis
            fp_count = len(categories['FP'])
            fn_count = len(categories['FN'])
            total_errors = fp_count + fn_count

            f.write("ERROR SUMMARY\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Errors: {total_errors}\n")
            f.write(f"  False Positives: {fp_count} ({fp_count/total_errors*100:.1f}% of errors)\n")
            f.write(f"  False Negatives: {fn_count} ({fn_count/total_errors*100:.1f}% of errors)\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 70 + "\n")

            if fp_count > fn_count:
                f.write("- High false positive rate detected\n")
                f.write("- Consider increasing classification threshold\n")
                f.write("- Review background samples for common misclassification patterns\n")
            elif fn_count > fp_count:
                f.write("- High false negative rate detected\n")
                f.write("- Consider decreasing classification threshold\n")
                f.write("- Review missed drone samples for common patterns\n")
            else:
                f.write("- Balanced error distribution\n")
                f.write("- Model performance is well-calibrated\n")

        print(f"[INFO] Error analysis report saved to: {report_file}")


def create_qualitative_analyzer(output_dir: Path) -> 'QualitativeAnalyzer':
    """
    Factory function to create qualitative analyzer.

    Args:
        output_dir: Output directory for results

    Returns:
        QualitativeAnalyzer instance

    Example:
        >>> analyzer = create_qualitative_analyzer(Path('results/analysis'))
        >>> categories = analyzer.analyze_predictions(y_true, y_pred, y_prob)
    """
    return QualitativeAnalyzer(output_dir)