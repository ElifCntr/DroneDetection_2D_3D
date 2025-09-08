"""
Universal evaluation plotting functions for binary classification models.
Works with any model (R3D-18, ResNet-18, etc.) and supports model comparison.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class EvaluationPlotter:
    """Universal plotting class for model evaluation and comparison."""

    @staticmethod
    def plot_threshold_analysis(results_df: pd.DataFrame,
                                model_name: str = "Model",
                                save_path: Optional[str] = None) -> None:
        """Plot comprehensive threshold analysis with 6 subplots."""

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Metrics vs Threshold
        axes[0, 0].plot(results_df['threshold'], results_df['precision'], 'b-o', label='Precision', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['recall'], 'r-s', label='Recall', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['f1_score'], 'g-^', label='F1-Score', linewidth=2)
        axes[0, 0].plot(results_df['threshold'], results_df['accuracy'], 'm-d', label='Accuracy', linewidth=2)
        axes[0, 0].set_xlabel('Confidence Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title(f'{model_name}: Performance vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # 2. Precision-Recall Curve
        axes[0, 1].plot(results_df['recall'], results_df['precision'], 'b-o', linewidth=2)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].set_ylim([0, 1])

        # Add threshold labels
        for i, row in results_df.iterrows():
            if i % 2 == 0:
                axes[0, 1].annotate(f'{row["threshold"]:.1f}',
                                    (row['recall'], row['precision']),
                                    xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 3. F1-Score vs Threshold
        axes[0, 2].plot(results_df['threshold'], results_df['f1_score'], 'g-o', linewidth=3)
        axes[0, 2].set_xlabel('Confidence Threshold')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].set_title('F1-Score vs Threshold')
        axes[0, 2].grid(True, alpha=0.3)

        # Mark best F1 score
        best_f1_idx = results_df['f1_score'].idxmax()
        best_threshold = results_df.loc[best_f1_idx, 'threshold']
        best_f1 = results_df.loc[best_f1_idx, 'f1_score']
        axes[0, 2].plot(best_threshold, best_f1, 'ro', markersize=10)
        axes[0, 2].annotate(f'Best: {best_threshold:.1f}\n(F1={best_f1:.3f})',
                            (best_threshold, best_f1),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 4. TP vs FP
        axes[1, 0].plot(results_df['false_positives'], results_df['true_positives'], 'b-o', linewidth=2)
        axes[1, 0].set_xlabel('False Positives')
        axes[1, 0].set_ylabel('True Positives')
        axes[1, 0].set_title('True Positives vs False Positives')
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Detection Counts
        total_gt_drones = results_df['true_positives'] + results_df['false_negatives']
        total_predictions = results_df['true_positives'] + results_df['false_positives']

        x_pos = np.arange(len(results_df))
        width = 0.35

        axes[1, 1].bar(x_pos - width / 2, total_gt_drones.iloc[0], width,
                       alpha=0.7, label='GT Drone Frames', color='blue')
        axes[1, 1].bar(x_pos + width / 2, total_predictions, width,
                       alpha=0.7, label='Predicted Drone Frames', color='red')
        axes[1, 1].set_xlabel('Threshold Index')
        axes[1, 1].set_ylabel('Number of Frames')
        axes[1, 1].set_title('Detection Counts by Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Confusion Matrix for Best Threshold
        best_row = results_df.loc[best_f1_idx]
        cm_data = [[best_row['true_negatives'], best_row['false_positives']],
                   [best_row['false_negatives'], best_row['true_positives']]]

        sns.heatmap(cm_data, annot=True, fmt='.0f', cmap='Blues', ax=axes[1, 2],
                    xticklabels=['No Drone', 'Drone'], yticklabels=['No Drone', 'Drone'])
        axes[1, 2].set_title(f'Confusion Matrix (Threshold={best_threshold:.1f})')
        axes[1, 2].set_ylabel('True Label')
        axes[1, 2].set_xlabel('Predicted Label')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {save_path}")

        plt.show()

    @staticmethod
    def plot_model_comparison(model_results: Dict[str, pd.DataFrame],
                              metric: str = 'f1_score',
                              save_path: Optional[str] = None) -> None:
        """Compare multiple models across confidence thresholds."""

        plt.figure(figsize=(12, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'd', 'v', '<']

        for i, (model_name, results_df) in enumerate(model_results.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            plt.plot(results_df['threshold'], results_df[metric],
                     f'{color}-{marker}', label=model_name, linewidth=2, markersize=6)

            # Mark best score
            best_idx = results_df[metric].idxmax()
            best_threshold = results_df.loc[best_idx, 'threshold']
            best_score = results_df.loc[best_idx, metric]
            plt.plot(best_threshold, best_score, 'ko', markersize=8)
            plt.annotate(f'{best_score:.3f}', (best_threshold, best_score),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.xlabel('Confidence Threshold')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()} vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved to: {save_path}")

        plt.show()

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: Dict[str, int],
                              model_name: str = "Model",
                              threshold: Optional[float] = None,
                              save_path: Optional[str] = None) -> None:
        """Plot single confusion matrix with annotations."""

        cm_data = [[confusion_matrix['TN'], confusion_matrix['FP']],
                   [confusion_matrix['FN'], confusion_matrix['TP']]]

        plt.figure(figsize=(8, 6))

        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Drone', 'Drone'],
                    yticklabels=['No Drone', 'Drone'],
                    cbar_kws={'label': 'Count'})

        title = f'{model_name} Confusion Matrix'
        if threshold is not None:
            title += f' (Threshold={threshold:.1f})'

        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")

        plt.show()

    @staticmethod
    def plot_precision_recall_comparison(model_results: Dict[str, pd.DataFrame],
                                         save_path: Optional[str] = None) -> None:
        """Plot precision-recall curves for multiple models."""

        plt.figure(figsize=(10, 8))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        for i, (model_name, results_df) in enumerate(model_results.items()):
            color = colors[i % len(colors)]
            plt.plot(results_df['recall'], results_df['precision'],
                     f'{color}-o', label=model_name, linewidth=2, markersize=4)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 P-R comparison saved to: {save_path}")

        plt.show()

    @staticmethod
    def plot_metrics_summary_bar(model_metrics: Dict[str, Dict[str, float]],
                                 metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                                 save_path: Optional[str] = None) -> None:
        """Plot bar chart comparing metrics across models."""

        models = list(model_metrics.keys())
        x_pos = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['blue', 'red', 'green', 'orange']

        for i, metric in enumerate(metrics_to_plot):
            values = [model_metrics[model][metric] for model in models]
            ax.bar(x_pos + i * width, values, width,
                   label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)

            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(x_pos[j] + i * width, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x_pos + width * (len(metrics_to_plot) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Metrics summary saved to: {save_path}")

        plt.show()

    @staticmethod
    def create_evaluation_report(results_df: pd.DataFrame,
                                 model_name: str = "Model",
                                 output_dir: str = "evaluation_plots") -> None:
        """Create complete evaluation report with all plots."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Threshold analysis
        EvaluationPlotter.plot_threshold_analysis(
            results_df, model_name,
            str(output_path / f"{model_name.lower()}_threshold_analysis.png")
        )

        # Best threshold confusion matrix
        best_idx = results_df['f1_score'].idxmax()
        best_metrics = results_df.iloc[best_idx]
        best_threshold = best_metrics['threshold']

        cm = {
            'TP': int(best_metrics['true_positives']),
            'FP': int(best_metrics['false_positives']),
            'TN': int(best_metrics['true_negatives']),
            'FN': int(best_metrics['false_negatives'])
        }

        EvaluationPlotter.plot_confusion_matrix(
            cm, model_name, best_threshold,
            str(output_path / f"{model_name.lower()}_confusion_matrix.png")
        )

        print(f"📁 Complete evaluation report saved to: {output_dir}")