"""
FRAME-LEVEL METRICS MODULE
=========================

Universal metrics calculation for binary classification tasks.
Can be used with any model (R3D-18, ResNet-18, etc.) for frame-level evaluation.

This module provides standardized metric calculations that work with:
- Ground truth labels (0/1)
- Predicted labels (0/1)
- Confidence scores (0.0-1.0)

Key Features:
- Confusion matrix calculation
- Standard classification metrics (accuracy, precision, recall, F1)
- Support for threshold-based evaluation
- Handles edge cases (zero division, empty datasets)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple


class FrameLevelMetrics:
    """
    Universal class for calculating frame-level binary classification metrics.

    Works with any model that produces binary predictions for drone detection.
    All methods are static for easy importing and reuse across different evaluators.
    """

    @staticmethod
    def calculate_confusion_matrix(y_true: Union[List, np.ndarray, pd.Series],
                                 y_pred: Union[List, np.ndarray, pd.Series]) -> Dict[str, int]:
        """
        Calculate confusion matrix components for binary classification.

        Args:
            y_true: Ground truth labels (0 = no drone, 1 = drone)
            y_pred: Predicted labels (0 = no drone, 1 = drone)

        Returns:
            Dictionary with TP, FP, TN, FN counts
        """

        # Convert to numpy arrays for consistent handling
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true({len(y_true)}) != y_pred({len(y_pred)})")

        if len(y_true) == 0:
            return {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

        # Calculate confusion matrix components
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        assert tp + fp + tn + fn == len(y_true), "Confusion matrix calculation error"

        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

    @staticmethod
    def calculate_metrics_from_confusion_matrix(confusion_matrix: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate all standard metrics from confusion matrix components.

        Args:
            confusion_matrix: Dictionary with TP, FP, TN, FN counts

        Returns:
            Dictionary with calculated metrics

        Metrics calculated:
        - accuracy: (TP + TN) / (TP + TN + FP + FN)
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)
        - f1_score: 2 * (precision * recall) / (precision + recall)
        - specificity: TN / (TN + FP)
        """

        tp = confusion_matrix['TP']
        fp = confusion_matrix['FP']
        tn = confusion_matrix['TN']
        fn = confusion_matrix['FN']

        total = tp + tn + fp + fn

        # Handle edge case of no samples
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }

        # Calculate metrics with proper zero-division handling
        accuracy = (tp + tn) / total

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # F1-score calculation with zero-division handling
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }

    @classmethod
    def calculate_all_metrics(cls, y_true: Union[List, np.ndarray, pd.Series],
                            y_pred: Union[List, np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        One-step calculation of all metrics from ground truth and predictions.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dictionary with all calculated metrics
        """

        confusion_matrix = cls.calculate_confusion_matrix(y_true, y_pred)
        return cls.calculate_metrics_from_confusion_matrix(confusion_matrix)

    @staticmethod
    def apply_confidence_threshold(y_scores: Union[List, np.ndarray, pd.Series],
                                 threshold: float) -> np.ndarray:
        """
        Convert confidence scores to binary predictions using threshold.

        Args:
            y_scores: Confidence scores (typically 0.0-1.0)
            threshold: Decision threshold

        Returns:
            Binary predictions (0/1)
        """

        y_scores = np.array(y_scores)
        return (y_scores >= threshold).astype(int)

    @classmethod
    def evaluate_multiple_thresholds(cls, y_true: Union[List, np.ndarray, pd.Series],
                                   y_scores: Union[List, np.ndarray, pd.Series],
                                   thresholds: List[float]) -> pd.DataFrame:
        """
        Evaluate performance across multiple confidence thresholds.

        Args:
            y_true: Ground truth labels
            y_scores: Confidence scores
            thresholds: List of thresholds to evaluate

        Returns:
            DataFrame with metrics for each threshold
        """

        results = []

        for threshold in thresholds:
            y_pred = cls.apply_confidence_threshold(y_scores, threshold)
            metrics = cls.calculate_all_metrics(y_true, y_pred)
            metrics['threshold'] = threshold
            results.append(metrics)

        df = pd.DataFrame(results)

        # Reorder columns for better readability
        column_order = ['threshold', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                       'true_positives', 'false_positives', 'true_negatives', 'false_negatives']

        return df[column_order]

    @staticmethod
    def find_optimal_threshold(results_df: pd.DataFrame,
                             metric: str = 'f1_score') -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric.

        Args:
            results_df: DataFrame from evaluate_multiple_thresholds()
            metric: Metric to optimize ('f1_score', 'accuracy', etc.)

        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """

        if metric not in results_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results. Available: {list(results_df.columns)}")

        best_idx = results_df[metric].idxmax()
        optimal_threshold = results_df.loc[best_idx, 'threshold']
        best_value = results_df.loc[best_idx, metric]

        return optimal_threshold, best_value

    @staticmethod
    def print_metrics_summary(metrics: Dict[str, float], model_name: str = "Model") -> None:
        """
        Print formatted metrics summary for easy reading.

        Args:
            metrics: Dictionary of calculated metrics
            model_name: Name of the model for display
        """

        print(f"\n{'='*50}")
        print(f"{model_name.upper()} PERFORMANCE SUMMARY")
        print(f"{'='*50}")
        print(f"{'Accuracy':<12}: {metrics['accuracy']:.4f}")
        print(f"{'Precision':<12}: {metrics['precision']:.4f}")
        print(f"{'Recall':<12}: {metrics['recall']:.4f}")
        print(f"{'F1-Score':<12}: {metrics['f1_score']:.4f}")
        print(f"{'Specificity':<12}: {metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"{'TP':<6}: {metrics['true_positives']:<6} {'FP':<6}: {metrics['false_positives']}")
        print(f"{'FN':<6}: {metrics['false_negatives']:<6} {'TN':<6}: {metrics['true_negatives']}")
        print(f"{'='*50}")