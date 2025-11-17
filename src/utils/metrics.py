"""
Metrics computation utilities for binary classification evaluation.
Provides functions to compute precision, recall, F1-score, accuracy, and confusion matrix.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix as sk_confusion_matrix
)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    average: str = 'binary') -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels (numpy array)
        y_pred: Predicted labels (numpy array)
        average: Averaging strategy for multi-class ('binary', 'macro', 'weighted')

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"F1: {metrics['f1_score']:.3f}")
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics


def compute_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray) -> Dict[str, int]:
    """
    Compute confusion matrix components.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing:
            - TP: True positives
            - FP: False positives
            - TN: True negatives
            - FN: False negatives

    Example:
        >>> cm = compute_confusion_matrix(y_true, y_pred)
        >>> print(f"TP: {cm['TP']}, FP: {cm['FP']}")
    """
    cm = sk_confusion_matrix(y_true, y_pred)

    # For binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp)
        }
    else:
        # Multi-class: return full matrix
        return {'confusion_matrix': cm}


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive set of metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC/AUC)

    Returns:
        Dictionary with all available metrics

    Example:
        >>> metrics = compute_all_metrics(y_true, y_pred, y_proba)
        >>> for name, value in metrics.items():
        ...     print(f"{name}: {value:.4f}")
    """
    # Basic metrics
    metrics = compute_metrics(y_true, y_pred)

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics.update(cm)

    # Derived metrics
    if 'TP' in cm and 'FP' in cm and 'TN' in cm and 'FN' in cm:
        tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']

        # Specificity
        if (tn + fp) > 0:
            metrics['specificity'] = tn / (tn + fp)

        # False positive rate
        if (fp + tn) > 0:
            metrics['fpr'] = fp / (fp + tn)

        # False negative rate
        if (fn + tp) > 0:
            metrics['fnr'] = fn / (fn + tp)

    # AUC-ROC if probabilities provided
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) == 2:  # Binary only
                metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            print(f"[WARNING] Could not compute AUC-ROC: {e}")

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title to display

    Example:
        >>> print_metrics(metrics, "Validation Metrics")
    """
    print(f"\n{title}:")
    print("=" * 50)
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            if isinstance(value, float):
                print(f"  {name:15s}: {value:.4f}")
            else:
                print(f"  {name:15s}: {value}")
    print("=" * 50)


def compute_threshold_metrics(y_true: np.ndarray,
                              y_proba: np.ndarray,
                              thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute metrics across multiple thresholds.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        thresholds: Array of thresholds to evaluate

    Returns:
        Dictionary with metrics arrays for each threshold

    Example:
        >>> thresholds = np.arange(0.1, 1.0, 0.1)
        >>> results = compute_threshold_metrics(y_true, y_proba, thresholds)
        >>> best_idx = np.argmax(results['f1_scores'])
        >>> print(f"Best threshold: {thresholds[best_idx]:.2f}")
    """
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_metrics(y_true, y_pred)

        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        accuracies.append(metrics['accuracy'])

    return {
        'thresholds': thresholds,
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'f1_scores': np.array(f1_scores),
        'accuracies': np.array(accuracies),
    }


__all__ = [
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_all_metrics',
    'print_metrics',
    'compute_threshold_metrics',
]