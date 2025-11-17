"""
Logging utilities for training and evaluation.
Provides formatted console and file logging with different verbosity levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(name: str = "DroneDetection",
                 log_file: Optional[str] = None,
                 level: int = logging.INFO,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Optional custom format string

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger("Training", "logs/train.log")
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "DroneDetection") -> logging.Logger:
    """
    Get existing logger or create a basic one if it doesn't exist.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up a basic console handler
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(handler)

    return logger


class TrainingLogger:
    """
    Logger for tracking training progress with metrics.
    """

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        # Setup logger
        self.logger = setup_logger(
            name=f"Training_{experiment_name}",
            log_file=str(log_file),
            level=logging.INFO
        )

        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: dict, val_metrics: dict) -> None:
        """
        Log epoch results.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
        """
        # Store metrics
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metrics)
        self.val_metrics.append(val_metrics)

        # Log to console/file
        self.logger.info(f"Epoch {epoch:03d} | "
                         f"Train Loss: {train_loss:.4f} | "
                         f"Val Loss: {val_loss:.4f}")

        # Log metrics
        if train_metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            self.logger.info(f"  Train Metrics: {metrics_str}")

        if val_metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(f"  Val Metrics:   {metrics_str}")

    def log_message(self, message: str, level: str = "info") -> None:
        """
        Log a general message.

        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)

    def save_metrics(self) -> None:
        """Save metrics to CSV file."""
        import pandas as pd

        metrics_file = self.log_dir / "training_metrics.csv"

        data = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }

        # Add train metrics
        if self.train_metrics and len(self.train_metrics) > 0:
            for key in self.train_metrics[0].keys():
                data[f'train_{key}'] = [m[key] for m in self.train_metrics]

        # Add val metrics
        if self.val_metrics and len(self.val_metrics) > 0:
            for key in self.val_metrics[0].keys():
                data[f'val_{key}'] = [m[key] for m in self.val_metrics]

        df = pd.DataFrame(data)
        df.to_csv(metrics_file, index=False)

        self.logger.info(f"Metrics saved to: {metrics_file}")


# Convenience function for quick logging
def log_info(message: str, logger_name: str = "DroneDetection") -> None:
    """Quick info logging."""
    logger = get_logger(logger_name)
    logger.info(message)


def log_warning(message: str, logger_name: str = "DroneDetection") -> None:
    """Quick warning logging."""
    logger = get_logger(logger_name)
    logger.warning(message)


def log_error(message: str, logger_name: str = "DroneDetection") -> None:
    """Quick error logging."""
    logger = get_logger(logger_name)
    logger.error(message)


__all__ = [
    'setup_logger',
    'get_logger',
    'TrainingLogger',
    'log_info',
    'log_warning',
    'log_error'
]