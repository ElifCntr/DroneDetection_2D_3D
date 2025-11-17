"""
Model checkpoint utilities for saving and loading trained models.
Handles both full checkpoints (with optimizer state) and model-only saves.
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any


def save_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    epoch: int,
                    filepath: str,
                    scheduler: Optional[Any] = None,
                    metrics: Optional[Dict[str, float]] = None,
                    **kwargs) -> None:
    """
    Save model checkpoint with training state.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state (optional)
        epoch: Current epoch number
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        metrics: Dictionary of metrics (optional)
        **kwargs: Additional items to save

    Example:
        >>> save_checkpoint(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     filepath='checkpoints/epoch_10.pth',
        ...     metrics={'val_f1': 0.85}
        ... )
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }

    # Add optional components
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    # Add any additional kwargs
    checkpoint.update(kwargs)

    # Save
    torch.save(checkpoint, filepath)
    print(f"[INFO] Checkpoint saved to: {filepath}")


def load_checkpoint(model: torch.nn.Module,
                    filepath: str,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[Any] = None,
                    device: str = 'cpu',
                    strict: bool = True) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.

    Args:
        model: PyTorch model to load weights into
        filepath: Path to checkpoint file
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        device: Device to map checkpoint to
        strict: Whether to strictly enforce state dict keys match

    Returns:
        Dictionary containing checkpoint metadata (epoch, metrics, etc.)

    Example:
        >>> info = load_checkpoint(
        ...     model=model,
        ...     filepath='checkpoints/best.pth',
        ...     optimizer=optimizer,
        ...     device='cuda'
        ... )
        >>> print(f"Loaded from epoch {info['epoch']}")
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)

    # Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    print(f"[INFO] Model weights loaded from: {filepath}")

    # Restore optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[INFO] Optimizer state restored")

    # Restore scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"[INFO] Scheduler state restored")

    # Return metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {}),
    }

    # Include any other keys that aren't state dicts
    for key, value in checkpoint.items():
        if not key.endswith('_state_dict') and key not in metadata:
            metadata[key] = value

    return metadata


def save_model_only(model: torch.nn.Module, filepath: str) -> None:
    """
    Save only model weights (no optimizer or training state).
    Useful for final model deployment.

    Args:
        model: PyTorch model to save
        filepath: Path to save model

    Example:
        >>> save_model_only(model, 'models/best_model.pth')
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"[INFO] Model saved to: {filepath}")


def load_model_only(model: torch.nn.Module,
                    filepath: str,
                    device: str = 'cpu',
                    strict: bool = True) -> None:
    """
    Load only model weights (no optimizer or training state).

    Args:
        model: PyTorch model to load weights into
        filepath: Path to model file
        device: Device to map model to
        strict: Whether to strictly enforce state dict keys match

    Example:
        >>> load_model_only(model, 'models/best_model.pth', device='cuda')
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    print(f"[INFO] Model loaded from: {filepath}")


class ModelCheckpoint:
    """
    Callback for saving model checkpoints during training.

    Saves checkpoints based on monitoring a metric (e.g., validation loss).
    Can save best model and/or periodic checkpoints.

    Example:
        >>> checkpoint_callback = ModelCheckpoint(
        ...     dirpath='checkpoints',
        ...     monitor='val_f1',
        ...     mode='max',
        ...     save_best=True,
        ...     save_last=True
        ... )
        >>>
        >>> # In training loop:
        >>> checkpoint_callback(model, optimizer, epoch, metrics)
    """

    def __init__(self,
                 dirpath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best: bool = True,
                 save_last: bool = True,
                 save_every_n_epochs: Optional[int] = None):
        """
        Initialize checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for metric
            save_best: Whether to save best model
            save_last: Whether to save last checkpoint
            save_every_n_epochs: Save checkpoint every N epochs (optional)
        """
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs

        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def __call__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epoch: int,
                 metrics: Dict[str, float],
                 scheduler: Optional[Any] = None) -> None:
        """
        Check if checkpoint should be saved.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of current metrics
            scheduler: Scheduler state (optional)
        """
        current_value = metrics.get(self.monitor)

        # Save best model
        if self.save_best and current_value is not None:
            is_better = (
                    (self.mode == 'min' and current_value < self.best_value) or
                    (self.mode == 'max' and current_value > self.best_value)
            )

            if is_better:
                self.best_value = current_value
                filepath = self.dirpath / 'best.pth'
                save_checkpoint(model, optimizer, epoch, str(filepath),
                                scheduler=scheduler, metrics=metrics)
                print(f"[INFO] New best {self.monitor}: {current_value:.4f}")

        # Save last checkpoint
        if self.save_last:
            filepath = self.dirpath / 'last.pth'
            save_checkpoint(model, optimizer, epoch, str(filepath),
                            scheduler=scheduler, metrics=metrics)

        # Save periodic checkpoints
        if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
            filepath = self.dirpath / f'epoch_{epoch:03d}.pth'
            save_checkpoint(model, optimizer, epoch, str(filepath),
                            scheduler=scheduler, metrics=metrics)


__all__ = [
    'save_checkpoint',
    'load_checkpoint',
    'save_model_only',
    'load_model_only',
    'ModelCheckpoint',
]