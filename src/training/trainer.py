# src/training/trainer.py
"""
Main training loop for 3D CNN drone detection.
Handles train/validation epochs, metrics computation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Trainer:
    """
    Handles training and validation of 3D CNN models.
    """

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: PyTorch model (e.g., R3D-18)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dict
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Training parameters
        self.num_epochs = config['epochs']
        self.lr = config['lr']
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.save_dir = config['save_dir']
        self.save_every = config.get('save_every', 5)  # Save every N epochs

        # Add timestamp to save directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"{self.save_dir}_{timestamp}"

        # Early stopping parameters
        self.patience = config.get('patience', 5)
        self.min_delta = config.get('min_delta', 0.001)

        # Setup optimizer and loss with class weighting for imbalanced data
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Add learning rate scheduler with more aggressive decay
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )

        # Calculate class weights to handle imbalance
        train_labels = [sample[1] for sample in train_loader.dataset.samples]
        class_counts = [train_labels.count(0), train_labels.count(1)]
        total_samples = sum(class_counts)
        class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        weight_tensor = torch.FloatTensor(class_weights)

        if torch.cuda.is_available():
            weight_tensor = weight_tensor.cuda()

        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Tracking
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []

        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"Trainer initialized:")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Weight decay: {self.weight_decay}")
        print(f"  - Early stopping patience: {self.patience}")
        print(f"  - Save directory: {self.save_dir}")

    def train_epoch(self):
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy())

            # Print progress
            if batch_idx % 50 == 0:
                print(f'    Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        # Compute epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = self._compute_metrics(all_labels, all_preds)

        return avg_loss, metrics

    def validate_epoch(self):
        """Run validation epoch."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in self.val_loader:
                # Move to GPU if available
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Track metrics
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(target.cpu().numpy())

        # Compute epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = self._compute_metrics(all_labels, all_preds)

        return avg_loss, metrics

    def _compute_metrics(self, y_true, y_pred):
        """Compute classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, 'last.pth'))

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))
            print(f"    Saved best model (F1: {self.best_f1:.4f})")

        # Save periodic checkpoint every N epochs
        if (epoch + 1) % self.save_every == 0:
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch + 1:03d}.pth')
            torch.save(checkpoint, epoch_path)
            print(f"    Saved periodic checkpoint: epoch_{epoch + 1:03d}.pth")

    def print_epoch_results(self, epoch, train_loss, train_metrics, val_loss, val_metrics, elapsed):
        """Print training results for current epoch."""
        print(f"\nEpoch {epoch + 1}/{self.num_epochs} ({elapsed:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

    def fit(self):
        """Main training loop."""
        print(f"\nStarting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train and validate
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()

            # Update scheduler with validation F1
            self.scheduler.step(val_metrics['f1'])

            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1s.append(val_metrics['f1'])

            # Check if best model and early stopping
            is_best = val_metrics['f1'] > self.best_f1 + self.min_delta
            if is_best:
                self.best_f1 = val_metrics['f1']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Print results
            elapsed = time.time() - start_time
            self.print_epoch_results(epoch, train_loss, train_metrics, val_loss, val_metrics, elapsed)

            # Early stopping check - only if patience > 0
            if self.patience > 0 and self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"No improvement in validation F1 for {self.patience} epochs")
                break

        print(f"\nTraining completed!")
        print(f"Best validation F1: {self.best_f1:.4f}")
        print(f"Model checkpoints saved to: {self.save_dir}")

        return {
            'best_f1': self.best_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1s': self.val_f1s
        }