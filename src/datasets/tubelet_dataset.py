# src/datasets/tubelet_dataset.py
"""
Loads 3-frame tubelet sequences (.npy files) for 3D CNN training.
Converts (T,H,W,C) format to (C,T,H,W) PyTorch format.
Handles CSV indexing and path normalization.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class TubeletDataset(Dataset):
    """
    Dataset for loading tubelet sequences for 3D CNN training.

    Expected tubelet format: (T, H, W, C) numpy arrays
    Output format: (C, T, H, W) tensors for PyTorch 3D CNN
    """

    def __init__(self, csv_path, root_dir=None, transform=None, normalize=True):
        """
        Args:
            csv_path: Path to CSV index file with columns [path, label] or [filename, label]
            root_dir: Root directory containing tubelet files (optional if CSV has full paths)
            transform: Optional transforms to apply
            normalize: Whether to normalize pixel values to [0,1]
        """
        self.root_dir = root_dir
        self.transform = transform
        self.normalize = normalize

        # Load samples from CSV
        self.samples = self._load_csv(csv_path)

        print(f"Loaded {len(self.samples)} samples from {csv_path}")
        self._print_label_distribution()

    def _load_csv(self, csv_path):
        """Load and parse the CSV index file."""
        samples = []

        with open(csv_path, 'r') as f:
            header = f.readline().strip()
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) >= 2:
                file_path = parts[0]
                label = int(parts[1])

                # Normalize path separators to forward slashes
                file_path = file_path.replace('\\', '/')

                samples.append((file_path, label))

        return samples

    def _print_label_distribution(self):
        """Print distribution of labels in dataset."""
        labels = [label for _, label in self.samples]
        unique_labels = set(labels)

        print("Label distribution:")
        for label in sorted(unique_labels):
            count = labels.count(label)
            percentage = (count / len(labels)) * 100
            print(f"  Label {label}: {count} samples ({percentage:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load a single tubelet sample.

        Returns:
            tuple: (tubelet_tensor, label)
                - tubelet_tensor: shape (C, T, H, W)
                - label: int
        """
        file_path, label = self.samples[idx]

        # Smart path handling:
        # If path starts with 'data/', it's already a full path from project root
        # Otherwise, it's relative to root_dir
        if file_path.startswith('data/'):
            # Already a full path from project root
            tubelet_path = Path(file_path)
        elif self.root_dir is not None:
            # Relative path - add root_dir
            tubelet_path = Path(self.root_dir) / file_path
        else:
            # No root_dir and not absolute - use as-is
            tubelet_path = Path(file_path)

        # Load tubelet
        try:
            tubelet = np.load(tubelet_path)  # Shape: (T, H, W, C)
        except Exception as e:
            raise RuntimeError(f"Failed to load tubelet {tubelet_path}: {e}")

        # Convert to float32 and normalize if requested
        tubelet = tubelet.astype(np.float32)
        if self.normalize:
            tubelet = tubelet / 255.0  # Normalize to [0, 1]

        # Transpose from (T, H, W, C) to (C, T, H, W) for PyTorch 3D CNN
        tubelet = tubelet.transpose(3, 0, 1, 2)

        # Convert to tensor
        tubelet_tensor = torch.from_numpy(tubelet)

        # Apply transforms if provided
        if self.transform is not None:
            tubelet_tensor = self.transform(tubelet_tensor)

        return tubelet_tensor, label


def create_dataloaders(train_csv, val_csv, root_dir=None, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders.

    Args:
        train_csv: Path to training CSV index
        val_csv: Path to validation CSV index
        root_dir: Root directory for tubelet files (optional if CSVs have full paths)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = TubeletDataset(train_csv, root_dir)
    val_dataset = TubeletDataset(val_csv, root_dir)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    train_csv = "../../data/tubelets/train_index.csv"
    root_dir = "../../data/tubelets"

    dataset = TubeletDataset(train_csv, root_dir)

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a sample
    tubelet, label = dataset[0]
    print(f"Sample shape: {tubelet.shape}")  # Should be (C, T, H, W)
    print(f"Sample dtype: {tubelet.dtype}")
    print(f"Sample range: [{tubelet.min():.3f}, {tubelet.max():.3f}]")
    print(f"Label: {label}")