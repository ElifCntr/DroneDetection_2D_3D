# src/training/datasets/region_dataset.py
"""
Simple dataset for loading pre-extracted 2D regions from center frames.
"""

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RegionDataset(Dataset):
    """Dataset for loading pre-extracted 2D regions."""

    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to CSV with columns [image_path, label]
            transform: Data transforms
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        print(f"Loaded {len(self.df)} samples from {csv_path}")
        self._print_distribution()

    def _print_distribution(self):
        pos = sum(self.df['label'] == 1)
        neg = sum(self.df['label'] == 0)
        print(f"  Positive: {pos}, Negative: {neg}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row['image_path']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        if self.transform:
            image = self.transform(image)

        label = row['label']
        return image, label


def create_2d_dataloaders(train_csv, val_csv, batch_size=32, num_workers=4):
    """Create data loaders for 2D regions."""

    # Standard ResNet transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = RegionDataset(train_csv, train_transform)
    val_dataset = RegionDataset(val_csv, val_transform)

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader