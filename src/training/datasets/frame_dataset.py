# src/dataset/frame_dataset.py
"""
Dataset class for loading video frames with drone annotations for ResNet-18 training.
Extracts frames from videos and applies bounding box annotations.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import List, Dict, Tuple
import random


class DroneFrameDataset(Dataset):
    """
    Dataset for loading individual frames from drone videos with annotations.
    Each sample is a single frame with binary drone label (0/1).
    """

    def __init__(self,
                 video_list: List[str],
                 video_dir: str,
                 annotations_dir: str,
                 input_size: int = 112,
                 frames_per_video: int = 100,
                 balance_classes: bool = True,
                 transform=None):
        """
        Args:
            video_list: List of video filenames
            video_dir: Directory containing videos
            annotations_dir: Directory containing annotation files
            input_size: Target image size (112x112 for fair comparison with R3D-18)
            frames_per_video: Max frames to extract per video
            balance_classes: Whether to balance drone/no-drone frames
            transform: Optional data augmentation transforms
        """
        self.video_dir = video_dir
        self.annotations_dir = annotations_dir
        self.input_size = input_size
        self.frames_per_video = frames_per_video
        self.balance_classes = balance_classes

        # Default transforms for ResNet
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ])
        else:
            self.transform = transform

        # Load all frame samples
        self.samples = self._load_frame_samples(video_list)

        print(f"Loaded {len(self.samples)} frame samples")
        self._print_class_distribution()

    def _load_frame_samples(self, video_list: List[str]) -> List[Tuple[str, int, int]]:
        """
        Load all frame samples from videos with annotations.
        Returns list of (video_path, frame_number, label) tuples.
        """
        all_samples = []

        for video_name in video_list:
            video_path = os.path.join(self.video_dir, video_name)

            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video not found: {video_path}")
                continue

            # Load annotations for this video
            annotations = self._load_video_annotations(video_name)

            # Extract frames from video
            video_samples = self._extract_video_frames(video_path, annotations)
            all_samples.extend(video_samples)

            print(f"  {video_name}: {len(video_samples)} frames")

        # Balance classes if requested
        if self.balance_classes:
            all_samples = self._balance_classes(all_samples)

        return all_samples

    def _load_video_annotations(self, video_name: str) -> Dict[int, int]:
        """
        Load frame-level annotations for a video.
        Returns dict mapping frame_number -> label (0 or 1).
        """
        # Get annotation file path
        video_base = os.path.splitext(video_name)[0]
        annotation_path = os.path.join(self.annotations_dir, f"{video_base}.txt")

        annotations = {}

        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file not found: {annotation_path}")
            return annotations

        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        frame_num = int(parts[0])
                        drone_count = int(parts[1])

                        # Binary label: 1 if drone present, 0 otherwise
                        label = 1 if drone_count > 0 else 0
                        annotations[frame_num] = label

        except Exception as e:
            print(f"Error reading annotation file {annotation_path}: {e}")

        return annotations

    def _extract_video_frames(self, video_path: str, annotations: Dict[int, int]) -> List[Tuple[str, int, int]]:
        """
        Extract frames from video and create samples.
        Returns list of (video_path, frame_number, label) tuples.
        """
        samples = []

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return samples

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine which frames to extract
        if annotations:
            # Use annotated frames
            annotated_frames = list(annotations.keys())

            # Add some random frames without drones for balance
            non_annotated_frames = [i for i in range(0, total_frames, 10)
                                    if i not in annotated_frames]
            random.shuffle(non_annotated_frames)

            # Combine annotated frames with some random frames
            selected_frames = annotated_frames + non_annotated_frames[:len(annotated_frames)]
            selected_frames = selected_frames[:self.frames_per_video]
        else:
            # No annotations - sample random frames (assume no drones)
            step = max(1, total_frames // self.frames_per_video)
            selected_frames = list(range(0, total_frames, step))[:self.frames_per_video]

        # Create samples
        for frame_num in selected_frames:
            label = annotations.get(frame_num, 0)  # Default to no drone
            samples.append((video_path, frame_num, label))

        cap.release()
        return samples

    def _balance_classes(self, samples: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """Balance drone and no-drone samples."""
        drone_samples = [s for s in samples if s[2] == 1]
        no_drone_samples = [s for s in samples if s[2] == 0]

        print(f"  Before balancing: {len(drone_samples)} drone, {len(no_drone_samples)} no-drone")

        # Balance to smaller class
        min_count = min(len(drone_samples), len(no_drone_samples))

        if min_count > 0:
            random.shuffle(drone_samples)
            random.shuffle(no_drone_samples)

            balanced_samples = drone_samples[:min_count] + no_drone_samples[:min_count]
            random.shuffle(balanced_samples)

            print(f"  After balancing: {min_count} drone, {min_count} no-drone")
            return balanced_samples

        return samples

    def _print_class_distribution(self):
        """Print class distribution statistics."""
        labels = [sample[2] for sample in self.samples]
        drone_count = sum(labels)
        no_drone_count = len(labels) - drone_count

        print(f"Class distribution: {drone_count} drone frames, {no_drone_count} no-drone frames")
        if len(labels) > 0:
            print(f"Drone percentage: {100 * drone_count / len(labels):.1f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single frame sample.
        Returns: (frame_tensor, label)
        """
        video_path, frame_number, label = self.samples[idx]

        # Extract frame from video
        frame = self._extract_frame(video_path, frame_number)

        if frame is None:
            # Return black frame if extraction fails
            frame = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            frame = self.transform(frame)

        return frame, label

    def _extract_frame(self, video_path: str, frame_number: int):
        """Extract specific frame from video."""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        return None


def create_data_loaders(train_videos: List[str],
                        val_videos: List[str],
                        video_dir: str,
                        annotations_dir: str,
                        batch_size: int = 32,
                        input_size: int = 112,
                        num_workers: int = 4):
    """
    Create train and validation data loaders for ResNet-18 training.
    """

    # Training dataset with augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation dataset without augmentation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = DroneFrameDataset(
        train_videos, video_dir, annotations_dir,
        input_size=input_size, transform=train_transform
    )

    val_dataset = DroneFrameDataset(
        val_videos, video_dir, annotations_dir,
        input_size=input_size, transform=val_transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader