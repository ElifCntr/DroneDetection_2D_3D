"""
Unified predictor for 3D tubelet-based models (R3D-18, R(2+1)D, etc.)
Handles inference on tubelets (short video clips).
"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple
import cv2


class Predictor3D:
    """Unified predictor for 3D tubelet-based drone detection models."""

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            input_size: int = 112,
            temporal_length: int = 3,
            threshold: float = 0.5
    ):
        """
        Initialize predictor.

        Args:
            model: Trained PyTorch 3D model
            device: Device to run inference on
            input_size: Spatial size of input (H, W)
            temporal_length: Number of frames per tubelet (T)
            threshold: Confidence threshold for classification
        """
        self.model = model
        self.device = device
        self.input_size = input_size
        self.temporal_length = temporal_length
        self.threshold = threshold

        self.model.to(device)
        self.model.eval()

    def predict_tubelet(
            self,
            tubelet: Union[str, Path, np.ndarray]
    ) -> Dict:
        """
        Predict on a single tubelet.

        Args:
            tubelet: Tubelet path (.npy file) or numpy array (T, H, W, C)

        Returns:
            Dictionary with prediction results
        """
        # Load tubelet if path
        if isinstance(tubelet, (str, Path)):
            tubelet = np.load(tubelet)

        # Validate shape
        if tubelet.ndim != 4:
            raise ValueError(f"Expected 4D tubelet (T, H, W, C), got shape {tubelet.shape}")

        T, H, W, C = tubelet.shape

        # Resize if needed
        if H != self.input_size or W != self.input_size:
            resized_frames = []
            for t in range(T):
                frame = cv2.resize(tubelet[t], (self.input_size, self.input_size))
                resized_frames.append(frame)
            tubelet = np.stack(resized_frames, axis=0)

        # Pad or truncate temporal dimension if needed
        if T < self.temporal_length:
            # Pad by repeating last frame
            padding = np.repeat(tubelet[-1:], self.temporal_length - T, axis=0)
            tubelet = np.concatenate([tubelet, padding], axis=0)
        elif T > self.temporal_length:
            # Use middle frames
            start_idx = (T - self.temporal_length) // 2
            tubelet = tubelet[start_idx:start_idx + self.temporal_length]

        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        tubelet_tensor = torch.from_numpy(tubelet).float()
        tubelet_tensor = tubelet_tensor.permute(3, 0, 1, 2)  # C, T, H, W

        # Normalize to [0, 1] if needed
        if tubelet_tensor.max() > 1.0:
            tubelet_tensor = tubelet_tensor / 255.0

        # Add batch dimension and move to device
        tubelet_tensor = tubelet_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(tubelet_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item()

        return {
            'prediction': prediction,
            'label': 'Drone' if prediction == 1 else 'Background',
            'confidence': confidence,
            'probabilities': {
                'background': probabilities[0, 0].item(),
                'drone': probabilities[0, 1].item()
            },
            'is_drone': prediction == 1 and confidence >= self.threshold
        }

    def predict_batch(
            self,
            tubelets: List[Union[str, Path, np.ndarray]],
            batch_size: int = 16
    ) -> List[Dict]:
        """
        Predict on a batch of tubelets.

        Args:
            tubelets: List of tubelet paths or arrays
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(tubelets), batch_size):
            batch = tubelets[i:i + batch_size]

            # Preprocess batch
            tensors = []
            for tubelet in batch:
                # Load if path
                if isinstance(tubelet, (str, Path)):
                    tubelet = np.load(tubelet)

                T, H, W, C = tubelet.shape

                # Resize spatial dimensions
                if H != self.input_size or W != self.input_size:
                    resized_frames = []
                    for t in range(T):
                        frame = cv2.resize(tubelet[t], (self.input_size, self.input_size))
                        resized_frames.append(frame)
                    tubelet = np.stack(resized_frames, axis=0)

                # Handle temporal dimension
                if T < self.temporal_length:
                    padding = np.repeat(tubelet[-1:], self.temporal_length - T, axis=0)
                    tubelet = np.concatenate([tubelet, padding], axis=0)
                elif T > self.temporal_length:
                    start_idx = (T - self.temporal_length) // 2
                    tubelet = tubelet[start_idx:start_idx + self.temporal_length]

                # Convert to tensor
                tubelet_tensor = torch.from_numpy(tubelet).float()
                tubelet_tensor = tubelet_tensor.permute(3, 0, 1, 2)

                if tubelet_tensor.max() > 1.0:
                    tubelet_tensor = tubelet_tensor / 255.0

                tensors.append(tubelet_tensor)

            # Stack and move to device
            batch_tensor = torch.stack(tensors).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

            # Collect results
            for j in range(len(batch)):
                pred = predictions[j].item()
                conf = probabilities[j, pred].item()

                results.append({
                    'prediction': pred,
                    'label': 'Drone' if pred == 1 else 'Background',
                    'confidence': conf,
                    'probabilities': {
                        'background': probabilities[j, 0].item(),
                        'drone': probabilities[j, 1].item()
                    },
                    'is_drone': pred == 1 and conf >= self.threshold
                })

        return results

    def predict_video_with_tubelets(
            self,
            video_path: Union[str, Path],
            stride: int = 1
    ) -> Dict:
        """
        Generate tubelets from video and predict.

        Args:
            video_path: Path to video file
            stride: Frame stride for tubelet generation

        Returns:
            Dictionary with tubelet predictions and video-level summary
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        print(f"[INFO] Loaded {len(frames)} frames from video")

        # Generate tubelets
        tubelets = []
        tubelet_indices = []

        for i in range(0, len(frames) - self.temporal_length + 1, stride):
            tubelet_frames = frames[i:i + self.temporal_length]
            tubelet = np.stack(tubelet_frames, axis=0)  # (T, H, W, C)
            tubelets.append(tubelet)
            tubelet_indices.append(i)

        print(f"[INFO] Generated {len(tubelets)} tubelets with stride={stride}")

        # Predict on tubelets
        results = self.predict_batch(tubelets, batch_size=16)

        # Add frame indices
        for i, result in enumerate(results):
            result['start_frame'] = tubelet_indices[i]
            result['end_frame'] = tubelet_indices[i] + self.temporal_length - 1

        # Compute video-level summary
        drone_tubelets = sum(1 for r in results if r['is_drone'])
        drone_percentage = (drone_tubelets / len(results) * 100) if results else 0

        # Aggregate confidence scores
        avg_confidence = np.mean([r['confidence'] for r in results])
        max_confidence = max([r['confidence'] for r in results])

        summary = {
            'video_path': str(video_path),
            'total_frames': len(frames),
            'total_tubelets': len(tubelets),
            'drone_tubelets': drone_tubelets,
            'drone_percentage': drone_percentage,
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'has_drone': drone_tubelets > 0
        }

        print(f"\n[RESULTS] Drone detected in {drone_tubelets}/{len(results)} tubelets ({drone_percentage:.1f}%)")

        return {
            'summary': summary,
            'tubelet_predictions': results
        }

    def set_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.threshold = threshold


# Factory function
def create_predictor_3d(
        model: torch.nn.Module,
        device: torch.device,
        input_size: int = 112,
        temporal_length: int = 3,
        threshold: float = 0.5
) -> Predictor3D:
    """
    Factory function to create 3D predictor.

    Args:
        model: Trained 3D model
        device: Device to use
        input_size: Spatial input size
        temporal_length: Number of frames
        threshold: Confidence threshold

    Returns:
        Predictor3D instance
    """
    return Predictor3D(model, device, input_size, temporal_length, threshold)