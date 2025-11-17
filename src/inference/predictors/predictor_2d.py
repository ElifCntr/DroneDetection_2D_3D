"""
Unified predictor for 2D frame-based models (ResNet18, etc.)
Handles inference on single images, batches, or video frames.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Tuple
import cv2
from PIL import Image
import torchvision.transforms as transforms


class Predictor2D:
    """Unified predictor for 2D frame-based drone detection models."""

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device,
            input_size: int = 112,
            threshold: float = 0.5
    ):
        """
        Initialize predictor.

        Args:
            model: Trained PyTorch model
            device: Device to run inference on
            input_size: Input image size (assumes square images)
            threshold: Confidence threshold for classification
        """
        self.model = model
        self.device = device
        self.input_size = input_size
        self.threshold = threshold

        self.model.to(device)
        self.model.eval()

        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_image(
            self,
            image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict:
        """
        Predict on a single image.

        Args:
            image: Image path, numpy array, or PIL Image

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        # Transform and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
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
            images: List[Union[str, Path, np.ndarray, Image.Image]],
            batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict on a batch of images.

        Args:
            images: List of images (paths, arrays, or PIL Images)
            batch_size: Batch size for processing

        Returns:
            List of prediction dictionaries
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Preprocess batch
            tensors = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)

                tensors.append(self.transform(img))

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

    def predict_video(
            self,
            video_path: Union[str, Path],
            sample_rate: int = 1,
            save_output: bool = False,
            output_path: str = None
    ) -> Dict:
        """
        Predict on video frames.

        Args:
            video_path: Path to video file
            sample_rate: Process every Nth frame (1 = all frames)
            save_output: Whether to save annotated video
            output_path: Path to save output video

        Returns:
            Dictionary with frame-level predictions and summary
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if saving
        writer = None
        if save_output:
            if output_path is None:
                output_path = str(Path(video_path).stem + '_annotated.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_results = []
        frame_idx = 0
        drone_frames = 0

        print(f"[INFO] Processing video: {Path(video_path).name}")
        print(f"[INFO] Total frames: {total_frames}, Sampling every {sample_rate} frame(s)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every Nth frame
            if frame_idx % sample_rate == 0:
                result = self.predict_image(frame)
                result['frame_idx'] = frame_idx
                frame_results.append(result)

                if result['is_drone']:
                    drone_frames += 1

                # Annotate frame if saving
                if save_output:
                    annotated = self._annotate_frame(frame, result)
                    writer.write(annotated)
            elif save_output:
                writer.write(frame)

            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames")

        cap.release()
        if writer:
            writer.release()
            print(f"[INFO] Annotated video saved to: {output_path}")

        # Compute summary statistics
        processed_frames = len(frame_results)
        drone_percentage = (drone_frames / processed_frames * 100) if processed_frames > 0 else 0

        summary = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'sample_rate': sample_rate,
            'drone_detections': drone_frames,
            'drone_percentage': drone_percentage,
            'has_drone': drone_frames > 0
        }

        print(f"\n[RESULTS] Drone detected in {drone_frames}/{processed_frames} frames ({drone_percentage:.1f}%)")

        return {
            'summary': summary,
            'frame_predictions': frame_results
        }

    def _annotate_frame(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Annotate frame with prediction results."""
        annotated = frame.copy()

        # Determine color and text
        if result['is_drone']:
            color = (0, 0, 255)  # Red for drone
            text = f"DRONE: {result['confidence']:.2f}"
        else:
            color = (0, 255, 0)  # Green for background
            text = f"Background: {result['confidence']:.2f}"

        # Add text to frame
        cv2.putText(
            annotated, text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, color, 2
        )

        return annotated

    def set_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.threshold = threshold


# Factory function
def create_predictor_2d(
        model: torch.nn.Module,
        device: torch.device,
        input_size: int = 112,
        threshold: float = 0.5
) -> Predictor2D:
    """
    Factory function to create 2D predictor.

    Args:
        model: Trained model
        device: Device to use
        input_size: Input image size
        threshold: Confidence threshold

    Returns:
        Predictor2D instance
    """
    return Predictor2D(model, device, input_size, threshold)