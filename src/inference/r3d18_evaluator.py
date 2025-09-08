"""
R3D-18 frame-level evaluator that orchestrates the complete evaluation pipeline.
Combines model inference, tubelet-to-frame mapping, and comprehensive analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.models.video as video_models
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.data_processing.frame_metrics import FrameLevelMetrics
from utils.data_processing.ground_truth_loader import GroundTruthLoader
from utils.data_processing.tubelet_mapper import TubeletFrameMapper
from utils.visualization.evaluation_plots import EvaluationPlotter


class TubeletDataset(Dataset):
    """Simple dataset for loading individual tubelets."""

    def __init__(self, csv_path: str):
        print(f"   Debug: Reading CSV from {csv_path}")

        # Read raw lines first to see what we're dealing with
        with open(csv_path, 'r') as f:
            first_lines = [f.readline().strip() for _ in range(3)]

        print(f"   Debug: First 3 raw lines:")
        for i, line in enumerate(first_lines):
            print(f"     Line {i}: '{line}'")

        # Try different approaches based on what we see
        if first_lines[0].endswith('.npy'):
            # No header row
            print("   Debug: No header detected, reading with custom names")
            self.df = pd.read_csv(csv_path, header=None, names=['tubelet_path', 'gt_label', 'video_name', 'frame_idx', 'max_iou'])
        else:
            # Has header row
            print("   Debug: Header detected, skipping first row")
            self.df = pd.read_csv(csv_path, header=0)
            # Rename columns to what we expect
            if len(self.df.columns) >= 5:
                self.df.columns = ['tubelet_path', 'gt_label', 'video_name', 'frame_idx', 'max_iou']

        print(f"   Debug: DataFrame shape: {self.df.shape}")
        print(f"   Debug: Column names: {list(self.df.columns)}")
        print(f"   Debug: First row data: {self.df.iloc[0].to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tubelet_path = row['tubelet_path']
        print(f"   Debug: Raw tubelet_path from CSV: '{tubelet_path}' (type: {type(tubelet_path)})")

        if not os.path.isabs(tubelet_path):
            project_root = Path(__file__).parent.parent.parent
            tubelet_path = project_root / tubelet_path

        print(f"   Debug: Final tubelet_path: {tubelet_path}")

        tubelet = np.load(tubelet_path)
        tubelet = torch.from_numpy(tubelet).float()
        tubelet = tubelet.permute(3, 0, 1, 2) / 255.0

        return tubelet, idx


class R3D18FrameEvaluator:
    """Complete R3D-18 frame-level evaluation pipeline."""

    def __init__(self,
                 model_path: str,
                 test_csv_path: str,
                 annotations_dir: str,
                 device: str = 'auto',
                 T: int = 3):
        """Initialize evaluator with model and data paths."""

        self.model_path = model_path
        self.test_csv_path = test_csv_path
        self.annotations_dir = annotations_dir
        self.T = T

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🔧 R3D-18 Evaluator initialized")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        print(f"   Test CSV: {test_csv_path}")
        print(f"   Annotations: {annotations_dir}")
        print(f"   Tubelet T: {T}")

        # Initialize components
        self.model = None
        self.dataset = None
        self.tubelet_results = None
        self.frame_gt = None

    def load_model(self, num_classes: int = 2) -> None:
        """Load trained R3D-18 model."""

        print("📁 Loading R3D-18 model...")

        model = video_models.r3d_18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle different model architectures
        if any(key.startswith('backbone.') for key in state_dict.keys()):
            print("   Detected model with 'backbone.' prefix, adjusting keys...")
            # Remove 'backbone.' prefix from keys
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key.replace('backbone.', '')
                    # Handle fc layer naming difference
                    if new_key.startswith('fc.1.'):
                        new_key = new_key.replace('fc.1.', 'fc.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self.model = model
        print("✅ Model loaded successfully")

    def load_data(self) -> None:
        """Load test dataset and ground truth annotations."""

        print("📊 Loading test dataset...")
        self.dataset = TubeletDataset(self.test_csv_path)

        print(f"   Total tubelets: {len(self.dataset)}")
        print(f"   Unique videos: {self.dataset.df['video_name'].nunique()}")

        # Load test video list from splits file
        print(f"📋 Loading test video list from: {self.test_videos_path}")
        try:
            with open(self.test_videos_path, 'r') as f:
                test_videos = [line.strip() for line in f.readlines() if line.strip()]
            print(f"   Test videos from splits file: {test_videos}")
        except FileNotFoundError:
            print(f"   ⚠️ Test videos file not found: {self.test_videos_path}")
            print("   Falling back to unique videos from CSV...")
            test_videos = [v for v in self.dataset.df['video_name'].unique()
                          if v != 'video_id' and v.strip() and not v.startswith('video_name')]

        print("📋 Loading ground truth annotations...")
        self.frame_gt = GroundTruthLoader.load_frame_level_annotations(
            test_videos, self.annotations_dir
        )

        GroundTruthLoader.print_annotation_summary(self.frame_gt)
        GroundTruthLoader.validate_ground_truth(self.frame_gt, test_videos)

    def run_inference(self) -> pd.DataFrame:
        """Run inference on all tubelets and return results."""

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        print("🚀 Running inference on all tubelets...")

        all_results = []

        with torch.no_grad():
            for i in range(len(self.dataset)):
                tubelet, idx = self.dataset[i]
                row = self.dataset.df.iloc[idx]

                tubelet = tubelet.unsqueeze(0).to(self.device)
                outputs = self.model(tubelet)

                probabilities = torch.softmax(outputs, dim=1)
                drone_confidence = float(probabilities[0, 1])
                prediction = int(torch.argmax(outputs, dim=1)[0])

                all_results.append({
                    'tubelet_path': row['tubelet_path'],
                    'video_name': row['video_name'],
                    'center_frame': int(row['frame_idx']),
                    'gt_label': row['gt_label'],
                    'prediction': prediction,
                    'drone_confidence': drone_confidence,
                    'max_iou': row['max_iou']
                })

                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(self.dataset)} tubelets")

        self.tubelet_results = pd.DataFrame(all_results)

        print(f"✅ Inference complete!")
        print(f"   Total tubelets: {len(self.tubelet_results)}")
        print(f"   Drone predictions: {sum(self.tubelet_results['prediction'])}")

        return self.tubelet_results

    def evaluate_single_threshold(self,
                                confidence_threshold: float = 0.5,
                                save_frame_results: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Evaluate performance at single confidence threshold."""

        if self.tubelet_results is None:
            raise ValueError("Inference not run. Call run_inference() first.")
        if self.frame_gt is None:
            raise ValueError("Ground truth not loaded. Call load_data() first.")

        print(f"🔍 Evaluating at confidence threshold: {confidence_threshold:.1f}")

        # Map tubelets to frame-level predictions
        frame_results = TubeletFrameMapper.evaluate_frame_level_with_threshold(
            self.tubelet_results, self.frame_gt, confidence_threshold, T=self.T
        )

        # Calculate metrics
        metrics = FrameLevelMetrics.calculate_all_metrics(
            frame_results['ground_truth'], frame_results['predicted_label']
        )
        metrics['threshold'] = confidence_threshold

        if save_frame_results:
            output_path = f"frame_results_threshold_{confidence_threshold:.1f}.csv"
            frame_results.to_csv(output_path, index=False)
            print(f"💾 Frame results saved: {output_path}")

        return frame_results, metrics

    def evaluate_multiple_thresholds(self,
                                   thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) -> pd.DataFrame:
        """Evaluate performance across multiple confidence thresholds."""

        print("🔍 Evaluating multiple confidence thresholds...")

        results = []

        for threshold in thresholds:
            print(f"   Testing threshold: {threshold:.1f}")

            _, metrics = self.evaluate_single_threshold(threshold)
            results.append(metrics)

        results_df = pd.DataFrame(results)

        # Find optimal threshold
        best_idx = results_df['f1_score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        best_f1 = results_df.loc[best_idx, 'f1_score']

        print(f"\n🎯 Best performance: Threshold {best_threshold:.1f} (F1: {best_f1:.4f})")

        return results_df

    def run_complete_evaluation(self,
                              thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                              create_plots: bool = True,
                              save_results: bool = True,
                              output_dir: str = "evaluation_results") -> Dict[str, any]:
        """Run complete evaluation pipeline."""

        print("🚀 Starting complete R3D-18 evaluation pipeline...")

        # Step 1: Load model and data
        if self.model is None:
            self.load_model()
        if self.dataset is None:
            self.load_data()

        # Step 2: Run inference
        if self.tubelet_results is None:
            self.run_inference()

        # Step 3: Analyze tubelet coverage
        print("\n" + "="*60)
        TubeletFrameMapper.print_tubelet_analysis(self.tubelet_results, self.frame_gt, self.T)

        # Step 4: Evaluate multiple thresholds
        results_df = self.evaluate_multiple_thresholds(thresholds)

        # Step 5: Print results table
        self._print_results_table(results_df)

        # Step 6: Find best threshold and get detailed results
        best_idx = results_df['f1_score'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']

        frame_results, best_metrics = self.evaluate_single_threshold(
            best_threshold, save_frame_results=True
        )

        # Step 7: Save results
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            results_df.to_csv(output_path / "threshold_evaluation_results.csv", index=False)
            self.tubelet_results.to_csv(output_path / "tubelet_inference_results.csv", index=False)
            frame_results.to_csv(output_path / "best_threshold_frame_results.csv", index=False)

            # Save detailed mapping
            TubeletFrameMapper.save_frame_mapping_details(
                self.tubelet_results, frame_results,
                str(output_path / "frame_mapping_details.csv"), self.T
            )

            print(f"\n💾 All results saved to: {output_dir}")

        # Step 8: Create visualizations
        if create_plots:
            EvaluationPlotter.create_evaluation_report(
                results_df, "R3D-18", output_dir
            )

        # Return summary
        return {
            'threshold_results': results_df,
            'tubelet_results': self.tubelet_results,
            'best_frame_results': frame_results,
            'best_metrics': best_metrics,
            'best_threshold': best_threshold
        }

    def _print_results_table(self, results_df: pd.DataFrame) -> None:
        """Print formatted results table."""

        print(f"\n{'='*80}")
        print("R3D-18 FRAME-LEVEL EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
        print("-" * 80)

        for _, row in results_df.iterrows():
            print(f"{row['threshold']:<10.1f} "
                  f"{row['accuracy']:<10.4f} "
                  f"{row['precision']:<10.4f} "
                  f"{row['recall']:<10.4f} "
                  f"{row['f1_score']:<10.4f} "
                  f"{row['true_positives']:<6.0f} "
                  f"{row['false_positives']:<6.0f} "
                  f"{row['true_negatives']:<6.0f} "
                  f"{row['false_negatives']:<6.0f}")

        # Best performance summary
        best_idx = results_df['f1_score'].idxmax()
        best_row = results_df.iloc[best_idx]

        print(f"\n🎯 BEST PERFORMANCE:")
        print(f"   Threshold: {best_row['threshold']:.1f}")
        print(f"   F1-Score: {best_row['f1_score']:.4f}")
        print(f"   Precision: {best_row['precision']:.4f}")
        print(f"   Recall: {best_row['recall']:.4f}")
        print(f"   Accuracy: {best_row['accuracy']:.4f}")
        print(f"{'='*80}")

    def compare_with_ground_truth_tubelets(self) -> Dict[str, any]:
        """Compare model predictions with ground truth tubelet labels."""

        if self.tubelet_results is None:
            raise ValueError("Inference not run. Call run_inference() first.")

        # Tubelet-level evaluation (traditional approach)
        tubelet_metrics = FrameLevelMetrics.calculate_all_metrics(
            self.tubelet_results['gt_label'], self.tubelet_results['prediction']
        )

        print(f"\n📊 TUBELET-LEVEL vs FRAME-LEVEL COMPARISON:")
        print(f"{'Metric':<15} {'Tubelet-Level':<15} {'Frame-Level':<15}")
        print("-" * 45)

        # Get best frame-level metrics for comparison
        if hasattr(self, '_last_frame_metrics'):
            frame_metrics = self._last_frame_metrics

            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                print(f"{metric.capitalize():<15} {tubelet_metrics[metric]:<15.4f} {frame_metrics[metric]:<15.4f}")

        return {
            'tubelet_metrics': tubelet_metrics,
            'comparison_available': hasattr(self, '_last_frame_metrics')
        }


# Main execution function
def main():
    """Main function for running R3D-18 evaluation."""

    # Configuration
    MODEL_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\models\r3d18_static_t3_20250828_011134\best.pth"
    TEST_CSV_PATH = "test_index.csv"
    ANNOTATIONS_DIR = "data/annotations"
    THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Create evaluator
    evaluator = R3D18FrameEvaluator(
        model_path=MODEL_PATH,
        test_csv_path=TEST_CSV_PATH,
        annotations_dir=ANNOTATIONS_DIR
    )

    # Run complete evaluation
    results = evaluator.run_complete_evaluation(
        thresholds=THRESHOLDS,
        create_plots=True,
        save_results=True,
        output_dir="r3d18_evaluation_results"
    )

    # Additional analysis
    evaluator.compare_with_ground_truth_tubelets()

    print("\n✅ Complete R3D-18 evaluation finished!")

    return results


if __name__ == "__main__":
    results = main()