"""
Tubelet-to-frame mapping utilities for R3D-18 evaluation.
Handles temporal coverage mapping where each tubelet covers multiple frames.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


class TubeletFrameMapper:
    """Maps tubelet predictions to frame-level predictions for R3D-18 evaluation."""

    @staticmethod
    def get_covered_frames(center_frame: int, T: int = 3) -> List[int]:
        """Get frames covered by a tubelet centered at center_frame."""

        start_frame = max(0, center_frame - T // 2)
        return list(range(start_frame, start_frame + T))

    @staticmethod
    def map_tubelets_to_frame_coverage(tubelet_predictions: pd.DataFrame,
                                       T: int = 3) -> Dict[Tuple[str, int], List[Dict]]:
        """Map tubelets to their frame coverage for drone predictions only."""

        frame_coverage = defaultdict(list)

        # Only process tubelets that predicted drone (prediction == 1)
        drone_predictions = tubelet_predictions[tubelet_predictions['prediction'] == 1]

        print(
            f"📊 Processing {len(drone_predictions)} drone predictions out of {len(tubelet_predictions)} total tubelets")

        for _, pred in drone_predictions.iterrows():
            covered_frames = TubeletFrameMapper.get_covered_frames(pred['center_frame'], T)

            for frame_num in covered_frames:
                frame_key = (pred['video_name'], frame_num)
                frame_coverage[frame_key].append({
                    'tubelet_center': pred['center_frame'],
                    'drone_confidence': pred['drone_confidence'],
                    'prediction': pred['prediction'],
                    'tubelet_path': pred.get('tubelet_path', '')
                })

        return dict(frame_coverage)

    @staticmethod
    def get_frame_prediction_with_confidence(covering_tubelets: List[Dict],
                                             confidence_threshold: float = 0.5) -> Tuple[int, float]:
        """Get frame-level prediction from covering drone tubelets."""

        if not covering_tubelets:
            return 0, 0.0

        # Get maximum confidence from all covering drone tubelets
        max_confidence = max(t['drone_confidence'] for t in covering_tubelets)

        # Apply confidence threshold
        frame_prediction = 1 if max_confidence >= confidence_threshold else 0

        return frame_prediction, max_confidence

    @staticmethod
    def create_complete_frame_evaluation_set(tubelet_predictions: pd.DataFrame,
                                             frame_gt: Dict[str, Dict[int, int]],
                                             all_annotated_frames: Optional[Dict[str, Set[int]]] = None,
                                             T: int = 3) -> Set[Tuple[str, int]]:
        """Create complete set of frames for evaluation."""

        evaluation_frames = set()

        # Add all frames that have ground truth annotations
        for video_name, video_gt in frame_gt.items():
            for frame_num in video_gt.keys():
                evaluation_frames.add((video_name, frame_num))

        # Add all frames covered by tubelets (both drone and non-drone predictions)
        for _, pred in tubelet_predictions.iterrows():
            covered_frames = TubeletFrameMapper.get_covered_frames(pred['center_frame'], T)
            for frame_num in covered_frames:
                evaluation_frames.add((pred['video_name'], frame_num))

        # Optionally add all annotated frames (including explicit no-drone frames)
        if all_annotated_frames:
            for video_name, frame_set in all_annotated_frames.items():
                for frame_num in frame_set:
                    evaluation_frames.add((video_name, frame_num))

        return evaluation_frames

    @staticmethod
    def evaluate_frame_level_with_threshold(tubelet_predictions: pd.DataFrame,
                                            frame_gt: Dict[str, Dict[int, int]],
                                            confidence_threshold: float = 0.5,
                                            all_annotated_frames: Optional[Dict[str, Set[int]]] = None,
                                            T: int = 3) -> pd.DataFrame:
        """Complete frame-level evaluation with confidence thresholding."""

        print(f"🔍 Evaluating frame-level performance (threshold={confidence_threshold:.1f})")

        # Get frame coverage from drone predictions
        frame_coverage = TubeletFrameMapper.map_tubelets_to_frame_coverage(tubelet_predictions, T)

        # Get complete set of frames to evaluate
        evaluation_frames = TubeletFrameMapper.create_complete_frame_evaluation_set(
            tubelet_predictions, frame_gt, all_annotated_frames, T
        )

        frame_results = []

        for video_name, frame_num in evaluation_frames:
            # Ground truth for this frame
            gt_label = frame_gt.get(video_name, {}).get(frame_num, 0)

            # Get covering drone tubelets
            frame_key = (video_name, frame_num)
            covering_tubelets = frame_coverage.get(frame_key, [])

            # Frame prediction
            pred_label, max_confidence = TubeletFrameMapper.get_frame_prediction_with_confidence(
                covering_tubelets, confidence_threshold
            )

            frame_results.append({
                'video_name': video_name,
                'frame_number': frame_num,
                'ground_truth': gt_label,
                'predicted_label': pred_label,
                'max_drone_confidence': max_confidence,
                'num_covering_tubelets': len(covering_tubelets),
                'covering_tubelet_centers': [t['tubelet_center'] for t in
                                             covering_tubelets] if covering_tubelets else []
            })

        results_df = pd.DataFrame(frame_results)

        print(f"📊 Evaluated {len(results_df)} frames")
        print(f"   - Frames with GT drones: {sum(results_df['ground_truth'])}")
        print(f"   - Frames with predicted drones: {sum(results_df['predicted_label'])}")

        return results_df

    @staticmethod
    def analyze_tubelet_coverage(tubelet_predictions: pd.DataFrame,
                                 frame_gt: Dict[str, Dict[int, int]],
                                 T: int = 3) -> Dict[str, any]:
        """Analyze how well tubelets cover ground truth frames."""

        # Get all ground truth drone frames
        gt_drone_frames = set()
        for video_name, video_gt in frame_gt.items():
            for frame_num in video_gt.keys():
                gt_drone_frames.add((video_name, frame_num))

        # Get all frames covered by tubelets
        covered_frames = set()
        for _, pred in tubelet_predictions.iterrows():
            covered_frames_list = TubeletFrameMapper.get_covered_frames(pred['center_frame'], T)
            for frame_num in covered_frames_list:
                covered_frames.add((pred['video_name'], frame_num))

        # Calculate coverage statistics
        covered_gt_frames = gt_drone_frames.intersection(covered_frames)
        uncovered_gt_frames = gt_drone_frames - covered_frames

        coverage_stats = {
            'total_gt_drone_frames': len(gt_drone_frames),
            'covered_gt_frames': len(covered_gt_frames),
            'uncovered_gt_frames': len(uncovered_gt_frames),
            'coverage_percentage': (len(covered_gt_frames) / len(gt_drone_frames) * 100) if gt_drone_frames else 0,
            'total_covered_frames': len(covered_frames),
            'uncovered_gt_frame_list': list(uncovered_gt_frames)
        }

        return coverage_stats

    @staticmethod
    def get_tubelet_statistics(tubelet_predictions: pd.DataFrame,
                               frame_gt: Dict[str, Dict[int, int]],
                               T: int = 3) -> Dict[str, any]:
        """Get comprehensive statistics about tubelet predictions."""

        total_tubelets = len(tubelet_predictions)
        drone_tubelets = len(tubelet_predictions[tubelet_predictions['prediction'] == 1])
        no_drone_tubelets = total_tubelets - drone_tubelets

        # Confidence statistics for drone predictions
        drone_preds = tubelet_predictions[tubelet_predictions['prediction'] == 1]

        if len(drone_preds) > 0:
            conf_stats = {
                'min_confidence': drone_preds['drone_confidence'].min(),
                'max_confidence': drone_preds['drone_confidence'].max(),
                'mean_confidence': drone_preds['drone_confidence'].mean(),
                'median_confidence': drone_preds['drone_confidence'].median()
            }
        else:
            conf_stats = {
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'mean_confidence': 0.0,
                'median_confidence': 0.0
            }

        # Coverage analysis
        coverage_stats = TubeletFrameMapper.analyze_tubelet_coverage(tubelet_predictions, frame_gt, T)

        return {
            'total_tubelets': total_tubelets,
            'drone_tubelets': drone_tubelets,
            'no_drone_tubelets': no_drone_tubelets,
            'drone_percentage': (drone_tubelets / total_tubelets * 100) if total_tubelets > 0 else 0,
            'confidence_stats': conf_stats,
            'coverage_stats': coverage_stats
        }

    @staticmethod
    def print_tubelet_analysis(tubelet_predictions: pd.DataFrame,
                               frame_gt: Dict[str, Dict[int, int]],
                               T: int = 3) -> None:
        """Print comprehensive tubelet analysis summary."""

        stats = TubeletFrameMapper.get_tubelet_statistics(tubelet_predictions, frame_gt, T)

        print(f"\n{'=' * 60}")
        print("TUBELET-TO-FRAME MAPPING ANALYSIS")
        print(f"{'=' * 60}")
        print(f"{'Total Tubelets':<25}: {stats['total_tubelets']}")
        print(f"{'Drone Tubelets':<25}: {stats['drone_tubelets']} ({stats['drone_percentage']:.1f}%)")
        print(f"{'No-Drone Tubelets':<25}: {stats['no_drone_tubelets']}")

        print(f"\nDrone Confidence Statistics:")
        conf = stats['confidence_stats']
        print(f"{'Min Confidence':<25}: {conf['min_confidence']:.3f}")
        print(f"{'Max Confidence':<25}: {conf['max_confidence']:.3f}")
        print(f"{'Mean Confidence':<25}: {conf['mean_confidence']:.3f}")
        print(f"{'Median Confidence':<25}: {conf['median_confidence']:.3f}")

        print(f"\nFrame Coverage Analysis:")
        cov = stats['coverage_stats']
        print(f"{'GT Drone Frames':<25}: {cov['total_gt_drone_frames']}")
        print(f"{'Covered GT Frames':<25}: {cov['covered_gt_frames']}")
        print(f"{'Uncovered GT Frames':<25}: {cov['uncovered_gt_frames']}")
        print(f"{'Coverage Percentage':<25}: {cov['coverage_percentage']:.1f}%")
        print(f"{'Total Covered Frames':<25}: {cov['total_covered_frames']}")

        if cov['uncovered_gt_frames'] > 0:
            print(f"\n⚠️ Uncovered GT frames (potential missed detections):")
            for video, frame in cov['uncovered_gt_frame_list'][:10]:  # Show first 10
                print(f"   {video}: frame {frame}")
            if len(cov['uncovered_gt_frame_list']) > 10:
                print(f"   ... and {len(cov['uncovered_gt_frame_list']) - 10} more")

        print(f"{'=' * 60}")

    @staticmethod
    def save_frame_mapping_details(tubelet_predictions: pd.DataFrame,
                                   frame_results: pd.DataFrame,
                                   output_path: str,
                                   T: int = 3) -> None:
        """Save detailed frame mapping information for analysis."""

        # Get frame coverage mapping
        frame_coverage = TubeletFrameMapper.map_tubelets_to_frame_coverage(tubelet_predictions, T)

        mapping_details = []

        for _, frame_result in frame_results.iterrows():
            video_name = frame_result['video_name']
            frame_num = frame_result['frame_number']
            frame_key = (video_name, frame_num)

            covering_tubelets = frame_coverage.get(frame_key, [])

            for i, tubelet in enumerate(covering_tubelets):
                mapping_details.append({
                    'video_name': video_name,
                    'frame_number': frame_num,
                    'frame_gt': frame_result['ground_truth'],
                    'frame_prediction': frame_result['predicted_label'],
                    'tubelet_index': i,
                    'tubelet_center': tubelet['tubelet_center'],
                    'tubelet_confidence': tubelet['drone_confidence'],
                    'tubelet_path': tubelet['tubelet_path']
                })

            # Also add frames with no covering tubelets
            if not covering_tubelets:
                mapping_details.append({
                    'video_name': video_name,
                    'frame_number': frame_num,
                    'frame_gt': frame_result['ground_truth'],
                    'frame_prediction': frame_result['predicted_label'],
                    'tubelet_index': -1,
                    'tubelet_center': -1,
                    'tubelet_confidence': 0.0,
                    'tubelet_path': 'NO_COVERAGE'
                })

        df = pd.DataFrame(mapping_details)
        df.to_csv(output_path, index=False)
        print(f"💾 Frame mapping details saved to: {output_path}")