"""
Universal ground truth loader for frame-level drone detection evaluation.
Handles annotation format: frame_number drone_count [drone_details...]
Works with any model requiring frame-level ground truth annotations.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


class GroundTruthLoader:
    """Universal ground truth loader for frame-level evaluation."""

    @staticmethod
    def load_frame_level_annotations(video_list: List[str],
                                     annotations_dir: str,
                                     file_extension: str = ".txt") -> Dict[str, Dict[int, int]]:
        """Load frame-level ground truth annotations for multiple videos."""

        print("📋 Loading frame-level ground truth...")
        frame_gt = {}

        for video_name in video_list:
            video_base = os.path.splitext(video_name)[0]
            gt_file = os.path.join(annotations_dir, f"{video_base}{file_extension}")

            if os.path.exists(gt_file):
                video_annotations = GroundTruthLoader._parse_annotation_file(gt_file)
                frame_gt[video_name] = video_annotations
                print(f"  {video_name}: {len(video_annotations)} frames with drones")
            else:
                print(f"  ⚠️ Ground truth file not found: {gt_file}")
                frame_gt[video_name] = {}

        return frame_gt

    @staticmethod
    def _parse_annotation_file(file_path: str) -> Dict[int, int]:
        """Parse annotation file and return frame-level labels."""

        video_frames = {}

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        frame_num = int(parts[0])
                        drone_count = int(parts[1])

                        # Frame has drone if drone_count > 0
                        if drone_count > 0:
                            video_frames[frame_num] = 1

                except ValueError as e:
                    print(f"  ⚠️ Skipping invalid line {line_num} in {file_path}: {line}")
                    continue

        except Exception as e:
            print(f"  ❌ Error reading {file_path}: {e}")
            return {}

        return video_frames

    @staticmethod
    def get_all_frame_numbers_from_annotations(annotations_dir: str,
                                               video_list: List[str],
                                               file_extension: str = ".txt") -> Dict[str, Set[int]]:
        """Get all frame numbers (both drone and no-drone) from annotation files."""

        all_frames = {}

        for video_name in video_list:
            video_base = os.path.splitext(video_name)[0]
            gt_file = os.path.join(annotations_dir, f"{video_base}{file_extension}")

            if os.path.exists(gt_file):
                video_frames = set()

                with open(gt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                parts = line.split()
                                if len(parts) >= 1:
                                    frame_num = int(parts[0])
                                    video_frames.add(frame_num)
                            except ValueError:
                                continue

                all_frames[video_name] = video_frames
            else:
                all_frames[video_name] = set()

        return all_frames

    @staticmethod
    def get_all_annotated_frames(frame_gt: Dict[str, Dict[int, int]]) -> Set[Tuple[str, int]]:
        """Get set of all (video, frame) pairs that have annotations."""

        annotated_frames = set()

        for video_name, video_annotations in frame_gt.items():
            for frame_num in video_annotations.keys():
                annotated_frames.add((video_name, frame_num))

        return annotated_frames

    @staticmethod
    def get_video_frame_ranges(frame_gt: Dict[str, Dict[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Get frame number ranges for each video."""

        frame_ranges = {}

        for video_name, video_annotations in frame_gt.items():
            if video_annotations:
                min_frame = min(video_annotations.keys())
                max_frame = max(video_annotations.keys())
                frame_ranges[video_name] = (min_frame, max_frame)
            else:
                frame_ranges[video_name] = (0, 0)

        return frame_ranges

    @staticmethod
    def get_annotation_statistics(frame_gt: Dict[str, Dict[int, int]]) -> Dict[str, any]:
        """Get comprehensive statistics about the ground truth annotations."""

        total_videos = len(frame_gt)
        total_drone_frames = 0
        videos_with_drones = 0
        frame_counts_per_video = []

        for video_name, video_annotations in frame_gt.items():
            drone_frames_in_video = len(video_annotations)
            total_drone_frames += drone_frames_in_video
            frame_counts_per_video.append(drone_frames_in_video)

            if drone_frames_in_video > 0:
                videos_with_drones += 1

        return {
            'total_videos': total_videos,
            'videos_with_drones': videos_with_drones,
            'videos_without_drones': total_videos - videos_with_drones,
            'total_drone_frames': total_drone_frames,
            'avg_drone_frames_per_video': total_drone_frames / total_videos if total_videos > 0 else 0,
            'min_drone_frames_per_video': min(frame_counts_per_video) if frame_counts_per_video else 0,
            'max_drone_frames_per_video': max(frame_counts_per_video) if frame_counts_per_video else 0
        }

    @staticmethod
    def validate_ground_truth(frame_gt: Dict[str, Dict[int, int]],
                              expected_videos: Optional[List[str]] = None) -> bool:
        """Validate ground truth data for completeness and consistency."""

        print("🔍 Validating ground truth data...")

        is_valid = True

        if not frame_gt:
            print("  ❌ No ground truth data loaded")
            return False

        if expected_videos:
            missing_videos = set(expected_videos) - set(frame_gt.keys())
            if missing_videos:
                print(f"  ⚠️ Missing ground truth for videos: {missing_videos}")
                is_valid = False

        empty_videos = [video for video, annotations in frame_gt.items() if not annotations]
        if empty_videos:
            print(f"  ⚠️ Videos with no annotations: {empty_videos}")

        for video_name, video_annotations in frame_gt.items():
            negative_frames = [frame for frame in video_annotations.keys() if frame < 0]
            if negative_frames:
                print(f"  ❌ Video {video_name} has negative frame numbers: {negative_frames}")
                is_valid = False

        if is_valid:
            print("  ✅ Ground truth validation passed")

        return is_valid

    @staticmethod
    def create_frame_level_dataframe(frame_gt: Dict[str, Dict[int, int]],
                                     include_all_frames: bool = False,
                                     video_frame_ranges: Optional[Dict[str, Tuple[int, int]]] = None) -> pd.DataFrame:
        """Create DataFrame with frame-level ground truth."""

        frame_data = []

        for video_name, video_annotations in frame_gt.items():
            if include_all_frames and video_frame_ranges and video_name in video_frame_ranges:
                min_frame, max_frame = video_frame_ranges[video_name]
                for frame_num in range(min_frame, max_frame + 1):
                    gt_label = video_annotations.get(frame_num, 0)
                    frame_data.append({
                        'video_name': video_name,
                        'frame_number': frame_num,
                        'ground_truth': gt_label
                    })
            else:
                for frame_num, gt_label in video_annotations.items():
                    frame_data.append({
                        'video_name': video_name,
                        'frame_number': frame_num,
                        'ground_truth': gt_label
                    })

        return pd.DataFrame(frame_data)

    @staticmethod
    def print_annotation_summary(frame_gt: Dict[str, Dict[int, int]]) -> None:
        """Print formatted summary of ground truth annotations."""

        stats = GroundTruthLoader.get_annotation_statistics(frame_gt)

        print(f"\n{'=' * 50}")
        print("GROUND TRUTH ANNOTATION SUMMARY")
        print(f"{'=' * 50}")
        print(f"{'Total Videos':<25}: {stats['total_videos']}")
        print(f"{'Videos with Drones':<25}: {stats['videos_with_drones']}")
        print(f"{'Videos without Drones':<25}: {stats['videos_without_drones']}")
        print(f"{'Total Drone Frames':<25}: {stats['total_drone_frames']}")
        print(f"{'Avg Frames per Video':<25}: {stats['avg_drone_frames_per_video']:.1f}")
        print(f"{'Min Frames per Video':<25}: {stats['min_drone_frames_per_video']}")
        print(f"{'Max Frames per Video':<25}: {stats['max_drone_frames_per_video']}")
        print(f"{'=' * 50}")

    @staticmethod
    def save_ground_truth_csv(frame_gt: Dict[str, Dict[int, int]],
                              output_path: str) -> None:
        """Save ground truth data to CSV for easy loading."""

        df = GroundTruthLoader.create_frame_level_dataframe(frame_gt)
        df.to_csv(output_path, index=False)
        print(f"💾 Ground truth saved to: {output_path}")

    @staticmethod
    def load_ground_truth_csv(csv_path: str) -> Dict[str, Dict[int, int]]:
        """Load ground truth data from previously saved CSV."""

        df = pd.read_csv(csv_path)
        frame_gt = {}

        for _, row in df.iterrows():
            video_name = row['video_name']
            frame_num = int(row['frame_number'])
            gt_label = int(row['ground_truth'])

            if video_name not in frame_gt:
                frame_gt[video_name] = {}

            if gt_label == 1:
                frame_gt[video_name][frame_num] = gt_label

        print(f"📋 Ground truth loaded from CSV: {csv_path}")
        return frame_gt