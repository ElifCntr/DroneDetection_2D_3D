# src/detection/inference/interactive_detect.py - CLEAN VERSION

import os
import cv2
import glob
import numpy as np
from detection.background import create as bg_create
from detection.preprocess import create as pp_create
from detection.utils import create as ut_create

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def load_video_list_from_splits(splits_file_path: str, split_name: str = "test"):
    """Load video names from splits file"""
    video_names = []
    try:
        with open(splits_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.startswith(("#", "train:", "val:", "test:")):
                video_names.append(line)
    except FileNotFoundError:
        print(f"[ERROR] Splits file not found: {splits_file_path}")
        return []

    return video_names


def find_video_files(raw_video_dir: str, video_names: list):
    """Find actual video files from raw directory"""
    found_videos = []
    for video_name in video_names:
        video_path = os.path.join(raw_video_dir, video_name)
        if os.path.exists(video_path):
            found_videos.append(video_path)
        else:
            print(f"[WARN] Video not found: {video_name}")
    return found_videos


def simple_frame_diff(frame, prev_frame, threshold=25):
    """Simple frame differencing for motion detection"""
    if prev_frame is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def interactive_process_video(video_path: str, cfg: dict):
    """Process video interactively - clean and simple"""

    scale = cfg.get('visualize', {}).get('scale_factor', 0.3)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return False

    print(f"[INFO] Processing {video_path}")
    print("[INFO] Controls: Any key = next frame, 'q' = quit, 'n' = next video")

    # Initialize processors
    bg = bg_create(cfg["background"]["method"], cfg)
    thr = pp_create("Threshold", cfg)
    blur = pp_create("Blur", cfg)
    morph = pp_create("Morphology", cfg)
    merge = pp_create("Merge_contours", cfg)  # Add merge processor

    # Ground truth setup
    ann_dir = cfg["paths"]["annotations"]
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    gt_file = os.path.join(ann_dir, f"{video_id}.txt")
    gt_mod = ut_create("ground_truth")

    try:
        gt_boxes = gt_mod.load_ground_truth(gt_file)
        print(f"[INFO] Ground-truth loaded for {video_id}")
    except FileNotFoundError:
        print(f"[WARN] No ground-truth file for {video_id}")
        gt_boxes = {}

    frame_idx = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[INFO] End of video reached")
            break

        # Hybrid BGS + Frame differencing for first 10 frames
        if frame_idx < 10 and prev_frame is not None:
            bgs_mask = bg.apply(frame)
            diff_mask = simple_frame_diff(frame, prev_frame, threshold=30)

            # Combine masks
            raw_mask = cv2.bitwise_or(bgs_mask, diff_mask)
            mode_text = f"Hybrid (frame {frame_idx})"

            # Debug info
            bgs_pixels = np.count_nonzero(bgs_mask)
            diff_pixels = np.count_nonzero(diff_mask)
            combined_pixels = np.count_nonzero(raw_mask)
            print(f"[DEBUG] BGS: {bgs_pixels}, Diff: {diff_pixels}, Combined: {combined_pixels}")

        else:
            raw_mask = bg.apply(frame)
            mode_text = "BGS only"

        # Processing pipeline
        binary_mask = thr.apply(raw_mask)

        # ADAPTIVE LOGIC - Check noise level BEFORE final processing
        # Quick contour count to determine filtering approach
        quick_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        quick_count = len(quick_contours)

        # Apply adaptive settings IMMEDIATELY
        if quick_count > 150:  # Threshold for switching to aggressive
            # AGGRESSIVE SETTINGS
            morph._min_area = 50
            morph._open_kernel = 1
            morph._close_kernel = 5
            blur._ksize = 3
            # Update BGS variance threshold for next frame
            bg._var_threshold = 16
            filter_mode = "aggressive"
            print(f"[ADAPTIVE] Switching to AGGRESSIVE mode ({quick_count} initial contours)")
        else:
            # GENTLE SETTINGS
            morph._min_area = 10
            morph._open_kernel = 1
            morph._close_kernel = 5
            blur._ksize = 1
            # Update BGS variance threshold for next frame
            bg._var_threshold = 12
            filter_mode = "gentle"
            print(f"[ADAPTIVE] Using GENTLE mode ({quick_count} initial contours)")

        # Apply blur with updated settings
        blurred_mask = blur.apply(binary_mask)

        # Apply morphology with updated settings
        morph_result = morph.apply(blurred_mask)
        if len(morph_result) == 2:
            cleaned_mask, raw_boxes = morph_result
        else:
            cleaned_mask, raw_boxes, _ = morph_result

        '''# Merge overlapping/close contours
        merged_boxes = merge.apply(raw_boxes)

        num_detections = len(raw_boxes)
        num_merged = len(merged_boxes)

        print(f"[MERGE] Raw boxes: {num_detections} → Merged boxes: {num_merged}")

        print(
            f"[INFO] Frame {frame_idx}: {mode_text} | {filter_mode} | Raw: {num_detections} → Merged: {num_merged} detections")
        '''
        # In your interactive_detect.py, skip merging:
        merged_boxes = raw_boxes  # Skip merging

        # Get ground truth
        gts = gt_boxes.get(frame_idx, [])

        # Create displays
        original = cv2.resize(frame, None, fx=scale, fy=scale)
        bgs_display = cv2.resize(cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale)
        morph_display = cv2.resize(cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale)

        # Draw detections on original
        detections_display = original.copy()
        for (x, y, w, h) in merged_boxes:  # Use merged boxes instead of raw boxes
            x_s, y_s, w_s, h_s = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
            cv2.rectangle(detections_display, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 255, 0), 2)

        # Draw GT
        gt_display = original.copy()
        for (x, y, w, h) in gts:
            x_s, y_s, w_s, h_s = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
            cv2.rectangle(gt_display, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 0, 255), 2)

        # Show windows
        cv2.imshow(f"Frame {frame_idx} - Original", original)
        cv2.imshow(f"Frame {frame_idx} - BGS ({mode_text})", bgs_display)
        cv2.imshow(f"Frame {frame_idx} - Morphology ({filter_mode})", morph_display)
        cv2.imshow(f"Frame {frame_idx} - Detections (Raw:{num_detections} Merged:{num_merged})", detections_display)
        cv2.imshow(f"Frame {frame_idx} - Ground Truth ({len(gts)})", gt_display)

        # Store previous frame
        prev_frame = frame.copy()

        # Wait for key
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("[INFO] Quitting...")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key == ord('n'):
            print("[INFO] Next video...")
            cap.release()
            cv2.destroyAllWindows()
            return True

        cv2.destroyAllWindows()
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return True


def run_interactive_detection(cfg: dict, splits_file: str = None):
    """Main function"""
    if splits_file and os.path.exists(splits_file):
        video_names = load_video_list_from_splits(splits_file)
        if not video_names:
            print(f"[ERROR] No videos found in splits file")
            return

        raw_video_dir = cfg["paths"]["raw_videos"]
        video_paths = find_video_files(raw_video_dir, video_names)
    else:
        # Find all videos in raw directory
        raw_video_dir = cfg["paths"]["raw_videos"]
        video_paths = []
        for ext in VIDEO_EXTS:
            video_paths.extend(glob.glob(os.path.join(raw_video_dir, ext)))

    if not video_paths:
        print(f"[ERROR] No videos found")
        return

    print(f"[INFO] Found {len(video_paths)} videos")

    for i, video_path in enumerate(video_paths):
        print(f"\n[INFO] Video {i + 1}/{len(video_paths)}: {os.path.basename(video_path)}")
        if not interactive_process_video(video_path, cfg):
            break

    print("[INFO] Done")


def test_bgs_interactive(cfg: dict, video_path: str = None, splits_file: str = None):
    """Simple wrapper function"""
    if video_path:
        interactive_process_video(video_path, cfg)
    else:
        run_interactive_detection(cfg, splits_file)