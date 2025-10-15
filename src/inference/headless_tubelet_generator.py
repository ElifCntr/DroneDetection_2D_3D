# src/inference/headless_tubelet_generator.py - No PIL dependencies

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from detection.background import create as bg_create
from detection.preprocessing import create as pp_create
from detection.utils import create as ut_create
from utils.metrics import iou


def simple_frame_diff(frame, prev_frame, threshold=5):
    """Frame differencing for early frames"""
    if prev_frame is None:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def extract_all_tubelets_from_video(video_path, all_detections, T=3, out_size=112):
    """Extract ALL tubelets in one pass through the video"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    print(f"[INFO] Pre-loading video frames...")
    frames = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames[frame_idx] = frame
        frame_idx += 1

    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames")

    # Generate all tubelets using cached frames
    tubelets = {}

    for center_frame, detections in all_detections.items():
        if center_frame < T // 2:
            continue

        frame_tubelets = []

        for det_idx, bbox in enumerate(detections):
            x, y, w, h = bbox

            # Get frame range
            start_frame = max(0, center_frame - T // 2)
            tubelet_frames = []

            for f_idx in range(start_frame, start_frame + T):
                if f_idx in frames:
                    frame = frames[f_idx]
                else:
                    # Use last available frame
                    frame = frames[max(frames.keys())]

                # Crop with padding
                pad = max(5, min(w, h) // 4)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                # Resize and convert
                resized = cv2.resize(cropped, (out_size, out_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tubelet_frames.append(rgb)

            if len(tubelet_frames) == T:
                tubelet = np.stack(tubelet_frames, axis=0)
                frame_tubelets.append(tubelet)

        if frame_tubelets:
            tubelets[center_frame] = frame_tubelets

    return tubelets


def process_video_optimized(video_path, cfg):
    """Optimized video processing"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, {}

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[INFO] Processing {video_id}")

    # Initialize pipeline
    bg = bg_create(cfg["background"]["method"], cfg)
    thr = pp_create("Threshold", cfg)
    blur = pp_create("Blur", cfg)
    morph = pp_create("Morphology", cfg)

    # Load ground truth
    ann_dir = cfg["paths"]["annotations_dir"]
    gt_file = os.path.join(ann_dir, f"{video_id}.txt")
    gt_mod = ut_create("ground_truth")

    try:
        gt_boxes = gt_mod.load_ground_truth(gt_file)
    except FileNotFoundError:
        gt_boxes = {}

    all_detections = {}
    frame_idx = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection logic
        if frame_idx < 10 and prev_frame is not None:
            bgs_mask = bg.apply(frame)
            diff_mask = simple_frame_diff(frame, prev_frame)
            raw_mask = cv2.bitwise_or(bgs_mask, diff_mask)
        else:
            raw_mask = bg.apply(frame)

        binary_mask = thr.apply(raw_mask)
        quick_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Adaptive settings
        if len(quick_contours) > 150:
            morph._min_area = 50
            blur._ksize = 3
            bg._var_threshold = 16
        else:
            morph._min_area = 10
            blur._ksize = 1
            bg._var_threshold = 12

        blurred_mask = blur.apply(binary_mask)
        morph_result = morph.apply(blurred_mask)

        if len(morph_result) == 2:
            cleaned_mask, boxes = morph_result
        else:
            cleaned_mask, boxes, _ = morph_result

        all_detections[frame_idx] = boxes
        prev_frame = frame.copy()
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames")

    cap.release()
    return all_detections, gt_boxes


def generate_tubelets_optimized(video_path, all_detections, gt_boxes, cfg):
    """Generate tubelets with single video pass"""

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    T = cfg["tubelet_gen"]["T"]
    out_size = cfg["tubelet_gen"]["out_size"]
    iou_thresh = cfg["scoring"]["iou_thresh"]

    print(f"[INFO] Generating tubelets for {video_id}...")

    # Get all tubelets in one pass
    all_tubelets_by_frame = extract_all_tubelets_from_video(video_path, all_detections, T, out_size)

    tubelets = []

    for frame_idx, frame_tubelets in all_tubelets_by_frame.items():
        detections = all_detections[frame_idx]
        frame_gts = gt_boxes.get(frame_idx, [])

        for det_idx, (tubelet, detection) in enumerate(zip(frame_tubelets, detections)):
            # Calculate IoU with ground truth
            max_iou = 0
            for gt_box in frame_gts:
                current_iou = iou(detection, gt_box)
                max_iou = max(max_iou, current_iou)

            label = 1 if max_iou >= iou_thresh else 0

            tubelets.append({
                'tubelet': tubelet,
                'label': label,
                'video_id': video_id,
                'frame_idx': frame_idx,
                'bbox': detection,
                'max_iou': max_iou
            })

    print(f"[INFO] Generated {len(tubelets)} tubelets")
    return tubelets


def process_test_split_clean(cfg):
    """Clean version - generate tubelets and save with proper CSV"""

    # Load test videos
    test_list_path = cfg["paths"]["splits"]["test_list"]
    with open(test_list_path, 'r') as f:
        video_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"[INFO] Processing {len(video_names)} videos")

    raw_dir = cfg["paths"]["input_video_dir"]
    tubelet_dir = Path(cfg["paths"]["tubelets_dir"]) / "test"
    tubelet_dir.mkdir(parents=True, exist_ok=True)

    all_tubelets = []

    for i, video_name in enumerate(video_names):
        print(f"\n[INFO] Video {i + 1}/{len(video_names)}: {video_name}")

        video_path = os.path.join(raw_dir, video_name)
        if not os.path.exists(video_path):
            print(f"[WARN] Video not found: {video_path}")
            continue

        # Run detection
        detections, gt_boxes = process_video_optimized(video_path, cfg)

        # Generate tubelets
        video_tubelets = generate_tubelets_optimized(video_path, detections, gt_boxes, cfg)

        all_tubelets.extend(video_tubelets)
        print(f"[INFO] Total tubelets so far: {len(all_tubelets)}")

    if not all_tubelets:
        print("[ERROR] No tubelets generated!")
        return None

    print(f"\n[INFO] Saving {len(all_tubelets)} tubelets...")

    # Save tubelets and create CSV index
    tubelet_data = []

    for i, tubelet_info in enumerate(all_tubelets):
        # Create filename
        filename = f"test_{i:06d}.npy"
        filepath = tubelet_dir / filename

        # Save tubelet
        np.save(filepath, tubelet_info['tubelet'])

        # Add to CSV index with all info
        tubelet_data.append({
            'tubelet_id': i,
            'filename': filename,  # Just filename, not full path
            'filepath': str(filepath),  # Full path for reference
            'label': tubelet_info['label'],
            'video_id': tubelet_info['video_id'],
            'frame_idx': tubelet_info['frame_idx'],
            'bbox_x': tubelet_info['bbox'][0],
            'bbox_y': tubelet_info['bbox'][1],
            'bbox_w': tubelet_info['bbox'][2],
            'bbox_h': tubelet_info['bbox'][3],
            'max_iou': tubelet_info['max_iou'],
            'tubelet_shape': str(tubelet_info['tubelet'].shape)
        })

        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{len(all_tubelets)} tubelets")

    # Save CSV index
    df = pd.DataFrame(tubelet_data)
    csv_path = cfg["paths"]["tubelet_indexes"]["test_csv"]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"\n[SUCCESS] Tubelet Generation Complete!")
    print(f"  Total tubelets: {len(all_tubelets)}")
    print(f"  Positive samples: {sum(df['label'] == 1)}")
    print(f"  Negative samples: {sum(df['label'] == 0)}")
    print(f"  Tubelets saved to: {tubelet_dir}")
    print(f"  Index saved to: {csv_path}")

    # Print sample of CSV
    print(f"\n[INFO] Sample of generated CSV:")
    print(df.head())

    return csv_path


if __name__ == "__main__":
    import yaml

    # Load config
    config_path = "configs/experiment.yaml"
    print(f"[INFO] Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Ensure test_csv path exists
    if "test_csv" not in cfg["paths"]["tubelet_indexes"]:
        cfg["paths"]["tubelet_indexes"]["test_csv"] = "data/tubelets/test_index.csv"

    print(f"[INFO] Config loaded:")
    print(f"  Input videos: {cfg['paths']['input_video_dir']}")
    print(f"  Annotations: {cfg['paths']['annotations_dir']}")
    print(f"  Output tubelets: {cfg['paths']['tubelets_dir']}")
    print(f"  Test list: {cfg['paths']['splits']['test_list']}")
    print(f"  Tubelet T={cfg['tubelet_gen']['T']}, size={cfg['tubelet_gen']['out_size']}")

    print("\n[INFO] Starting CLEAN tubelet generation...")

    try:
        csv_path = process_test_split_clean(cfg)
        if csv_path:
            print(f"\n✅ SUCCESS! Ready for model evaluation")
            print(f"   CSV index: {csv_path}")
        else:
            print(f"\n❌ FAILED - No tubelets generated")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback

        traceback.print_exc()