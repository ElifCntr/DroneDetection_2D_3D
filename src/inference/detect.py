# src/detection/inference/detect.py

import os
import cv2
from detection.background import create as bg_create
from detection.preprocessing import create as pp_create
from detection.utils import create as ut_create
from utils.metrics import iou
from detection.preprocessing.roi import ROI

VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv")


def process_one_video(video_path: str, cfg: dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    print(f"[INFO] Processing {video_path}")

    # —— Initialize streaming metrics —— #
    total_frames     = 0
    total_props      = 0
    total_gt_boxes   = 0
    detected_gt_boxes = 0

    # 3) Instantiate pipeline stages
    bg      = bg_create(cfg["background"]["method"], cfg)
    thr     = pp_create("Threshold", cfg)
    blur    = pp_create("Blur", cfg)
    morph   = pp_create("Morphology", cfg)
    merge   = pp_create("Merge_contours", cfg)
    roi_ext = ROI(cfg)

    # Ground-truth setup
    ann_dir  = cfg["paths"]["annotations"]
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    gt_file  = os.path.join(ann_dir, f"{video_id}.txt")
    gt_mod   = ut_create("ground_truth")
    try:
        gt_boxes = gt_mod.load_ground_truth(gt_file)
        print(f"[INFO] Ground-truth file found for {gt_file}.")
    except FileNotFoundError:
        print(f"[WARN] No ground-truth file for {video_id}, skipping GT.")
        gt_boxes = {}

    # Visualization settings
    show = cfg.get("visualize", {}).get("show_preprocess", False)
    wait = cfg.get("visualize", {}).get("wait_ms", 1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4) Background subtraction
        raw_bgs_mask = bg.apply(frame)

        # 5) Preprocessing chain
        binary_mask, = (thr.apply(raw_bgs_mask),)
        blurred_mask = blur.apply(binary_mask)
        cleaned_mask, raw_boxes = morph.apply(blurred_mask)
        merged_boxes = merge.apply(raw_boxes)

        # —— Update streaming metrics —— #
        total_frames += 1
        total_props  += len(merged_boxes)

        # For every GT box in this frame, check if any proposal hits it
        gts = gt_boxes.get(frame_idx, [])
        total_gt_boxes += len(gts)
        for gt in gts:
            if any(iou(prop, gt) >= cfg["scoring"]["iou_thresh"] for prop in merged_boxes):
                detected_gt_boxes += 1


        # 6) Crop & save ROIs
        roi_ext.save_rois(frame, merged_boxes, frame_idx)

        # 7) Visualization
        if show:
            disp = frame.copy()
            # proposals in green
            for (x, y, w, h) in merged_boxes:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0,255,0), 2)
            # GT in red
            for (x, y, w, h) in gts:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0,0,255), 2)
            cv2.imshow("detections", disp)
            if cv2.waitKey(wait) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if show:
            cv2.destroyAllWindows()

    cap.release()

    # —— Compute final metrics —— #
    avg_props = total_props / total_frames if total_frames else 0.0
    det_rate  = detected_gt_boxes / total_gt_boxes if total_gt_boxes else 1.0

    return avg_props, det_rate
