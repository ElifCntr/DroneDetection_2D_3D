# src/inference/generate_tubelets.py

from __future__ import annotations
import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2 as cv
import numpy as np
import yaml

# your packages
from detection.background import create as bg_create
from detection.preprocessing import create as pp_create
from detection.utils import create as ut_create
from detection.preprocessing.roi import adjust_bounding_boxes_to_square

Box = Tuple[int, int, int, int]  # (x, y, w, h)


def enlarge_then_square_clamped(
    box: Box, img_w: int, img_h: int,
    scale_factor: float, pad_px: int, min_side: int
) -> Optional[Box]:
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    cx, cy = x + w // 2, y + h // 2
    s = max(w, h)
    S = int(round(s * max(1.0, scale_factor))) + 2 * max(0, pad_px)
    X, Y = cx - S // 2, cy - S // 2
    sq = adjust_bounding_boxes_to_square([(X, Y, S, S)], img_w, img_h)
    if not sq:
        return None
    sx, sy, sw, sh = sq[0]
    if max(sw, sh) < min_side:
        return None
    return (sx, sy, sw, sh)


def crop_tubelet_rgb(
    frames_bgr: List[np.ndarray], sq_box: Box, T: int, out_size: int
) -> np.ndarray:
    x, y, w, h = sq_box
    H = W = out_size
    clip = []
    for t in range(T):
        f = frames_bgr[t] if t < len(frames_bgr) else frames_bgr[-1]
        crop = f[y:y+h, x:x+w]  # BGR
        interp = cv.INTER_AREA if (w > W or h > H) else cv.INTER_LINEAR
        resized = cv.resize(crop, (W, H), interpolation=interp)
        rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        clip.append(rgb)
    return np.stack(clip, axis=0).astype(np.uint8)


def read_list(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def process_split(
    split_name: str,
    video_list_path: str,
    cfg: dict,
    save_csv_path: str,
) -> None:
    videos = read_list(video_list_path)
    if not videos:
        print(f"[WARN] Empty list: {video_list_path}")
        return

    tg = cfg.get("tubelet_gen", {})
    T = int(tg.get("T", 3))
    out_size = int(tg.get("out_size", 112))
    scale_factor = float(tg.get("scale_factor", 1.5))
    pad_px = int(tg.get("pad_px", 4))
    min_side = int(tg.get("min_side", 24))
    cap_neg_per_frame = int(tg.get("cap_neg_per_frame", 10)) # per-frame cap for negatives only

    raw_dir = cfg["paths"]["input_video_dir"]
    ann_dir = cfg["paths"]["annotations_dir"]
    tubelets_root = cfg["paths"].get("tubelets_dir", "data/tubelets")
    split_root = os.path.join(tubelets_root, split_name)
    ensure_dir(split_root)

    bg = bg_create(cfg["background"]["method"], cfg)
    thr = pp_create("Threshold", cfg)
    blur = pp_create("Blur", cfg)
    morph = pp_create("Morphology", cfg)
    merge = pp_create("Merge_contours", cfg)
    gt_mod = ut_create("ground_truth")

    ensure_dir(Path(save_csv_path).parent)
    csv_file = open(save_csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["path", "label"])

    print(f"[INFO] Processing split: {split_name}, {len(videos)} videos found")
    total_saved_pos = 0
    total_saved_neg = 0

    for vid_idx, vid_name in enumerate(videos):
        video_path = os.path.join(raw_dir, vid_name)
        base = Path(vid_name).stem
        vid_out_dir = os.path.join(split_root, base)
        ensure_dir(vid_out_dir)

        print(f"[INFO] ({vid_idx+1}/{len(videos)}) Starting video: {base}")

        if hasattr(bg, "reset"):
            bg.reset()

        gt_txt = os.path.join(ann_dir, f"{base}.txt")
        gt_csv = os.path.join(ann_dir, f"{base}.csv")
        gt_file = gt_txt if os.path.exists(gt_txt) else (gt_csv if os.path.exists(gt_csv) else None)
        if gt_file is None:
            print(f"[WARN] No GT for {base}, skipping.")
            continue

        try:
            gt_by_frame = gt_mod.load_ground_truth(gt_file)
        except Exception as e:
            print(f"[ERROR] Failed to load GT for {base}: {e}")
            continue

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {video_path}")
            continue

        frames_buf: List[np.ndarray] = []
        frame_idx = 0
        while len(frames_buf) < T:
            ret, f = cap.read()
            if not ret:
                break
            frames_buf.append(f)

        while len(frames_buf) > 0:
            frame0 = frames_buf[0]
            H, W = frame0.shape[:2]
            gts = gt_by_frame.get(frame_idx, [])

            raw_mask = bg.apply(frame0)
            binary = thr.apply(raw_mask)
            blurred = blur.apply(binary)
            _, raw_boxes = morph.apply(blurred)
            boxes = merge.apply(raw_boxes)

            gt_boxes_sq: List[Box] = []
            for gt in gts:
                sq = enlarge_then_square_clamped(gt, W, H, scale_factor, pad_px, min_side)
                if sq:
                    gt_boxes_sq.append(sq)
            if False:
                neg_saved_this_frame = 0
                for box in boxes:
                    sq = enlarge_then_square_clamped(box, W, H, scale_factor, pad_px, min_side)
                    if sq and all(iou(sq, gt) == 0.0 for gt in gt_boxes_sq):
                        if neg_saved_this_frame >= cap_neg_per_frame:
                            break
                        fname = f"neg_{frame_idx:06d}_{neg_saved_this_frame:03d}.npy"
                        out_path = os.path.join(vid_out_dir, fname)
                        if os.path.exists(out_path):
                            print(f"[INFO] Skipping existing {fname} (video={base}, frame={frame_idx})")
                            neg_saved_this_frame += 1
                            continue
                        clip = crop_tubelet_rgb(frames_buf, sq, T, out_size)
                        np.save(out_path, clip)
                        rel = os.path.join(split_root, base, fname).replace("\\", "/")
                        writer.writerow([rel, 0])
                        neg_saved_this_frame += 1
                        total_saved_neg += 1
                        print(f"[INFO] Saved {fname} (video={base}, frame={frame_idx})")
                    else:
                        print(f"[SKIP] Proposal at frame={frame_idx} overlaps with GT, skipping.")

            # POSITIVES (no cap, but skip if already saved)
            for idx, gt in enumerate(gt_boxes_sq):
                fname = f"pos_{frame_idx:06d}_{idx:03d}.npy"
                out_path = os.path.join(vid_out_dir, fname)
                if os.path.exists(out_path):
                    print(f"[INFO] Skipping existing {fname} (video={base}, frame={frame_idx})")
                    continue
                clip = crop_tubelet_rgb(frames_buf, gt, T, out_size)
                np.save(out_path, clip)
                rel = os.path.join(split_root, base, fname).replace("\\", "/")
                writer.writerow([rel, 1])
                total_saved_pos += 1
                print(f"[INFO] Saved {fname} (video={base}, frame={frame_idx})")

            frame_idx += 1
            frames_buf.pop(0)
            ret, f = cap.read()
            if ret:
                frames_buf.append(f)

        cap.release()
        print(f"[INFO] Finished {base}")

    csv_file.close()
    print(f"[INFO] Done with split: {split_name}. Total saved: {total_saved_pos} positives, {total_saved_neg} negatives. Index → {save_csv_path}")


# ---------------- entry point ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to experiment.yaml")
    ap.add_argument(
        "--split", action="append", choices=["train", "val", "test"],
        help="If omitted, processes both train and val."
    )
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    train_list = cfg["paths"]["splits"]["train_list"]
    val_list   = cfg["paths"]["splits"]["val_list"]
    test_list   = cfg["paths"]["splits"]["test_list"]

    train_csv  = cfg["paths"]["tubelet_indexes"]["train_csv"]
    val_csv    = cfg["paths"]["tubelet_indexes"]["val_csv"]
    test_csv = cfg["paths"]["tubelet_indexes"]["test_csv"]

    splits = args.split or ["train", "val", "test"]
    if "train" in splits:
        process_split("train", train_list, cfg, train_csv)
    if "val" in splits:
        process_split("val",   val_list,   cfg, val_csv)
    if "test" in splits:
        test_list = cfg["paths"]["splits"]["test_list"]
        test_csv = cfg["paths"]["tubelet_indexes"]["test_csv"]
        process_split("test", test_list, cfg, test_csv)


if __name__ == "__main__":
    main()
