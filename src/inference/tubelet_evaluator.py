# src/evaluation/tubelet_evaluator.py - Evaluate model on tubelets with voting

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import os
import yaml


def load_r3d_model(model_path, device='cuda', num_classes=2):
    """Load trained R3D-18 model"""

    print(f"[INFO] Loading R3D-18 model from {model_path}")

    # Import here to avoid the PIL issue
    try:
        from torchvision.models.video import r3d_18
    except ImportError:
        print("[ERROR] Could not import torchvision models")
        print("Try: pip install --upgrade torch torchvision")
        raise

    # Create model
    model = r3d_18(pretrained=False, num_classes=num_classes)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded model_state_dict")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[INFO] Loaded state_dict")
    else:
        model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded direct checkpoint")

    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully on {device}")
    return model


def load_tubelets_from_csv(csv_path, tubelets_dir, max_tubelets=None):
    """Load tubelets based on CSV index"""

    print(f"[INFO] Loading tubelet index from: {csv_path}")
    df = pd.read_csv(csv_path)

    if max_tubelets:
        df = df.head(max_tubelets)
        print(f"[INFO] Limited to first {max_tubelets} tubelets for testing")

    print(f"[INFO] Loading {len(df)} tubelets...")

    tubelets = []
    valid_indices = []

    for idx, row in df.iterrows():
        # Build tubelet path
        if 'filename' in row:
            tubelet_path = Path(tubelets_dir) / row['filename']
        else:
            tubelet_path = Path(row['filepath'])

        if tubelet_path.exists():
            tubelet = np.load(tubelet_path)
            tubelets.append(tubelet)
            valid_indices.append(idx)
        else:
            print(f"[WARN] Tubelet not found: {tubelet_path}")

    # Filter dataframe to only valid tubelets
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    print(f"[INFO] Loaded {len(tubelets)} valid tubelets")
    print(f"[INFO] Positive: {sum(df_valid['label'] == 1)}, Negative: {sum(df_valid['label'] == 0)}")

    return tubelets, df_valid


def predict_tubelets_batch(model, tubelets, batch_size=16, device='cuda'):
    """Get model predictions in batches"""

    print(f"[INFO] Running model inference with batch_size={batch_size}")

    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(tubelets), batch_size):
            batch_tubelets = tubelets[i:i + batch_size]

            # Convert to tensor: (B, C, T, H, W)
            batch_tensor = []
            for tubelet in batch_tubelets:
                # (T, H, W, C) -> (C, T, H, W)
                tensor = torch.from_numpy(tubelet).permute(3, 0, 1, 2).float()
                # Normalize to [0, 1]
                if tensor.max() > 1.0:
                    tensor = tensor / 255.0
                batch_tensor.append(tensor)

            batch_tensor = torch.stack(batch_tensor).to(device)

            # Get predictions
            outputs = model(batch_tensor)

            # Convert to probabilities
            if outputs.shape[1] == 2:  # Binary classification
                probs = F.softmax(outputs, dim=1)[:, 1]  # Positive class probability
            else:  # Single output
                probs = torch.sigmoid(outputs.squeeze())

            all_predictions.extend(probs.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i + len(batch_tubelets)}/{len(tubelets)} tubelets")

    predictions = np.array(all_predictions)
    print(
        f"[INFO] Predictions: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")

    return predictions


def build_frame_to_tubelets_mapping(df_tubelets, T=3):
    """Build mapping from video_frame -> list of tubelet indices"""

    frame_to_tubelets = defaultdict(list)

    for idx, row in df_tubelets.iterrows():
        video_id = row['video_id']
        center_frame = row['frame_idx']

        # This tubelet covers frames [center-T//2, center+T//2]
        start_frame = center_frame - T // 2
        end_frame = center_frame + T // 2

        for frame_idx in range(start_frame, end_frame + 1):
            frame_key = f"{video_id}_{frame_idx}"

            # Store tubelet info for this frame
            frame_to_tubelets[frame_key].append({
                'tubelet_idx': idx,
                'center_frame': center_frame,
                'bbox': [row['bbox_x'], row['bbox_y'], row['bbox_w'], row['bbox_h']],
                'video_id': video_id,
                'frame_position': frame_idx - start_frame,  # 0, 1, or 2 for T=3
                'is_center': (frame_idx == center_frame)
            })

    print(f"[INFO] Built mapping for {len(frame_to_tubelets)} video-frames")
    return frame_to_tubelets


def predict_frame_detections(frame_to_tubelets, tubelet_predictions,
                             voting_method='majority', confidence_threshold=0.5, T=3):
    """Convert tubelet predictions to frame-level detections using voting"""

    print(f"[INFO] Converting tubelet predictions to frame detections")
    print(f"[INFO] Voting method: {voting_method}, Threshold: {confidence_threshold}")

    frame_detections = defaultdict(list)

    for frame_key, tubelets_info in frame_to_tubelets.items():

        votes = []
        confidences = []
        bboxes = []
        weights = []

        for tubelet_info in tubelets_info:
            idx = tubelet_info['tubelet_idx']
            frame_pos = tubelet_info['frame_position']
            bbox = tubelet_info['bbox']
            is_center = tubelet_info['is_center']

            if idx < len(tubelet_predictions):
                confidence = tubelet_predictions[idx]
            else:
                confidence = 0.0

            # Position weighting (center frame gets more weight)
            if voting_method == 'center_only':
                if not is_center:
                    continue  # Skip non-center positions
                weight = 1.0
            else:
                # Weight based on position in tubelet
                if T == 3:
                    pos_weights = [0.7, 1.0, 0.7]  # Less weight for edge frames
                else:
                    pos_weights = [1.0] * T  # Equal weight
                weight = pos_weights[frame_pos] if frame_pos < len(pos_weights) else 0.5

            votes.append(confidence >= confidence_threshold)
            confidences.append(confidence)
            bboxes.append(bbox)
            weights.append(weight * confidence)

        if not votes:
            continue

        # Apply voting strategy
        if voting_method == 'majority':
            # Majority vote
            if sum(votes) > len(votes) / 2:
                # Take bbox with highest weighted confidence
                best_idx = np.argmax(weights)
                frame_detections[frame_key].append(bboxes[best_idx])

        elif voting_method == 'any':
            # Any positive vote counts
            for i, vote in enumerate(votes):
                if vote:
                    frame_detections[frame_key].append(bboxes[i])

        elif voting_method == 'weighted_average':
            # Weighted average of confidences
            if weights:
                avg_confidence = np.average(confidences, weights=weights)
                if avg_confidence >= confidence_threshold:
                    best_idx = np.argmax(weights)
                    frame_detections[frame_key].append(bboxes[best_idx])

        elif voting_method == 'center_only':
            # Only center-frame tubelets (already filtered above)
            if any(votes):
                best_idx = np.argmax(weights)
                frame_detections[frame_key].append(bboxes[best_idx])

    print(f"[INFO] Generated predictions for {len(frame_detections)} frames")
    return dict(frame_detections)


def load_ground_truth(annotations_dir):
    """Load all ground truth annotations"""

    print(f"[INFO] Loading ground truth from: {annotations_dir}")

    # Import here to avoid circular imports
    from detection.utils import create as ut_create

    annotations_dir = Path(annotations_dir)
    all_gt = {}

    for gt_file in annotations_dir.glob("*.txt"):
        video_id = gt_file.stem

        try:
            gt_mod = ut_create("ground_truth")
            video_gt = gt_mod.load_ground_truth(str(gt_file))

            # Add video prefix to frame keys
            for frame_idx, boxes in video_gt.items():
                frame_key = f"{video_id}_{frame_idx}"
                all_gt[frame_key] = boxes

        except Exception as e:
            print(f"[WARN] Could not load GT for {video_id}: {e}")
            continue

    total_gt_boxes = sum(len(boxes) for boxes in all_gt.values())
    print(f"[INFO] Loaded GT for {len(all_gt)} frames, {total_gt_boxes} total boxes")

    return all_gt


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
    box2_xyxy = [x2, y2, x2 + w2, y2 + h2]

    # Intersection
    x_left = max(box1_xyxy[0], box2_xyxy[0])
    y_top = max(box1_xyxy[1], box2_xyxy[1])
    x_right = min(box1_xyxy[2], box2_xyxy[2])
    y_bottom = min(box1_xyxy[3], box2_xyxy[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_frame_metrics(frame_detections, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, F1 from frame predictions and ground truth"""

    print(f"[INFO] Calculating frame-level metrics (IoU threshold: {iou_threshold})")

    all_metrics = []

    # Get all unique frames
    all_frames = set(frame_detections.keys()) | set(ground_truth.keys())

    for frame_key in all_frames:
        pred_boxes = frame_detections.get(frame_key, [])
        gt_boxes = ground_truth.get(frame_key, [])

        # Match predictions to ground truth
        matched_gt = set()
        tp = 0
        fp = 0

        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue

                current_iou = calculate_iou(pred_box, gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        video_id, frame_idx = frame_key.split('_', 1)
        all_metrics.append({
            'video_id': video_id,
            'frame_idx': int(frame_idx),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'num_predictions': len(pred_boxes),
            'num_ground_truth': len(gt_boxes)
        })

    # Calculate overall metrics
    df = pd.DataFrame(all_metrics)

    total_tp = df['tp'].sum()
    total_fp = df['fp'].sum()
    total_fn = df['fn'].sum()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn
    }, df


def evaluate_model_with_voting(model_path, csv_path, tubelets_dir, annotations_dir,
                               voting_methods=['center_only', 'majority', 'weighted_average'],
                               confidence_thresholds=[0.3, 0.5, 0.7],
                               batch_size=16, device='cuda', max_tubelets=None):
    """Complete evaluation with multiple voting strategies"""

    print(f"\n{'=' * 60}")
    print(f"TUBELET MODEL EVALUATION")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")

    # Load model
    model = load_r3d_model(model_path, device)

    # Load tubelets
    tubelets, df_tubelets = load_tubelets_from_csv(csv_path, tubelets_dir, max_tubelets)

    # Get model predictions
    predictions = predict_tubelets_batch(model, tubelets, batch_size, device)

    # Build frame mapping
    frame_to_tubelets = build_frame_to_tubelets_mapping(df_tubelets, T=3)

    # Load ground truth
    ground_truth = load_ground_truth(annotations_dir)

    # Test all combinations
    results = []

    for voting_method in voting_methods:
        for conf_thresh in confidence_thresholds:
            print(f"\n[EVAL] Testing {voting_method} with threshold {conf_thresh}")

            # Convert tubelet predictions to frame detections
            frame_detections = predict_frame_detections(
                frame_to_tubelets, predictions, voting_method, conf_thresh, T=3
            )

            # Calculate metrics
            metrics, detailed = calculate_frame_metrics(frame_detections, ground_truth)

            result = {
                'voting_method': voting_method,
                'confidence_threshold': conf_thresh,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'total_tp': metrics['total_tp'],
                'total_fp': metrics['total_fp'],
                'total_fn': metrics['total_fn']
            }

            results.append(result)

            print(f"  Results: P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | F1={metrics['f1_score']:.3f}")

    # Show results summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame(results)

    # Sort by F1 score
    results_df = results_df.sort_values('f1_score', ascending=False)

    print(f"{'Method':<15} {'Threshold':<9} {'Precision':<9} {'Recall':<9} {'F1':<9}")
    print(f"{'-' * 60}")

    for _, row in results_df.head(10).iterrows():
        print(f"{row['voting_method']:<15} {row['confidence_threshold']:<9.1f} "
              f"{row['precision']:<9.3f} {row['recall']:<9.3f} {row['f1_score']:<9.3f}")

    # Best result
    best = results_df.iloc[0]
    print(f"\n🏆 BEST RESULT:")
    print(f"   Method: {best['voting_method']}")
    print(f"   Threshold: {best['confidence_threshold']}")
    print(f"   F1-Score: {best['f1_score']:.3f}")
    print(f"   Precision: {best['precision']:.3f}")
    print(f"   Recall: {best['recall']:.3f}")

    return results_df


if __name__ == "__main__":

    # Configuration
    model_path = "models/r3d18_static_t3_20250828_112848/best.pth"
    csv_path = "data/tubelets/test_index.csv"
    tubelets_dir = "data/tubelets/test/"
    annotations_dir = "data/annotations/"

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")

    try:
        # Run evaluation
        results = evaluate_model_with_voting(
            model_path=model_path,
            csv_path=csv_path,
            tubelets_dir=tubelets_dir,
            annotations_dir=annotations_dir,
            voting_methods=['center_only', 'majority', 'weighted_average'],
            confidence_thresholds=[0.3, 0.5, 0.7],
            batch_size=16,
            device=device,
            max_tubelets=5000  # Limit for testing - remove for full evaluation
        )

        # Save results
        results.to_csv("evaluation_results.csv", index=False)
        print(f"\n✅ Evaluation complete! Results saved to evaluation_results.csv")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()