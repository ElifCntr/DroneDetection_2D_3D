# src/detection/utils/metrics.py
"""
Evaluation metrics for object detection tasks.
- iou: IoU for two boxes (x,y,w,h)
- match_detections: one-to-one greedy matching by IoU
- precision / recall / f1_score: computed from one-to-one matches
"""
from typing import List, Tuple, Union

Num = Union[int, float]
Box = Tuple[Num, Num, Num, Num]  # (x, y, width, height)


def iou(boxA: Box, boxB: Box) -> float:
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB
    xA2, yA2 = xA + wA, yA + hA
    xB2, yB2 = xB + wB, yB + hB

    xi1 = max(xA, xB)
    yi1 = max(yA, yB)
    xi2 = min(xA2, xB2)
    yi2 = min(yA2, yB2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    areaA = max(0, wA) * max(0, hA)
    areaB = max(0, wB) * max(0, hB)
    union = areaA + areaB - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def match_detections(preds: List[Box], gts: List[Box], iou_thresh: float = 0.5):
    """
    One-to-one greedy matching (each pred and each GT used at most once).
    Returns:
      matches: list of (pred_idx, gt_idx, iou_val)
      tp: int, fp: int, fn: int
    """
    if not preds and not gts:
        return [], 0, 0, 0
    if not preds:
        return [], 0, 0, len(gts)
    if not gts:
        return [], 0, len(preds), 0

    # Build all candidate pairs with IoU >= thresh
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            i = iou(p, g)
            if i >= iou_thresh:
                pairs.append((i, pi, gi))

    # Sort by IoU descending (greedy best-first)
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_pred = set()
    used_gt = set()
    matches = []

    for iou_val, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, float(iou_val)))

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return matches, tp, fp, fn


def precision(preds: List[Box], gts: List[Box], iou_thresh: float = 0.5) -> float:
    _, tp, fp, _ = match_detections(preds, gts, iou_thresh)
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(preds: List[Box], gts: List[Box], iou_thresh: float = 0.5) -> float:
    _, tp, _, fn = match_detections(preds, gts, iou_thresh)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(preds: List[Box], gts: List[Box], iou_thresh: float = 0.5) -> float:
    p = precision(preds, gts, iou_thresh)
    r = recall(preds, gts, iou_thresh)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
