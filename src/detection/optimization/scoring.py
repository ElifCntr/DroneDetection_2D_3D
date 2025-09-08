# src/detection/optimization/scoring.py

"""
Optimization-scoring module.

Defines a loss function for grid search that balances detection rate against
average proposal count.

Loss formula:
    L = -alpha * det_rate + beta * avg_props

Where:
  - det_rate  : fraction of ground-truth frames correctly detected (IoU ≥ threshold)
  - avg_props : average number of proposals per frame
  - alpha     : weight for detection rate (higher => prioritize detection)
  - beta      : weight for proposal count (higher => penalize more proposals)
"""

from typing import Dict, List, Tuple
from detection.utils.metrics import iou

Box = Tuple[int, int, int, int]


def loss(
    proposals_by_frame: Dict[int, List[Box]],
    gts_by_frame:       Dict[int, List[Box]],
    alpha:     float = 1.0,
    beta:      float = 1.0,
    iou_thresh: float = 0.5
) -> float:
    """
    Compute a combined loss for a set of proposals against ground-truth boxes.

    Args:
        proposals_by_frame: dict mapping frame index to list of proposed boxes (x, y, w, h)
        gts_by_frame      : dict mapping frame index to list of ground-truth boxes (x, y, w, h)
        alpha             : weight for detection-rate term
        beta              : weight for average-proposals term
        iou_thresh        : IoU threshold to consider a GT box as detected

    Returns:
        Loss value (lower is better).
    """
    # Total frames processed
    total_frames = len(proposals_by_frame)
    if total_frames == 0:
        return 0.0

    # 1) Average proposals per frame
    total_props = sum(len(props) for props in proposals_by_frame.values())
    avg_props = total_props / total_frames

    # 2) Detection rate (only over frames that have GT boxes)
    gt_frames = [f for f, boxes in gts_by_frame.items() if boxes]
    if not gt_frames:
        det_rate = 1.0  # no ground-truth to miss
    else:
        hits = 0
        for f in gt_frames:
            props = proposals_by_frame.get(f, [])
            # Check if at least one proposal overlaps any GT
            if any(iou(prop, gt) >= iou_thresh for prop in props for gt in gts_by_frame[f]):
                hits += 1
        det_rate = hits / len(gt_frames)

    # Loss: penalize misses and proposal volume
    return -alpha * det_rate + beta * avg_props
