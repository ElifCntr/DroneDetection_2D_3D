# src/detection/preprocess/merge_contours.py

import numpy as np
from detection.utils import create as ut_create


class Merge_contours:
    """
    Merges overlapping/nearby bounding boxes based on both distance and IoU.

    Config keys (in experiment.yaml):
      preprocess:
        merge_contours:
          iou_thresh: 0.3      # merge boxes if IoU > this
          distance_thresh: 8   # merge boxes if distance < this (pixels)

    Usage:
      merger = MergeContours(cfg)
      merged_bboxes = merger.apply(raw_bboxes)
      # raw_bboxes is a list of (x, y, w, h)
    """

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]["merge_contours"]
        self._iou_thresh = p.get("iou_thresh", 0.3)
        self._distance_thresh = p.get("distance_thresh", 8)

    def apply(self, bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """
        Merge boxes using both distance and IoU criteria
        First merges touching/close boxes, then merges overlapping boxes
        """
        if not bboxes:
            return []

        # First pass: merge touching/close boxes by distance
        distance_merged = self.merge_touching_contours(bboxes)

        # Second pass: merge overlapping boxes using IoU
        final_merged = self.merge_overlapping_contours(distance_merged)

        return final_merged

    def merge_touching_contours(self, bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Merge bounding boxes that are within distance_threshold pixels of each other"""
        if not bboxes:
            return []

        # Convert to [x1, y1, x2, y2] format for easier calculation
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bboxes], dtype=float)
        used = np.zeros(len(boxes), dtype=bool)
        merged = []

        for i in range(len(boxes)):
            if used[i]:
                continue
            base = boxes[i].copy()
            used[i] = True

            # Keep merging until no more close boxes found
            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue

                    # Check if boxes are within distance threshold
                    if self._boxes_are_close(base, boxes[j], self._distance_thresh):
                        # Merge boxes by expanding base to cover both
                        base[0] = min(base[0], boxes[j][0])  # min x
                        base[1] = min(base[1], boxes[j][1])  # min y
                        base[2] = max(base[2], boxes[j][2])  # max x
                        base[3] = max(base[3], boxes[j][3])  # max y
                        used[j] = True
                        changed = True  # Continue looking for more boxes to merge

            merged.append(base)

        # Convert back to (x, y, w, h)
        result = []
        for x1, y1, x2, y2 in merged:
            w = x2 - x1
            h = y2 - y1
            result.append((int(x1), int(y1), int(w), int(h)))

        return result

    def merge_overlapping_contours(self, bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Merge overlapping boxes based on IoU threshold"""
        if not bboxes:
            return []

        # Convert to [x1, y1, x2, y2] format
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bboxes], dtype=float)
        used = np.zeros(len(boxes), dtype=bool)
        merged = []

        metrics = ut_create("metrics")

        for i in range(len(boxes)):
            if used[i]:
                continue
            base = boxes[i].copy()
            used[i] = True

            # Keep merging until no more overlapping boxes found
            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    if metrics.iou(base, boxes[j]) > self._iou_thresh:
                        # Expand base to cover both boxes
                        base[0] = min(base[0], boxes[j][0])
                        base[1] = min(base[1], boxes[j][1])
                        base[2] = max(base[2], boxes[j][2])
                        base[3] = max(base[3], boxes[j][3])
                        used[j] = True
                        changed = True  # Continue looking for more boxes to merge

            merged.append(base)

        # Convert back to (x, y, w, h)
        result = []
        for x1, y1, x2, y2 in merged:
            w = x2 - x1
            h = y2 - y1
            result.append((int(x1), int(y1), int(w), int(h)))

        return result

    def _boxes_are_close(self, box1: np.ndarray, box2: np.ndarray, threshold: float) -> bool:
        """
        Check if two boxes are within threshold distance of each other
        Uses minimum distance between box edges
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate minimum distance between box edges
        # If boxes overlap, distance will be 0
        dx = max(0, max(x1_min - x2_max, x2_min - x1_max))
        dy = max(0, max(y1_min - y2_max, y2_min - y1_max))
        distance = np.sqrt(dx * dx + dy * dy)

        return distance <= threshold

    def _calculate_center_distance(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate distance between box centers (alternative distance metric)"""
        x1_center = (box1[0] + box1[2]) / 2
        y1_center = (box1[1] + box1[3]) / 2
        x2_center = (box2[0] + box2[2]) / 2
        y2_center = (box2[1] + box2[3]) / 2

        dx = x1_center - x2_center
        dy = y1_center - y2_center
        return np.sqrt(dx * dx + dy * dy)