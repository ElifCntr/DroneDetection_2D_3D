# src/detection/preprocess/contour.py

import cv2 as cv
import numpy as np


class Contour:
    """
    Config keys (in experiment.yaml):
      preprocess:
        contour:
          min_area: 10       # ignore blobs smaller than this

    Usage:
      det = ContourDetector(cfg)
      bboxes = det.apply(binary_mask)
      # bboxes is a list of (x, y, w, h) tuples
    """

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]["contour"]
        self._min_area = p.get("min_area", 10)

    def apply(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        mask: 2D uint8, values {0, 255}

        Returns:
          List of bounding boxes (x, y, w, h) for each contour whose area
          is at least min_area.
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self._min_area:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            bboxes.append((x, y, w, h))
        return bboxes
