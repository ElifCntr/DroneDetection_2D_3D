# src/detection/preprocess/threshold.py

import cv2 as cv
import numpy as np


class Threshold:
    """
    Config keys (in experiment.yaml):
      preprocess:
        threshold:
          type: "fixed"       # "fixed" or "otsu"
          fixed_value: 200    # only used if type=="fixed"
          shadow_value: 127   # treat this value as background

    Usage:
      thr = Threshold(cfg)
      binary = thr.apply(soft_mask)
    """

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]["threshold"]
        self._mode = p.get("type", "fixed").lower()
        self._fixed = p.get("fixed_value", 200)
        self._shadow_val = p.get("shadow_value", 127)

    def apply(self, soft_mask: np.ndarray) -> np.ndarray:
        # 1) Drop shadows
        if self._shadow_val is not None:
            soft_mask = soft_mask.copy()
            soft_mask[soft_mask == self._shadow_val] = 0

        # 2) Threshold to binary
        if self._mode == "otsu":
            _, binary = cv.threshold(
                soft_mask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
            )
        else:
            _, binary = cv.threshold(
                soft_mask, self._fixed, 255, cv.THRESH_BINARY
            )
        return binary
