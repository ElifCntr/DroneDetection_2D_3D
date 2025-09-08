# src/detection/background/mog2.py
"""
MOG2 background-subtraction wrapper.
Usage:
    from detection.background.mog2 import MOG2
    bg = MOG2(cfg)          # cfg is the full config dict
    mask = bg.apply(frame)  # returns a binary mask
"""

import cv2 as cv
import numpy as np


class MOG2:
    def __init__(self, cfg: dict):
        """
        cfg: from experiment.yaml, e.g. {"method": "MOG2", "overrides": {...}}
        defaults: from background.yaml, e.g. background["MOG2"]

        Final params = defaults + overrides
        """
        overrides = cfg.get("overrides", {})
        param = {**overrides}  # merge dicts

        self.history = param.get("history", 50)
        self.var_threshold = param.get("var_threshold", 16)
        self.detect_shadows = param.get("detect_shadows", True)
        self.learning_rate = param.get("learning_rate", -1)

        self._build()

    def _build(self):
        self._bs = cv.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Feed one BGR frame → get soft mask (values 0 / 127 / 255)."""
        return self._bs.apply(frame, learningRate=self.learning_rate)


    def reset(self):
        """Start the background model from scratch."""
        self._build()

