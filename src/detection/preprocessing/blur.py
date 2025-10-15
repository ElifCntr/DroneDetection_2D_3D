# src/detection/preprocess/blur.py

import cv2 as cv
import numpy as np


class Blur:
    """
    Config keys:
      preprocess:
        blur:
          method: "gaussian"    # "gaussian" or "median"
          kernel_size: 5        # must be odd (e.g. 3,5,7)

    Usage:
      blur = Blur(cfg)
      smoothed = blur.apply(binary_mask)
    """

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]["blur"]
        k = p.get("kernel_size", 5)
        # enforce odd kernel size
        self._ksize = k if (k % 2 == 1) else (k + 1)
        self._method = p.get("method", "gaussian").lower()

        # Validate method
        if self._method not in ["gaussian", "median"]:
            print(f"[WARN] Unknown blur method '{self._method}', using gaussian")
            self._method = "gaussian"

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: 2D uint8 (0/255)
        returns: blurred mask (still uint8)
        """
        if self._method == "median":
            return cv.medianBlur(mask, self._ksize)
        else:  # gaussian
            return cv.GaussianBlur(mask, (self._ksize, self._ksize), 0)