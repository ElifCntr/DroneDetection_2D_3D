"""
ViBe background-subtraction wrapper.
Usage:
    from detection.background.vibe import VIBE
    bg = VIBE(cfg)           # cfg is the full config dict
    mask = bg.apply(frame)   # returns a binary mask
"""

import numpy as np
import cv2 as cv


class VIBE:
    def __init__(self, cfg: dict):
        """
        cfg must contain:
            cfg["background"]["vibe"] = { ... }
        Example keys: num_samples, min_matches, radius, subsample_factor
        """

        param = cfg["background"].get("vibe", {})

        # Parameters (falling back to defaults if not provided)
        self.num_samples      = param.get("num_samples", 20)
        self.min_matches      = param.get("min_matches", 2)
        self.radius           = param.get("radius", 20)
        self.subsample_factor = param.get("subsample_factor", 16)

        self.fg_model = None
        self.samples = None
        self.height = None
        self.width = None

    def _initialize(self, frame: np.ndarray):
        """Initializes the sample set for the first frame."""
        self.height, self.width = frame.shape[:2]
        self.samples = np.zeros(
            (self.num_samples, self.height, self.width), dtype=np.uint8
        )
        for i in range(self.num_samples):
            self.samples[i] = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Feed one BGR frame → get binary mask (values 0 / 255).
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.samples is None:
            self._initialize(frame)
            return np.zeros_like(gray)

        # Compute distance to all samples
        dist = np.abs(self.samples.astype(np.int16) - gray.astype(np.int16))
        matches = (dist < self.radius).sum(axis=0)

        # Foreground mask
        fg_mask = (matches < self.min_matches).astype(np.uint8) * 255

        # Update model with probability 1 / subsample_factor
        update_mask = (np.random.randint(0, self.subsample_factor, size=gray.shape) == 0)
        for i in range(self.num_samples):
            self.samples[i][update_mask] = gray[update_mask]

        return fg_mask

    def reset(self):
        """Reset the model (start fresh on next frame)."""
        self.samples = None
