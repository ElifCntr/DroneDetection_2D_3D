# src/detection/preprocess/morphology.py

import cv2 as cv
import numpy as np


class Morphology:
    """
    Config keys:
      preprocess:
        morphology:
          open_kernel: 3       # odd integer
          close_kernel: 5      # odd integer
          min_area: 10        # remove blobs smaller
          padding_percentage: 0.1  # percentage padding for bounding boxes

    Usage:
      morph = Morphology(cfg)
      cleaned = morph.apply(blurred_mask)
    """

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]["morphology"]
        ok = p.get("open_kernel", 3)
        ck = p.get("close_kernel", 5)
        self._open_kernel = ok if (ok % 2 == 1) else (ok + 1)
        self._close_kernel = ck if (ck % 2 == 1) else (ck + 1)
        self._min_area = p.get("min_area", 10)
        self._padding_percentage = p.get("padding_percentage", 0.1)

    def get_percentage_padding(self, w: int, h: int) -> int:
        """Add padding as percentage of bounding box size"""
        pad_w = int(w * self._padding_percentage)
        pad_h = int(h * self._padding_percentage)
        return max(pad_w, pad_h, 3)  # Minimum 3 pixels

    def apply(self, mask: np.ndarray, frame_shape: tuple = None) -> tuple:
        # 1) Opening (erode → dilate)
        ek = cv.getStructuringElement(
            cv.MORPH_RECT, (self._open_kernel, self._open_kernel)
        )
        opened = cv.morphologyEx(mask, cv.MORPH_OPEN, ek)

        # 2) Closing (dilate → erode)
        ck = cv.getStructuringElement(
            cv.MORPH_RECT, (self._close_kernel, self._close_kernel)
        )
        closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, ck)

        # 3) Connected-components + stats
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(closed)
        filtered = np.zeros_like(closed)
        boxes = []

        # Get frame dimensions for bounds checking
        if frame_shape is not None:
            frame_h, frame_w = frame_shape[:2]
        else:
            frame_h, frame_w = mask.shape[:2]

        print(f"[DEBUG] Found {num_labels - 1} connected components")

        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv.CC_STAT_AREA]
            print(f"[DEBUG] Component {i}: area={area}, min_area={self._min_area}")

            if area >= self._min_area:
                # draw blob into filtered mask
                filtered[labels == i] = 255

                # extract bounding box via stats
                x = stats[i, cv.CC_STAT_LEFT]
                y = stats[i, cv.CC_STAT_TOP]
                w = stats[i, cv.CC_STAT_WIDTH]
                h = stats[i, cv.CC_STAT_HEIGHT]

                #print(f"[DEBUG] Original box: ({x}, {y}, {w}, {h}) area={w * h}")

                # Apply percentage-based padding
                padding = self.get_percentage_padding(w, h)
                #print(f"[DEBUG] Calculated padding: {padding}")

                # Apply padding with bounds checking
                x_pad = max(0, x - padding)
                y_pad = max(0, y - padding)
                w_pad = min(frame_w - x_pad, w + 2 * padding)
                h_pad = min(frame_h - y_pad, h + 2 * padding)

                #print(f"[DEBUG] Padded box: ({x_pad}, {y_pad}, {w_pad}, {h_pad}) area={w_pad * h_pad}")

                # THIS WAS MISSING - ADD THE BOX TO THE LIST!
                boxes.append((x_pad, y_pad, w_pad, h_pad))
            else:
                print(f"[DEBUG] Component {i} rejected (area {area} < {self._min_area})")

        #print(f"[DEBUG] Returning {len(boxes)} boxes")
        return filtered, boxes