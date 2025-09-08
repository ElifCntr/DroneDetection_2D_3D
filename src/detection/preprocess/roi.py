# src/detection/preprocess/roi.py

import os
import cv2 as cv
import numpy as np


def adjust_bounding_boxes_to_square(
    boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int
) -> list[tuple[int, int, int, int]]:
    """
    For each box (x, y, w, h), return a square (xs, ys, ss, ss) where
    ss = max(w, h), centered on the original box, clamped to image bounds.
    """
    square_boxes = []
    for (x, y, w, h) in boxes:
        side = max(w, h)
        cx = x + w // 2
        cy = y + h // 2

        xs = cx - side // 2
        ys = cy - side // 2

        xs = max(xs, 0)
        ys = max(ys, 0)
        if xs + side > img_w:
            xs = img_w - side
        if ys + side > img_h:
            ys = img_h - side

        if xs < 0 or ys < 0 or side <= 0:
            continue

        square_boxes.append((int(xs), int(ys), int(side), int(side)))
    return square_boxes


class ROI:
    """
    Crop and save square ROIs from a color frame based on bounding boxes.

    Reads its parameters from cfg["preprocess"]["roi"]:
      pad: int           # pixels of padding around each square box
      out_size: [H, W]   # final square size (height, width)

    Also expects cfg["paths"]["roi_output_dir"] and cfg["paths"]["base_name"]
    to know where to save files.
    """

    def __init__(self, cfg: dict):
        p = cfg["preprocess"]["roi"]
        self._pad = p.get("pad", 5)
        self._out_h, self._out_w = p.get("out_size", [100, 100])
        self._save_roi = p.get("save_roi", False)

        # Read save paths from cfg
        paths = cfg.get("paths", {})
        self._save_dir = paths.get("roi_output_dir", "output/rois")
        self._base_name = paths.get("base_name", "video")

        # create output directory only if we plan to save ROIs
        if self._save_roi:
            os.makedirs(self._save_dir, exist_ok=True)

    def save_rois(
        self,
        frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
        frame_idx: int
    ) -> None:
        """
        Parameters
        ----------
        frame : np.ndarray (H×W×3, BGR)
            Original color frame.
        bboxes : list of (x, y, w, h)
            Rectangular boxes from contour/merge.
        frame_idx : int
            Frame index (used in filename).

        Uses:
          self._pad, self._out_h, self._out_w
          self._save_dir, self._base_name
        """

        # if saving is turned off, do nothing
        if not self._save_roi:
            return

        H, W = frame.shape[:2]

        # 1) Convert all rectangles into squares
        square_boxes = adjust_bounding_boxes_to_square(bboxes, W, H)

        # 2) Iterate, crop, resize, and save
        for count, (x, y, w, h) in enumerate(square_boxes):
            if w <= 0 or h <= 0:
                continue

            # Apply padding and clamp
            x1 = max(x - self._pad, 0)
            y1 = max(y - self._pad, 0)
            x2 = min(x + w + self._pad, W - 1)
            y2 = min(y + h + self._pad, H - 1)

            crop = frame[y1 : y2 + 1, x1 : x2 + 1]
            resized = cv.resize(crop, (self._out_w, self._out_h))

            fname = f"{self._base_name}_frame{frame_idx}_{count}.png"
            save_path = os.path.join(self._save_dir, fname)
            cv.imwrite(save_path, resized)
