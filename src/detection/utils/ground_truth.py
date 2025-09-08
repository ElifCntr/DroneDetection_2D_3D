# src/detection/utils/ground_truth.py
"""
Utility function to load ground-truth annotations for video frames.
Supports two formats:
  - CSV with columns: frame,count,x,y,width,height [,label]
    where count=0 → no objects; count=N → N objects follow (repeated groups of x,y,width,height,label)
  - TXT where each line is:
      <frame> 0            # no objects
      <frame> N <x1> <y1> <w1> <h1> [label] ... <xN> <yN> <wN> <hN> [label]

Ground-truth file path is supplied externally (e.g., from your config).
"""
from pathlib import Path
from typing import Dict, List, Tuple


def load_ground_truth(file_path: str) -> Dict[int, List[Tuple[int, int, int, int]]]:
    """
    Load ground-truth boxes from a CSV or TXT file.

    :param file_path: Path to the annotation file (with or without .csv/.txt extension).
    :return: Mapping from frame index to a list of (x, y, width, height) boxes.
    """
    p = Path(file_path)
    # if no extension given, try .csv then .txt
    if p.suffix == "":
        if p.with_suffix('.csv').exists():
            p = p.with_suffix('.csv')
        elif p.with_suffix('.txt').exists():
            p = p.with_suffix('.txt')
    if not p.exists():
        raise FileNotFoundError(f"Annotation file not found: {file_path}")

    gt: Dict[int, List[Tuple[int, int, int, int]]] = {}
    # CSV format
    if p.suffix.lower() == ".csv":
        import csv
        with p.open(newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                frame = int(row[0])
                count = int(row[1])
                if count <= 0:
                    continue
                # for each object, parse x,y,width,height
                for i in range(count):
                    base = 2 + i * 4
                    if len(row) >= base + 4:
                        x = int(row[base])
                        y = int(row[base + 1])
                        w = int(row[base + 2])
                        h = int(row[base + 3])
                        gt.setdefault(frame, []).append((x, y, w, h))
    # TXT format
    elif p.suffix.lower() == ".txt":
        with p.open() as f:
            for ln_no, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    frame = int(parts[0])
                    count = int(parts[1])
                except ValueError as e:
                    # bad header; skip line
                    # print(f"[WARN] {p.name}:{ln_no}: bad header -> {e}")
                    continue

                if count <= 0:
                    # optional: still create empty list so downstream .get(frame, []) works
                    gt.setdefault(frame, [])
                    continue

                remaining = parts[2:]

                # Boxes can be 4-tuple (x y w h) or 5-tuple (x y w h label)
                # Prefer stride=5 if we have enough tokens; otherwise fall back to 4.
                if len(remaining) >= 5 * count:
                    stride = 5
                elif len(remaining) >= 4 * count:
                    stride = 4
                else:
                    # Not enough tokens for declared count; best-effort parse with stride=4
                    stride = 4
                    # print(f"[WARN] {p.name}:{ln_no}: tokens fewer than declared count={count}")

                # Parse EXACTLY `count` boxes, ignore any extras after that
                parsed = 0
                for i in range(count):
                    base = i * stride
                    if len(remaining) < base + 4:
                        break  # not enough tokens left
                    try:
                        x = int(remaining[base + 0])
                        y = int(remaining[base + 1])
                        w = int(remaining[base + 2])
                        h = int(remaining[base + 3])
                    except ValueError:
                        # print(f"[WARN] {p.name}:{ln_no}: non-integer coords at box {i}")
                        continue
                    gt.setdefault(frame, []).append((x, y, w, h))
                    parsed += 1
    else:
        raise ValueError(f"Unsupported annotation format: {p.suffix}")

    return gt
