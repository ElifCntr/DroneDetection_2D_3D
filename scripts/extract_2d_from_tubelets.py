# scripts/extract_2d_from_tubelets.py
"""
Quick script to extract 2D center frames from existing 3D tubelets.
This ensures ResNet-18 uses the exact same ROI regions as R3D-18.
"""

import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import argparse


def extract_center_frame_from_tubelet(tubelet_path):
    """Extract center frame from 3D tubelet."""
    tubelet = np.load(tubelet_path)  # Shape: (T, H, W, C)
    T = tubelet.shape[0]
    center_idx = T // 2
    center_frame = tubelet[center_idx]  # Shape: (H, W, C)
    return center_frame


def process_tubelet_csv(csv_path, tubelets_root, output_dir):
    """Process one CSV file to extract 2D regions."""

    print(f"Processing {csv_path}...")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} tubelets")

    # Create output directories
    pos_dir = os.path.join(output_dir, "positive")
    neg_dir = os.path.join(output_dir, "negative")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    samples_2d = []

    for idx, row in df.iterrows():
        # Get tubelet path
        rel_path = row['path'] if 'path' in row else row.iloc[0]

        # Handle different CSV formats - fix backslashes
        if rel_path.startswith("data/tubelets/"):
            clean_path = rel_path[len("data/tubelets/"):]
        elif rel_path.startswith("data\\tubelets\\"):
            clean_path = rel_path[len("data\\tubelets\\"):]
        else:
            clean_path = rel_path

        tubelet_path = os.path.join(tubelets_root, clean_path)
        label = int(row['label'] if 'label' in row else row.iloc[1])

        # Debug print
        if idx < 5:  # Print first 5 for debugging
            print(f"Debug {idx}: rel_path={rel_path}")
            print(f"Debug {idx}: clean_path={clean_path}")
            print(f"Debug {idx}: tubelet_path={tubelet_path}")
            print(f"Debug {idx}: exists={os.path.exists(tubelet_path)}")

        if not os.path.exists(tubelet_path):
            continue

        try:
            # Extract center frame
            center_frame = extract_center_frame_from_tubelet(tubelet_path)

            # Create filename
            base_name = Path(tubelet_path).stem
            img_filename = f"{base_name}.jpg"

            # Save to appropriate directory
            if label == 1:
                img_path = os.path.join(pos_dir, img_filename)
            else:
                img_path = os.path.join(neg_dir, img_filename)

            # Save as image (convert RGB to BGR for OpenCV)
            cv2.imwrite(img_path, cv2.cvtColor(center_frame, cv2.COLOR_RGB2BGR))

            # Record sample
            samples_2d.append({
                'image_path': img_path,
                'label': label,
                'original_tubelet': tubelet_path
            })

        except Exception as e:
            print(f"Error processing {tubelet_path}: {e}")
            continue

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    # Save 2D index CSV
    df_2d = pd.DataFrame(samples_2d)
    split_name = Path(csv_path).stem.split('_')[0]  # Extract 'train', 'val', 'test'
    csv_out = os.path.join(output_dir, f"{split_name}_2d_index.csv")
    df_2d.to_csv(csv_out, index=False)

    print(f"Completed {split_name}: {len(samples_2d)} regions extracted")
    print(f"  Positive: {sum(df_2d['label'] == 1)}")
    print(f"  Negative: {sum(df_2d['label'] == 0)}")
    print(f"  Index saved: {csv_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tubelets_root', default='D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data/tubelets', help='Root tubelet directory')
    parser.add_argument('--output_root', default='D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data/2d_regions', help='Output directory')
    args = parser.parse_args()

    # Process all splits
    splits = ['test']

    for split in splits:
        csv_path = os.path.join(args.tubelets_root, f"{split}_index.csv")
        if os.path.exists(csv_path):
            output_dir = os.path.join(args.output_root, split)
            process_tubelet_csv(csv_path, args.tubelets_root, output_dir)
        else:
            print(f"Warning: {csv_path} not found, skipping {split}")

    print(f"\nAll 2D regions extracted to: {args.output_root}")


if __name__ == "__main__":
    main()