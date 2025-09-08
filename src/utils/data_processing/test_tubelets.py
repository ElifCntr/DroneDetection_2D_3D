# test_tubelets.py
"""
Simple script to verify tubelet data is properly formatted and loadable.
Run this before building the training pipeline.
"""

import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path


def test_tubelets():
    """Test loading and inspect tubelet data."""

    # Running from data/tubelets/ folder, so files are in current directory
    train_csv = "train_index.csv"
    val_csv = "val_index.csv"
    tubelets_root = "."

    print("=== Tubelet Data Inspection ===\n")

    # 1. Check if files exist
    print("1. Checking file existence:")
    print(f"   Train CSV: {os.path.exists(train_csv)}")
    print(f"   Val CSV: {os.path.exists(val_csv)}")
    print(f"   Tubelets dir: {os.path.exists(tubelets_root)}")

    if not os.path.exists(train_csv):
        print(f"ERROR: {train_csv} not found!")
        return

    # 2. Inspect CSV structure
    print("\n2. CSV Structure:")
    try:
        with open(train_csv, 'r') as f:
            lines = f.readlines()[:10]  # First 10 lines
        print(f"   Total lines in train CSV: {len(open(train_csv).readlines())}")
        print("   First few lines:")
        for i, line in enumerate(lines):
            print(f"   {i}: {line.strip()}")

    except Exception as e:
        print(f"   ERROR reading CSV: {e}")
        return

    # 3. Parse CSV and check samples
    print("\n3. Sample Analysis:")
    try:
        samples = []
        with open(train_csv, 'r') as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    samples.append((parts[0], int(parts[1])))

        print(f"   Total samples: {len(samples)}")

        # Count labels
        labels = [label for _, label in samples]
        unique_labels = set(labels)
        print(f"   Unique labels: {unique_labels}")
        for label in unique_labels:
            count = labels.count(label)
            print(f"   Label {label}: {count} samples")

    except Exception as e:
        print(f"   ERROR parsing CSV: {e}")
        return

    # 4. Test loading actual tubelet files
    print("\n4. Loading Sample Tubelets:")
    test_samples = samples[:5]  # Test first 5

    for i, (rel_path, label) in enumerate(test_samples):
        # Strip "data/tubelets/" prefix if it exists in the CSV path
        if rel_path.startswith("data/tubelets/"):
            clean_path = rel_path[len("data/tubelets/"):]
        else:
            clean_path = rel_path
        tubelet_path = os.path.join(tubelets_root, clean_path)
        print(f"\n   Sample {i + 1}: {rel_path} (label: {label})")
        print(f"   Clean path: {clean_path}")
        print(f"   Full path: {tubelet_path}")
        print(f"   File exists: {os.path.exists(tubelet_path)}")

        if os.path.exists(tubelet_path):
            try:
                # Load tubelet
                tubelet = np.load(tubelet_path)
                print(f"   Shape: {tubelet.shape}")
                print(f"   Dtype: {tubelet.dtype}")
                print(f"   Value range: [{tubelet.min():.2f}, {tubelet.max():.2f}]")

                # Expected format check
                if len(tubelet.shape) == 4:  # T,H,W,C
                    T, H, W, C = tubelet.shape
                    print(f"   Format: T={T}, H={H}, W={W}, C={C}")
                    if C == 3:
                        print("   ✓ Looks like BGR/RGB format")
                    else:
                        print(f"   ⚠ Unexpected channels: {C}")
                else:
                    print(f"   ⚠ Unexpected dimensions: {len(tubelet.shape)}")

            except Exception as e:
                print(f"   ERROR loading tubelet: {e}")
        else:
            print("   ⚠ File not found")

    # 5. Visualize one tubelet (optional)
    print("\n5. Visualization Test:")
    if len(test_samples) > 0:
        rel_path, label = test_samples[0]
        tubelet_path = os.path.join(tubelets_root, rel_path)

        if os.path.exists(tubelet_path):
            try:
                tubelet = np.load(tubelet_path)
                if len(tubelet.shape) == 4 and tubelet.shape[3] == 3:
                    print("   Showing first frame of first tubelet...")
                    first_frame = tubelet[0]  # Shape: H,W,C

                    # Convert to uint8 if needed
                    if first_frame.dtype != np.uint8:
                        if first_frame.max() <= 1.0:
                            first_frame = (first_frame * 255).astype(np.uint8)
                        else:
                            first_frame = first_frame.astype(np.uint8)

                    cv2.imshow("Sample Tubelet - Frame 0", first_frame)
                    print("   Press any key to close the image window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print("   ✓ Visualization successful")
                else:
                    print("   ⚠ Cannot visualize - unexpected format")
            except Exception as e:
                print(f"   ERROR in visualization: {e}")

    print("\n=== Inspection Complete ===")
    print("If all checks passed, your tubelet data is ready for training!")


if __name__ == "__main__":
    test_tubelets()