"""
Diagnostic script to check your tubelet data structure.
This will show us exactly what you have so we can load it correctly.

Run from project root:
    python check_tubelet_structure.py
"""

import os
import pandas as pd
from pathlib import Path

print("=" * 70)
print("TUBELET STRUCTURE DIAGNOSTIC")
print("=" * 70)

# ============================================================================
# CHECK DIRECTORY STRUCTURE
# ============================================================================
print("\n[1] Checking directory structure...")

tubelet_dir = Path('data/tubelets')
if not tubelet_dir.exists():
    print(f"❌ Tubelets directory not found: {tubelet_dir}")
    exit(1)

print(f"✅ Found tubelets directory: {tubelet_dir}")

# Check subdirectories
for subdir in ['train', 'val', 'test']:
    subdir_path = tubelet_dir / subdir
    if subdir_path.exists():
        npy_files = list(subdir_path.glob('**/*.npy'))
        print(f"  ✅ {subdir}/: {len(npy_files)} .npy files")

        # Show first few files
        if npy_files:
            print(f"     Example files:")
            for f in npy_files[:3]:
                print(f"       - {f.relative_to(tubelet_dir)}")
    else:
        print(f"  ❌ {subdir}/: Not found")

# ============================================================================
# CHECK CSV FILES
# ============================================================================
print("\n[2] Checking CSV index files...")

csv_files = {
    'train': tubelet_dir / 'train_index.csv',
    'val': tubelet_dir / 'val_index.csv',
    'test': tubelet_dir / 'test_index.csv',
}

csv_info = {}
for split, csv_path in csv_files.items():
    if csv_path.exists():
        print(f"\n  ✅ {split}_index.csv exists")

        # Read CSV
        try:
            df = pd.read_csv(csv_path)
            csv_info[split] = df

            print(f"     - Rows: {len(df)}")
            print(f"     - Columns: {list(df.columns)}")

            # Show first few rows
            print(f"     - First 3 rows:")
            print(df.head(3).to_string(index=False))

            # Check label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                print(f"     - Label distribution:")
                for label, count in label_counts.items():
                    print(f"       Label {label}: {count} ({100 * count / len(df):.1f}%)")

        except Exception as e:
            print(f"     ❌ Error reading CSV: {e}")
    else:
        print(f"  ❌ {split}_index.csv not found")

# ============================================================================
# VERIFY PATHS IN CSV MATCH FILES
# ============================================================================
print("\n[3] Verifying CSV paths match actual files...")

if 'train' in csv_info:
    df = csv_info['train']

    # Check what column has the paths
    path_column = None
    for col in ['tubelet_path', 'image_path', 'path', 'file_path', 'filepath']:
        if col in df.columns:
            path_column = col
            break

    if path_column:
        print(f"  ℹ️  Path column found: '{path_column}'")

        # Check if first few paths exist
        print(f"  📁 Checking if paths exist:")
        for i in range(min(5, len(df))):
            rel_path = df.iloc[i][path_column]

            # Try different base paths
            possible_paths = [
                tubelet_dir / rel_path,
                Path(rel_path),
                Path('data') / rel_path,
            ]

            found = False
            for full_path in possible_paths:
                if full_path.exists():
                    print(f"     ✅ {rel_path}")
                    print(f"        -> {full_path}")
                    found = True
                    break

            if not found:
                print(f"     ❌ {rel_path}")
                print(f"        Tried: {[str(p) for p in possible_paths]}")

            if i == 0 and found:
                # Load one tubelet to show shape
                import numpy as np

                tubelet = np.load(full_path)
                print(f"     📊 Tubelet shape: {tubelet.shape}")
                print(f"     📊 Data type: {tubelet.dtype}")
    else:
        print(f"  ⚠️  Could not find path column in CSV")
        print(f"     Available columns: {list(df.columns)}")

# ============================================================================
# DATASET API GUESS
# ============================================================================
print("\n[4] Suggested dataset loading code:")

if csv_info and path_column:
    print(f"""
Based on your structure, try loading like this:

from datasets import create_dataset

# Check what parameters your create_dataset accepts
import inspect
print(inspect.signature(create_dataset))

# Then try something like:
dataset = create_dataset(
    dataset_type='tubelet',
    # Try these different argument names:
    csv_path='data/tubelets/train_index.csv',  # or
    csv_file='data/tubelets/train_index.csv',  # or
    index_file='data/tubelets/train_index.csv',

    tubelets_root='data/tubelets',  # or
    data_root='data/tubelets',      # or
    root_dir='data/tubelets',
)
""")
else:
    print("  ⚠️  Need more information about your CSV structure")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
print("\nNext step: Share the output above so I can create the correct")
print("data loading script based on your actual structure!")
print("=" * 70)