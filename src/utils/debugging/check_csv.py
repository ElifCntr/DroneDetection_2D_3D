import pandas as pd
from pathlib import Path
import os

# Your paths
csv_path = r"/data/tubelets/test_index_corrected.csv"
tubelets_dir = r"/data/tubelets/test"

# Load CSV
df = pd.read_csv(csv_path)

# Check first few rows
print("First 3 CSV entries:")
for i in range(3):
    filename = df.iloc[i]['filename']
    base_filename = os.path.basename(filename)
    expected_path = Path(tubelets_dir) / base_filename

    print(f"  CSV filename: {filename}")
    print(f"  Basename: {base_filename}")
    print(f"  Expected path: {expected_path}")
    print(f"  Exists: {expected_path.exists()}")
    print()

# Check what files actually exist in the directory
print(f"Files in {tubelets_dir}:")
if os.path.exists(tubelets_dir):
    files = list(Path(tubelets_dir).glob("*.npy"))[:5]
    for f in files:
        print(f"  {f}")
else:
    print("  Directory doesn't exist!")