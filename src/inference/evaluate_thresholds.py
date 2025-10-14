import pandas as pd
import sys

sys.path.append(r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection")

from utils.data_processing.frame_metrics import FrameLevelMetrics

# Load the CSV and skip the header row (row 0)
df = pd.read_csv(
    r"/inference/evaluation_results\tubelet_inference_results.csv",
    header=0)  # This will use the first row as column names

print("Column names from CSV:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

# Make sure we have numeric data
df['drone_confidence'] = pd.to_numeric(df['drone_confidence'], errors='coerce')
df['gt_label'] = pd.to_numeric(df['gt_label'], errors='coerce')

# Remove any rows with NaN values
df = df.dropna()

print(f"\nAfter cleaning: {len(df)} rows")
print("Testing different confidence thresholds on existing predictions...")

for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    # Apply threshold to drone confidences
    new_predictions = (df['drone_confidence'] >= threshold).astype(int)

    # Calculate metrics
    metrics = FrameLevelMetrics.calculate_all_metrics(df['gt_label'], new_predictions)

    print(
        f"Threshold {threshold}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")