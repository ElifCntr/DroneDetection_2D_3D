#!/usr/bin/env python3
"""
Qualitative analysis script for R3D-18 using existing prediction results.
No need to run inference - just analyze the saved results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Configuration
PROJECT_ROOT = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection"
R3D_RESULTS_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\src\inference\evaluation_results\tubelet_inference_results.csv"
TEST_INDEX_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\tubelets\test_index.csv"
OUTPUT_DIR = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\r3d_qualitative_analysis"


def load_r3d_predictions():
    """Load R3D-18 prediction results"""
    print("Loading R3D-18 prediction results...")

    # Check first line to see if it's a header
    with open(R3D_RESULTS_PATH, 'r') as f:
        first_line = f.readline().strip()
        print(f"First line: {first_line}")

    # Load with header if detected
    if 'video_name' in first_line or 'gt_label' in first_line:
        print("Detected header row")
        results_df = pd.read_csv(R3D_RESULTS_PATH, header=0)

        # Rename columns to standard names
        column_mapping = {
            'video_name': 'video_name',
            'center_frame': 'frame_idx',
            'gt_label': 'true_label',
            'prediction': 'predicted_label',
            'drone_confidence': 'confidence'
        }

        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in results_df.columns:
                results_df = results_df.rename(columns={old_name: new_name})

        print(f"Column names after mapping: {list(results_df.columns)}")
    else:
        # No header - use original approach
        results_df = pd.read_csv(R3D_RESULTS_PATH, header=None,
                                 names=['video_name', 'frame_idx', 'true_label', 'predicted_label', 'confidence',
                                        'extra'])

    print(f"Loaded {len(results_df)} R3D predictions")
    print(f"Sample row after loading: {results_df.iloc[0].to_dict()}")

    # Calculate prediction types
    def get_prediction_type(row):
        try:
            true_label = int(row['true_label'])
            pred_label = int(row['predicted_label'])

            if true_label == 1 and pred_label == 1:
                return "TP"
            elif true_label == 0 and pred_label == 0:
                return "TN"
            elif true_label == 0 and pred_label == 1:
                return "FP"
            elif true_label == 1 and pred_label == 0:
                return "FN"
        except (ValueError, TypeError) as e:
            print(f"Error processing row: {row}, error: {e}")
            return "UNKNOWN"

    results_df['prediction_type'] = results_df.apply(get_prediction_type, axis=1)

    # Remove any rows that couldn't be processed
    results_df = results_df[results_df['prediction_type'] != 'UNKNOWN']
    print(f"Successfully processed {len(results_df)} predictions")

    return results_df


def load_test_index():
    """Load test index to get tubelet paths"""
    print("Loading test index...")

    test_df = pd.read_csv(TEST_INDEX_PATH, header=None,
                          names=['tubelet_path', 'label', 'video_name', 'frame_idx', 'conf'])

    print(f"Loaded {len(test_df)} test samples")

    # Fix tubelet paths to be absolute
    def fix_tubelet_path(path):
        if not os.path.isabs(path):
            path = path.replace('\\', '/')
            abs_path = os.path.join(PROJECT_ROOT, path)
            return abs_path
        return path

    test_df['tubelet_path'] = test_df['tubelet_path'].apply(fix_tubelet_path)

    # Create image paths for visualization
    def create_image_path(tubelet_path, label):
        basename = os.path.basename(tubelet_path).replace('.npy', '.jpg')
        subdir = 'positive' if label == 1 else 'negative'
        img_path = os.path.join(PROJECT_ROOT, 'data', '2d_regions', 'test', subdir, basename)
        return img_path

    test_df['image_path'] = test_df.apply(lambda row: create_image_path(row['tubelet_path'], row['label']), axis=1)

    return test_df


def merge_results_with_paths(results_df, test_df):
    """Merge prediction results with file paths"""
    print("Merging prediction results with file paths...")

    # Fix data type mismatches before merging
    print("Fixing data types for merge...")

    # Convert frame_idx to string in both dataframes for consistent merging
    results_df['frame_idx'] = results_df['frame_idx'].astype(str)
    test_df['frame_idx'] = test_df['frame_idx'].astype(str)

    # Also ensure video_name is string in both
    results_df['video_name'] = results_df['video_name'].astype(str)
    test_df['video_name'] = test_df['video_name'].astype(str)

    print(f"Results DF frame_idx type: {results_df['frame_idx'].dtype}")
    print(f"Test DF frame_idx type: {test_df['frame_idx'].dtype}")

    print(f"Sample results data: video={results_df.iloc[0]['video_name']}, frame={results_df.iloc[0]['frame_idx']}")
    print(f"Sample test data: video={test_df.iloc[0]['video_name']}, frame={test_df.iloc[0]['frame_idx']}")

    # Merge on video_name and frame_idx
    merged_df = pd.merge(results_df, test_df,
                         on=['video_name', 'frame_idx'],
                         how='inner',
                         suffixes=('_pred', '_test'))

    print(
        f"Successfully merged {len(merged_df)} samples out of {len(results_df)} predictions and {len(test_df)} test samples")
    print(f"Merged columns: {list(merged_df.columns)}")

    if len(merged_df) == 0:
        print("WARNING: No matches found. Checking for mismatches...")
        print(f"Unique videos in results: {results_df['video_name'].nunique()}")
        print(f"Unique videos in test: {test_df['video_name'].nunique()}")
        print(f"Sample video names from results: {list(results_df['video_name'].unique()[:5])}")
        print(f"Sample video names from test: {list(test_df['video_name'].unique()[:5])}")
        return merged_df

    # Rename columns to remove suffixes for easier access
    column_renames = {}
    for col in merged_df.columns:
        if col.endswith('_test'):
            new_name = col.replace('_test', '')
            if new_name not in merged_df.columns:  # Avoid conflicts
                column_renames[col] = new_name

    merged_df = merged_df.rename(columns=column_renames)
    print(f"Renamed columns: {column_renames}")
    print(f"Final columns: {list(merged_df.columns)}")

    # Use the true label from test data (more reliable)
    if 'label' in merged_df.columns:
        merged_df['true_label'] = merged_df['label']
    elif 'label_test' in merged_df.columns:
        merged_df['true_label'] = merged_df['label_test']

    # Recalculate prediction types with correct labels
    def get_prediction_type(row):
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])

        if true_label == 1 and pred_label == 1:
            return "TP"
        elif true_label == 0 and pred_label == 0:
            return "TN"
        elif true_label == 0 and pred_label == 1:
            return "FP"
        elif true_label == 1 and pred_label == 0:
            return "FN"

    merged_df['prediction_type'] = merged_df.apply(get_prediction_type, axis=1)

    return merged_df


def extract_representative_frames(merged_df, output_dir):
    """Extract representative frames from tubelets"""
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'representative_frames')
    os.makedirs(frames_dir, exist_ok=True)

    print("Extracting representative frames...")

    for idx, row in merged_df.iterrows():
        try:
            tubelet_path = row['tubelet_path']
            pred_type = row['prediction_type']

            if not os.path.exists(tubelet_path):
                continue

            # Load tubelet
            tubelet = np.load(tubelet_path)

            # Get middle frame
            if len(tubelet.shape) == 4:  # (T, H, W, C)
                middle_frame_idx = tubelet.shape[0] // 2
                frame = tubelet[middle_frame_idx]
            elif len(tubelet.shape) == 3:  # (T, H, W)
                middle_frame_idx = tubelet.shape[0] // 2
                frame = tubelet[middle_frame_idx]
                frame = np.stack([frame, frame, frame], axis=-1)  # Convert to RGB
            else:
                continue

            # Ensure proper format
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            # Convert to PIL and save
            if len(frame.shape) == 3:
                pil_image = Image.fromarray(frame)
                filename = f"{pred_type}_{os.path.basename(tubelet_path).replace('.npy', '.png')}"
                save_path = os.path.join(frames_dir, filename)
                pil_image.save(save_path)

                # Add frame path to dataframe
                merged_df.at[idx, 'representative_frame'] = save_path

        except Exception as e:
            if idx < 10:  # Only print first few errors
                print(f"Error extracting frame from {row['tubelet_path']}: {e}")
            continue

    print(f"Extracted frames saved to: {frames_dir}")


def analyze_r3d_results(merged_df):
    """Analyze R3D-18 results"""
    print(f"Analyzing {len(merged_df)} R3D-18 predictions...")

    # Count each type
    counts = merged_df['prediction_type'].value_counts().to_dict()

    # Ensure all categories are present
    for cat in ['TP', 'TN', 'FP', 'FN']:
        if cat not in counts:
            counts[cat] = 0

    print(f"R3D Prediction counts: {counts}")

    # Calculate metrics
    y_true = merged_df['true_label'].values
    y_pred = merged_df['predicted_label'].values

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    print(f"R3D Metrics: {metrics}")

    # Separate by type and sort by confidence
    analysis_results = {}
    for pred_type in ['TP', 'FP', 'FN', 'TN']:
        subset = merged_df[merged_df['prediction_type'] == pred_type].copy()
        subset = subset.sort_values('confidence', ascending=False)
        analysis_results[pred_type] = subset

    analysis_results['counts'] = counts
    analysis_results['metrics'] = metrics

    return analysis_results


def create_r3d_summary_figure(analysis_results, output_dir):
    """Create summary figure for R3D results"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('R3D-18 Drone Detection Results: Examples by Prediction Type', fontsize=16)

    categories = [
        ('True Positives', analysis_results['TP'], 'green'),
        ('False Positives', analysis_results['FP'], 'red'),
        ('False Negatives', analysis_results['FN'], 'orange'),
        ('True Negatives', analysis_results['TN'], 'blue')
    ]

    for cat_idx, (title, examples_df, color) in enumerate(categories):
        row = cat_idx // 2
        start_col = (cat_idx % 2) * 2

        # Get top 2 examples
        examples = examples_df.head(2)

        for col, (_, example) in enumerate(examples.iterrows()):
            if col < 2:
                ax = axes[row, start_col + col]

                # Try to load representative frame
                if 'representative_frame' in example and pd.notna(example['representative_frame']):
                    try:
                        img = Image.open(example['representative_frame'])
                        ax.imshow(img)
                        ax.axis('off')

                        conf = example['confidence']
                        filename = os.path.basename(example['tubelet_path'])
                        ax.set_title(f"{title}\n{filename}\nConf: {conf:.3f}",
                                     fontsize=8, color=color)
                    except:
                        ax.text(0.5, 0.5, f'{title}\nFrame not\navailable',
                                ha='center', va='center', color=color)
                        ax.set_title(title, color=color)
                else:
                    ax.text(0.5, 0.5, f'{title}\nNo examples\navailable',
                            ha='center', va='center', color=color)
                    ax.set_title(title, color=color)
                ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r3d_detection_examples.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def save_r3d_analysis(analysis_results, output_dir):
    """Save detailed R3D analysis to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary = {
        'total_samples': sum(analysis_results['counts'].values()),
        'counts': analysis_results['counts'],
        'metrics': analysis_results['metrics']
    }

    with open(os.path.join(output_dir, 'r3d_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed CSV files
    for category in ['TP', 'FP', 'FN', 'TN']:
        if category in analysis_results and len(analysis_results[category]) > 0:
            examples_df = analysis_results[category].head(20)  # Top 20

            # Select relevant columns for CSV
            output_cols = ['tubelet_path', 'image_path', 'video_name', 'frame_idx',
                           'true_label', 'predicted_label', 'confidence', 'prediction_type']

            # Only include columns that exist
            available_cols = [col for col in output_cols if col in examples_df.columns]
            csv_df = examples_df[available_cols].copy()

            # Add readable labels
            csv_df['true_label_name'] = csv_df['true_label'].apply(lambda x: 'Drone' if x == 1 else 'Background')
            csv_df['predicted_label_name'] = csv_df['predicted_label'].apply(
                lambda x: 'Drone' if x == 1 else 'Background')

            csv_df.to_csv(os.path.join(output_dir, f'r3d_{category}_examples.csv'), index=False)
            print(f"Saved {len(csv_df)} R3D {category} examples to CSV")


def main():
    """Main R3D qualitative analysis pipeline"""
    print("Starting R3D-18 qualitative analysis from existing results...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing prediction results
    results_df = load_r3d_predictions()

    # Load test index for file paths
    test_df = load_test_index()

    # Merge results with paths
    merged_df = merge_results_with_paths(results_df, test_df)

    if len(merged_df) == 0:
        print("No matching data found between results and test index. Check the merge keys.")
        return

    # Extract representative frames
    extract_representative_frames(merged_df, OUTPUT_DIR)

    # Analyze results
    analysis_results = analyze_r3d_results(merged_df)

    # Create visualizations
    create_r3d_summary_figure(analysis_results, OUTPUT_DIR)

    # Save detailed analysis
    save_r3d_analysis(analysis_results, OUTPUT_DIR)

    print(f"\n✅ R3D analysis complete! Results saved to: {OUTPUT_DIR}")
    print(f"📊 R3D Results Summary:")
    print(f"  Accuracy: {analysis_results['metrics']['accuracy']:.4f}")
    print(f"  Precision: {analysis_results['metrics']['precision']:.4f}")
    print(f"  Recall: {analysis_results['metrics']['recall']:.4f}")
    print(f"  F1-Score: {analysis_results['metrics']['f1']:.4f}")
    print(f"  False Positives: {analysis_results['counts'].get('FP', 0)}")
    print(f"  False Negatives: {analysis_results['counts'].get('FN', 0)}")


if __name__ == "__main__":
    main()