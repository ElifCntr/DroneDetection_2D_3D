#!/usr/bin/env python3
"""
Qualitative analysis script to identify specific examples of:
- True Positives (successful drone detections)
- False Positives (background misclassified as drone)
- False Negatives (missed drone detections)
- True Negatives (correct background classification)

This will help create figures for your paper.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from sklearn.metrics import confusion_matrix
import json

# Configuration
PROJECT_ROOT = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection"
CHECKPOINT_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\models\resnet18_20250909_174755_20250909_174755\best.pth"
TEST_CSV_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\2d_regions\test\test_2d_index.csv"
OUTPUT_DIR = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\qualitative_analysis"


class RegionDataset:
    """Dataset for loading images with paths preserved"""

    def __init__(self, csv_file, transform=None):
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()

        if 'image_path' in first_line.lower():
            self.data = pd.read_csv(csv_file, header=0)
            columns = list(self.data.columns)
            if len(columns) >= 3:
                self.data.columns = ['image_path', 'label', 'tubelet_path']
        else:
            self.data = pd.read_csv(csv_file, header=None, names=['image_path', 'label', 'tubelet_path'])

        # Check and fix paths if needed
        sample_path = self.data.iloc[0]['image_path']
        if not os.path.exists(sample_path):
            csv_dir = os.path.dirname(csv_file)
            self.data['image_path'] = self.data['image_path'].apply(
                lambda x: os.path.join(csv_dir, x)
            )

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = int(row['label'])

        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()  # Keep original for visualization

        if self.transform:
            image = self.transform(image)

        return image, label, image_path, original_image


def load_model():
    """Load the trained ResNet-18 model"""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    original_state_dict = checkpoint['model_state_dict']

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    # Fix backbone prefix
    new_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith('backbone.'):
            new_key = key[9:]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def custom_collate_fn(batch):
    """Custom collate function to handle PIL images"""
    tensors = []
    labels = []
    paths = []
    original_images = []

    for tensor, label, path, orig_img in batch:
        tensors.append(tensor)
        labels.append(label)
        paths.append(path)
        original_images.append(orig_img)

    # Stack tensors and labels normally
    tensors = torch.stack(tensors)
    labels = torch.tensor(labels)

    # Return as separate lists for paths and images
    return tensors, labels, paths, original_images


def run_inference_with_paths():
    """Run inference and collect results with image paths"""
    print("Loading model and running inference...")

    model = load_model()

    test_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = RegionDataset(TEST_CSV_PATH, transform=test_transforms)
    # Use custom collate function
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=custom_collate_fn)

    results = []

    with torch.no_grad():
        for batch_idx, (data, target, image_paths, original_images) in enumerate(test_loader):
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0, prediction].item()

            true_label = target.item()

            # Determine prediction type
            if true_label == 1 and prediction == 1:
                pred_type = "TP"  # True Positive
            elif true_label == 0 and prediction == 0:
                pred_type = "TN"  # True Negative
            elif true_label == 0 and prediction == 1:
                pred_type = "FP"  # False Positive
            elif true_label == 1 and prediction == 0:
                pred_type = "FN"  # False Negative

            results.append({
                'image_path': image_paths[0],
                'true_label': true_label,
                'predicted_label': prediction,
                'confidence': confidence,
                'drone_prob': probabilities[0, 1].item(),
                'bg_prob': probabilities[0, 0].item(),
                'prediction_type': pred_type,
                'original_image': original_images[0]
            })

            if batch_idx % 1000 == 0:
                print(f"Processed {batch_idx}/{len(test_loader)} samples")

    return results


def analyze_results(results):
    """Analyze and categorize results"""
    print(f"Analyzing {len(results)} predictions...")

    # Count each type
    counts = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for result in results:
        counts[result['prediction_type']] += 1

    print(f"Prediction counts: {counts}")

    # Separate by type
    true_positives = [r for r in results if r['prediction_type'] == 'TP']
    false_positives = [r for r in results if r['prediction_type'] == 'FP']
    false_negatives = [r for r in results if r['prediction_type'] == 'FN']
    true_negatives = [r for r in results if r['prediction_type'] == 'TN']

    # Sort by confidence for better examples
    true_positives.sort(key=lambda x: x['confidence'], reverse=True)
    false_positives.sort(key=lambda x: x['confidence'], reverse=True)
    false_negatives.sort(key=lambda x: x['bg_prob'], reverse=True)  # Most confident wrong predictions

    return {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'TN': true_negatives,
        'counts': counts
    }


def extract_frame_info(image_path):
    """Extract video and frame information from path"""
    # Assuming path format like: .../video_name/frame_xxx.jpg
    parts = image_path.replace('\\', '/').split('/')
    if len(parts) >= 2:
        video_name = parts[-2] if 'frame' not in parts[-2] else parts[-3]
        frame_name = parts[-1]
        return video_name, frame_name
    return "unknown", os.path.basename(image_path)


def create_summary_figure(analysis_results, output_dir):
    """Create a summary figure showing examples of each prediction type"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Drone Detection Results: Examples by Prediction Type', fontsize=16)

    # Show top examples of each type
    categories = [
        ('True Positives', analysis_results['TP'][:2], 'green'),
        ('False Positives', analysis_results['FP'][:2], 'red'),
        ('False Negatives', analysis_results['FN'][:2], 'orange'),
        ('True Negatives', analysis_results['TN'][:2], 'blue')
    ]

    for cat_idx, (title, examples, color) in enumerate(categories):
        row = cat_idx // 2  # 0 for first two categories, 1 for last two
        start_col = (cat_idx % 2) * 2  # 0 or 2

        for col, example in enumerate(examples):
            if col < 2:  # Show 2 examples per category
                ax = axes[row, start_col + col]

                # Display image
                img = example['original_image']
                ax.imshow(img)
                ax.axis('off')

                # Add title with confidence
                video, frame = extract_frame_info(example['image_path'])
                conf = example['confidence']
                ax.set_title(f"{title}\n{conf:.3f}", fontsize=10, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_examples_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_analysis(analysis_results, output_dir):
    """Create detailed analysis files and individual example images"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results to JSON
    summary = {
        'total_samples': sum(analysis_results['counts'].values()),
        'counts': analysis_results['counts'],
        'accuracy': (analysis_results['counts']['TP'] + analysis_results['counts']['TN']) / sum(
            analysis_results['counts'].values()),
        'precision': analysis_results['counts']['TP'] / (
                    analysis_results['counts']['TP'] + analysis_results['counts']['FP']) if analysis_results['counts'][
                                                                                                'TP'] > 0 else 0,
        'recall': analysis_results['counts']['TP'] / (
                    analysis_results['counts']['TP'] + analysis_results['counts']['FN']) if analysis_results['counts'][
                                                                                                'TP'] > 0 else 0
    }

    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Create detailed CSV files for each category
    for category, examples in analysis_results.items():
        if category != 'counts' and examples:
            df_data = []
            for example in examples[:20]:  # Top 20 of each type
                video, frame = extract_frame_info(example['image_path'])
                df_data.append({
                    'image_path': example['image_path'],
                    'video': video,
                    'frame': frame,
                    'true_label': 'Drone' if example['true_label'] == 1 else 'Background',
                    'predicted_label': 'Drone' if example['predicted_label'] == 1 else 'Background',
                    'confidence': example['confidence'],
                    'drone_probability': example['drone_prob'],
                    'background_probability': example['bg_prob']
                })

            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(output_dir, f'{category}_examples.csv'), index=False)
            print(f"Saved {len(df)} {category} examples to CSV")


def analyze_environmental_conditions(analysis_results):
    """Analyze performance across different environmental conditions"""
    # This would require additional metadata about lighting, weather, etc.
    # For now, we can analyze by video source (assuming different videos = different conditions)

    all_results = []
    for category, examples in analysis_results.items():
        if category != 'counts':
            all_results.extend(examples)

    # Group by video
    video_performance = {}
    for result in all_results:
        video, _ = extract_frame_info(result['image_path'])
        if video not in video_performance:
            video_performance[video] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        video_performance[video][result['prediction_type']] += 1

    # Calculate metrics per video
    video_metrics = {}
    for video, counts in video_performance.items():
        total = sum(counts.values())
        if total > 0:
            accuracy = (counts['TP'] + counts['TN']) / total
            precision = counts['TP'] / (counts['TP'] + counts['FP']) if (counts['TP'] + counts['FP']) > 0 else 0
            recall = counts['TP'] / (counts['TP'] + counts['FN']) if (counts['TP'] + counts['FN']) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            video_metrics[video] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_samples': total,
                'drone_samples': counts['TP'] + counts['FN'],
                'bg_samples': counts['TN'] + counts['FP']
            }

    return video_metrics


def main():
    """Main analysis pipeline"""
    print("Starting qualitative analysis for paper...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run inference and collect results
    results = run_inference_with_paths()

    # Analyze results
    analysis_results = analyze_results(results)

    # Create visualizations
    create_summary_figure(analysis_results, OUTPUT_DIR)

    # Create detailed analysis files
    create_detailed_analysis(analysis_results, OUTPUT_DIR)

    # Analyze environmental conditions
    video_metrics = analyze_environmental_conditions(analysis_results)

    # Save video-level analysis
    video_df = pd.DataFrame.from_dict(video_metrics, orient='index')
    video_df.to_csv(os.path.join(OUTPUT_DIR, 'video_level_performance.csv'))

    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("- detection_examples_summary.png: Overview figure with examples")
    print("- TP_examples.csv, FP_examples.csv, FN_examples.csv: Detailed lists")
    print("- analysis_summary.json: Overall metrics")
    print("- video_level_performance.csv: Performance by video/condition")

    # Print some key findings
    print(f"\n📊 Key Findings:")
    print(f"False Positives: {len(analysis_results['FP'])} cases")
    print(f"False Negatives: {len(analysis_results['FN'])} cases")
    print(f"Best performing videos: {sorted(video_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]}")
    print(f"Most challenging videos: {sorted(video_metrics.items(), key=lambda x: x[1]['f1'])[:3]}")

    return analysis_results, video_metrics


if __name__ == "__main__":
    analysis_results, video_metrics = main()