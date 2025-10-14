#!/usr/bin/env python3
"""
Simple ResNet-18 evaluation script that handles the backbone prefix issue
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection"
CHECKPOINT_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\models\resnet18_20250909_174755_20250909_174755\best.pth"
TEST_CSV_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\2d_regions\test\test_2d_index.csv"


class RegionDataset:
    """Simple dataset for loading images"""

    def __init__(self, csv_file, transform=None):
        print(f"Loading CSV: {csv_file}")
        # Try reading with different separators and check the format
        try:
            # First, peek at the file to understand format
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()
                print(f"First line of CSV: {first_line}")

            # Check if first line contains header words
            if 'image_path' in first_line.lower() or 'label' in first_line.lower():
                print("Detected header row, skipping it...")
                # Read with header row (skip first line)
                self.data = pd.read_csv(csv_file, header=0)  # Use first row as header
                # Rename columns to standard names
                columns = list(self.data.columns)
                if len(columns) >= 3:
                    self.data.columns = ['image_path', 'label', 'tubelet_path']
            else:
                # No header, read normally
                self.data = pd.read_csv(csv_file, header=None, names=['image_path', 'label', 'tubelet_path'])

            print(f"Loaded {len(self.data)} samples")
            print(f"Sample row: {self.data.iloc[0].to_dict()}")

            # Verify the first image path exists
            sample_path = self.data.iloc[0]['image_path']
            print(f"Sample image path: {sample_path}")
            print(f"Sample path exists: {os.path.exists(sample_path)}")

            # Check if paths are relative and need to be made absolute
            if not os.path.exists(sample_path):
                # Try making path relative to CSV file location
                csv_dir = os.path.dirname(csv_file)
                abs_sample_path = os.path.join(csv_dir, sample_path)
                print(f"Trying absolute path: {abs_sample_path}")
                print(f"Absolute path exists: {os.path.exists(abs_sample_path)}")

                if os.path.exists(abs_sample_path):
                    print("Converting all paths to absolute paths...")
                    self.data['image_path'] = self.data['image_path'].apply(
                        lambda x: os.path.join(csv_dir, x)
                    )

        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label = int(row['label'])

        # Debug: print first few paths to check format
        if idx < 3:
            print(f"Loading sample {idx}: {image_path} (label: {label})")

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        if self.transform:
            image = self.transform(image)

        return image, label


def load_model_with_backbone_fix():
    """Load model and fix the backbone prefix issue"""
    print("Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    original_state_dict = checkpoint['model_state_dict']

    print("Creating ResNet-18 model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 2)
    )

    print("Fixing state dict keys...")
    new_state_dict = {}
    for key, value in original_state_dict.items():
        if key.startswith('backbone.'):
            new_key = key[9:]  # Remove 'backbone.' prefix
            new_state_dict[new_key] = value
            print(f"  {key} -> {new_key}")
        else:
            new_state_dict[key] = value

    print("Loading fixed state dict...")
    model.load_state_dict(new_state_dict)
    print("✓ Model loaded successfully!")

    return model, checkpoint


def evaluate_model():
    """Run evaluation"""
    print("=" * 60)
    print("SIMPLE RESNET-18 EVALUATION")
    print("=" * 60)

    # Load model
    model, checkpoint = load_model_with_backbone_fix()
    model.eval()

    # Create test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load test data
    print(f"\nLoading test data from: {TEST_CSV_PATH}")
    test_dataset = RegionDataset(TEST_CSV_PATH, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Test set size: {len(test_dataset)} samples")

    # Run inference
    print("\nRunning inference...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Forward pass
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

            # Store results
            all_preds.extend(predictions.numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probabilities.numpy())

            if batch_idx % 20 == 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches")

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Compute metrics
    print("\nComputing metrics...")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    val_f1 = checkpoint.get('best_f1', 0)
    print(f"\nOVERALL METRICS:")
    print(f"  Test Accuracy:   {accuracy:.4f}")
    print(f"  Test Precision:  {precision:.4f}")
    print(f"  Test Recall:     {recall:.4f}")
    print(f"  Test F1-Score:   {f1:.4f}")

    if val_f1 > 0:
        print(f"\nCOMPARISON:")
        print(f"  Validation F1:   {val_f1:.4f}")
        print(f"  Test F1:         {f1:.4f}")
        print(f"  Difference:      {f1 - val_f1:+.4f}")

    print(f"\nPER-CLASS METRICS:")
    class_names = ['Background', 'Drone']
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")

    print(f"\nCONFUSION MATRIX:")
    print(f"              Predicted")
    print(f"              Bg    Drone")
    print(f"  Actual Bg   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"        Drone {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    print(f"\nTEST SET DISTRIBUTION:")
    for label, count in zip(unique, counts):
        class_name = 'Background' if label == 0 else 'Drone'
        print(f"  {class_name}: {count} samples ({count / len(y_true):.1%})")

    # Error analysis
    errors = y_true != y_pred
    error_count = errors.sum()
    total_count = len(y_true)

    print(f"\nERROR ANALYSIS:")
    print(
        f"  Correct predictions: {total_count - error_count} / {total_count} ({(total_count - error_count) / total_count:.1%})")
    print(f"  Errors: {error_count} / {total_count} ({error_count / total_count:.1%})")

    if error_count > 0:
        false_positives = ((y_true == 0) & (y_pred == 1)).sum()
        false_negatives = ((y_true == 1) & (y_pred == 0)).sum()
        print(f"    False Positives (Bg→Drone): {false_positives}")
        print(f"    False Negatives (Drone→Bg): {false_negatives}")

    # Recommendation
    print(f"\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if f1 > 0.85:
        print("🟢 EXCELLENT - Model ready for deployment!")
    elif f1 > 0.80:
        print("🟡 GOOD - Consider deployment with monitoring")
    else:
        print("🔴 NEEDS IMPROVEMENT - Consider retraining")

    if val_f1 > 0 and abs(f1 - val_f1) > 0.03:
        print("⚠️  Large validation/test gap - check for overfitting")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'val_f1': val_f1
    }


if __name__ == "__main__":
    try:
        results = evaluate_model()
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Final test F1-score: {results['f1']:.4f}")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()