"""
Image viewer for displaying qualitative analysis results from CSV files.
Used to visualize TP, FP, TN, FN examples from model evaluation.
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from typing import Optional, Dict, List


def display_images_from_csv(csv_path: str,
                            num_images: int = 4,
                            title_prefix: str = "",
                            cols: int = 4) -> None:
    """
    Display images from a CSV file containing image paths and predictions.

    Args:
        csv_path: Path to CSV file with 'image_path' column
        num_images: Number of images to display
        title_prefix: Prefix for the plot title
        cols: Number of columns in the grid
    """
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} entries from {os.path.basename(csv_path)}")

    # Limit to available images
    num_images = min(num_images, len(df))

    if num_images == 0:
        print("⚠ No images to display")
        return

    # Calculate grid size
    rows = (num_images + cols - 1) // cols

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))

    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{title_prefix} - Top {num_images} Examples', fontsize=16, fontweight='bold')

    for i in range(num_images):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        # Get image info
        image_path = df.iloc[i]['image_path']
        confidence = df.iloc[i].get('confidence', 'N/A')
        true_label = df.iloc[i].get('true_label', 'N/A')
        predicted_label = df.iloc[i].get('predicted_label', 'N/A')

        try:
            # Load and display image
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')

            # Create title with info
            filename = os.path.basename(image_path)

            if confidence != 'N/A':
                title = f"{filename}\nTrue: {true_label} | Pred: {predicted_label}\nConf: {confidence:.3f}"
            else:
                title = filename

            ax.set_title(title, fontsize=8)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading:\n{os.path.basename(image_path)}\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')

    # Hide empty subplots
    for i in range(num_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def display_all_categories(base_dir: str,
                           csv_files: Optional[Dict[str, str]] = None,
                           num_images_per_category: int = 8) -> None:
    """
    Display images from multiple CSV files (TP, FP, TN, FN).

    Args:
        base_dir: Base directory containing CSV files
        csv_files: Dictionary mapping category names to CSV filenames
        num_images_per_category: Number of images to show per category
    """
    # Default CSV files
    if csv_files is None:
        csv_files = {
            "True Positives": "TP_examples.csv",
            "False Positives": "FP_examples.csv",
            "False Negatives": "FN_examples.csv",
            "True Negatives": "TN_examples.csv"
        }

    # Display images from each CSV
    for category, filename in csv_files.items():
        csv_path = os.path.join(base_dir, filename)

        print(f"\n{'=' * 50}")
        print(f"Displaying {category}")
        print(f"{'=' * 50}")

        display_images_from_csv(csv_path,
                                num_images=num_images_per_category,
                                title_prefix=category)


def display_specific_csv(csv_path: str, num_images: int = 12) -> None:
    """
    Convenience function to display images from a specific CSV file.

    Args:
        csv_path: Path to CSV file
        num_images: Number of images to display
    """
    display_images_from_csv(csv_path, num_images, os.path.basename(csv_path))


def create_comparison_grid(csv_paths: Dict[str, str],
                           num_images: int = 4,
                           save_path: Optional[str] = None) -> None:
    """
    Create a comparison grid showing examples from multiple categories side by side.

    Args:
        csv_paths: Dictionary mapping category names to CSV paths
        num_images: Number of images per category
        save_path: Optional path to save the figure
    """
    categories = list(csv_paths.keys())
    n_categories = len(categories)

    # Create figure
    fig, axes = plt.subplots(n_categories, num_images,
                             figsize=(4 * num_images, 4 * n_categories))

    if n_categories == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Qualitative Analysis Comparison', fontsize=16, fontweight='bold')

    for row, (category, csv_path) in enumerate(csv_paths.items()):
        # Read CSV
        if not os.path.exists(csv_path):
            print(f"⚠ Skipping {category}: CSV not found")
            continue

        df = pd.read_csv(csv_path)
        n_available = min(num_images, len(df))

        # Add category label
        axes[row, 0].text(-0.1, 0.5, category,
                          transform=axes[row, 0].transAxes,
                          fontsize=12, fontweight='bold',
                          rotation=90, va='center', ha='right')

        for col in range(num_images):
            ax = axes[row, col]

            if col < n_available:
                image_path = df.iloc[col]['image_path']
                confidence = df.iloc[col].get('confidence', 0)

                try:
                    image = Image.open(image_path)
                    ax.imshow(image)
                    ax.set_title(f"Conf: {confidence:.2f}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center')

            ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison grid to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Image Viewer - Example Usage:")
    print("1. Display all categories: display_all_categories('results/qualitative_analysis')")
    print("2. Display specific CSV: display_specific_csv('TP_examples.csv', num_images=16)")
    print("3. Create comparison: create_comparison_grid({'TP': 'tp.csv', 'FP': 'fp.csv'})")