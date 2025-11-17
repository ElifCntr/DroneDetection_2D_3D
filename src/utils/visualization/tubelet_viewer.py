"""
Tubelet visualization utilities for both single and batch viewing.
Supports both .npy tubelet files for 3D models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Optional


def visualize_single_tubelet(npy_path: str, title: Optional[str] = None) -> None:
    """
    Display a single tubelet (3 frames side by side).

    Args:
        npy_path: Path to .npy tubelet file
        title: Optional custom title
    """
    # Load the tubelet
    tubelet = np.load(npy_path)

    print(f"Tubelet shape: {tubelet.shape}")

    # Handle different possible shapes
    if tubelet.shape[0] == 3:
        # Shape is (3, H, W, C)
        frames = tubelet
    elif tubelet.shape[1] == 3:
        # Shape is (C, 3, H, W) - transpose
        frames = np.transpose(tubelet, (1, 0, 2, 3))
    else:
        frames = tubelet

    # Create figure with 3 frames
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for i in range(min(3, len(frames))):
        axes[i].imshow(frames[i])
        axes[i].axis('off')
        axes[i].set_title(f't-{1 - i}' if i == 0 else ('t' if i == 1 else 't+1'))

    # Set overall title
    if title:
        plt.suptitle(title, fontsize=10)
    else:
        plt.suptitle(os.path.basename(npy_path), fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_batch_tubelets(base_path: str,
                             output_folder: str = 'tubelet_visualizations',
                             max_files: Optional[int] = None) -> None:
    """
    Batch process and save visualizations for all tubelets in a directory.

    Args:
        base_path: Path to folder containing .npy tubelet files
        output_folder: Where to save the PNG visualizations
        max_files: Optional limit on number of files to process
    """
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get all .npy files
    npy_files = [f for f in os.listdir(base_path) if f.endswith('.npy')]

    if max_files:
        npy_files = npy_files[:max_files]

    print(f"Found {len(npy_files)} tubelet files to visualize")

    total_saved = 0
    total_errors = 0

    # Process each tubelet
    for npy_file in npy_files:
        npy_path = os.path.join(base_path, npy_file)

        try:
            # Load tubelet
            tubelet = np.load(npy_path)

            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            # Display 3 frames
            for i in range(3):
                axes[i].imshow(tubelet[i])
                axes[i].axis('off')
                axes[i].set_title(f't-{1 - i}' if i == 0 else ('t' if i == 1 else 't+1'))

            plt.suptitle(npy_file, fontsize=10)
            plt.tight_layout()

            # Save
            output_name = npy_file.replace('.npy', '.png')
            output_file = output_path / output_name
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            total_saved += 1

            if total_saved % 50 == 0:
                print(f"  ✓ Saved {total_saved} visualizations...")

        except Exception as e:
            print(f"  ✗ Error processing {npy_file}: {e}")
            total_errors += 1
            plt.close('all')  # Clean up any open figures

    print(f"\n✓ Done! Saved {total_saved} tubelet visualizations to '{output_folder}/'")
    if total_errors > 0:
        print(f"⚠ {total_errors} files failed to process")


def compare_tubelets(npy_paths: list, titles: Optional[list] = None) -> None:
    """
    Compare multiple tubelets side by side.

    Args:
        npy_paths: List of paths to .npy tubelet files
        titles: Optional list of titles for each tubelet
    """
    n_tubelets = len(npy_paths)

    fig, axes = plt.subplots(n_tubelets, 3, figsize=(9, 3 * n_tubelets))

    if n_tubelets == 1:
        axes = axes.reshape(1, -1)

    for row, npy_path in enumerate(npy_paths):
        tubelet = np.load(npy_path)

        # Handle different shapes
        if tubelet.shape[0] == 3:
            frames = tubelet
        elif tubelet.shape[1] == 3:
            frames = np.transpose(tubelet, (1, 0, 2, 3))
        else:
            frames = tubelet

        # Display 3 frames
        for col in range(3):
            axes[row, col].imshow(frames[col])
            axes[row, col].axis('off')

            if row == 0:  # Add time labels only on first row
                axes[row, col].set_title(f't-{1 - col}' if col == 0 else ('t' if col == 1 else 't+1'))

        # Add row label
        if titles and row < len(titles):
            axes[row, 0].set_ylabel(titles[row], rotation=90, fontsize=10, labelpad=10)
        else:
            axes[row, 0].set_ylabel(os.path.basename(npy_path), rotation=90, fontsize=8, labelpad=10)

    plt.tight_layout()
    plt.show()


# Convenience functions for common use cases
def quick_view(npy_path: str) -> None:
    """Quick single tubelet viewer."""
    visualize_single_tubelet(npy_path)


def batch_save(base_path: str, output_folder: str = 'tubelet_visualizations') -> None:
    """Quick batch processing."""
    visualize_batch_tubelets(base_path, output_folder)


if __name__ == "__main__":
    # Example usage
    print("Tubelet Viewer - Example Usage:")
    print("1. View single tubelet: visualize_single_tubelet('path/to/file.npy')")
    print("2. Batch save: visualize_batch_tubelets('path/to/tubelets/', 'output_folder')")
    print("3. Compare: compare_tubelets(['file1.npy', 'file2.npy'], ['Positive', 'Negative'])")