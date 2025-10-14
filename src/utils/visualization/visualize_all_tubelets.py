import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_all_tubelets(base_path, output_folder='tubelet_visualizations'):
    """
    Go through all tubelet .npy files and save visualizations

    Args:
        base_path: Path to tubelets/test folder
        output_folder: Where to save the visualizations
    """

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get all .npy files directly in base_path
    npy_files = [f for f in os.listdir(base_path)
                 if f.endswith('.npy')]

    print(f"Found {len(npy_files)} tubelet files")

    total_saved = 0

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
                axes[i].set_title(f't-{1 - i}' if i == 0 else (f't' if i == 1 else f't+1'))

            plt.suptitle(npy_file, fontsize=10)
            plt.tight_layout()

            # Save
            output_name = npy_file.replace('.npy', '.png')
            output_path = os.path.join(output_folder, output_name)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            total_saved += 1

            if total_saved % 50 == 0:
                print(f"  Saved {total_saved} visualizations...")

        except Exception as e:
            print(f"  Error processing {npy_file}: {e}")

    print(f"\nDone! Saved {total_saved} tubelet visualizations to '{output_folder}/'")


# Run this
base_path = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\tubelets\test"
visualize_all_tubelets(base_path)