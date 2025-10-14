import numpy as np
import matplotlib.pyplot as plt


def show_tubelet(npy_path):
    """Show one tubelet from .npy file - 3 frames side by side"""

    # Load the tubelet
    tubelet = np.load(npy_path)

    print(f"Tubelet shape: {tubelet.shape}")

    # Handle different possible shapes
    # Could be (3, H, W, C) or (C, 3, H, W) or other
    if tubelet.shape[0] == 3:
        # Shape is (3, H, W, C)
        frames = tubelet
    elif tubelet.shape[1] == 3:
        # Shape is (C, 3, H, W) - transpose
        frames = np.transpose(tubelet, (1, 0, 2, 3))
    else:
        frames = tubelet

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    for i in range(min(3, len(frames))):
        axes[i].imshow(frames[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Just call the function directly with your path
npy_path = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\tubelets\train\2019_11_14_C0001_3922_matrice\pos_001868_000.npy"
show_tubelet(npy_path)