"""
Dataset module for drone detection.
Provides factory functions for creating 2D and 3D datasets.
"""
from .frame_dataset import DroneFrameDataset, create_data_loaders
from .region_dataset import RegionDataset, create_2d_dataloaders
from .tubelet_dataset import TubeletDataset, create_dataloaders

# Dataset registry
DATASET_REGISTRY = {
    'frame': DroneFrameDataset,
    'region': RegionDataset,
    'tubelet': TubeletDataset,
}


def create_dataset(dataset_type, config, split='train', transform=None):
    """
    Factory function to create datasets.

    Args:
        dataset_type: Type of dataset ('frame', 'region', 'tubelet')
        config: Dataset configuration dictionary
        split: Dataset split ('train', 'val', 'test')
        transform: Optional transforms to apply

    Returns:
        Dataset instance

    Example:
        >>> config = {'data_dir': 'data/', 'csv_path': 'train.csv'}
        >>> dataset = create_dataset('tubelet', config, split='train')
    """
    if dataset_type not in DATASET_REGISTRY:
        available = ', '.join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available types: {available}"
        )

    dataset_class = DATASET_REGISTRY[dataset_type]

    # Create dataset based on type
    if dataset_type == 'frame':
        # For frame dataset
        return dataset_class(
            video_list=config.get('video_list', []),
            video_dir=config.get('video_dir', 'data/raw'),
            annotations_dir=config.get('annotations_dir', 'data/annotations'),
            input_size=config.get('input_size', 112),
            transform=transform
        )

    elif dataset_type == 'region':
        # For 2D region dataset
        csv_path = config.get(f'{split}_csv', f'data/2d_regions/{split}/{split}_2d_index.csv')
        return dataset_class(
            csv_path=csv_path,
            transform=transform
        )

    elif dataset_type == 'tubelet':
        # For 3D tubelet dataset
        csv_path = config.get(f'{split}_csv', f'data/tubelets/{split}_index.csv')
        root_dir = config.get('tubelets_root', 'data/tubelets')
        return dataset_class(
            csv_path=csv_path,
            root_dir=root_dir,
            transform=transform
        )

    else:
        raise ValueError(f"Dataset type {dataset_type} not implemented")


__all__ = [
    'DroneFrameDataset',
    'RegionDataset',
    'TubeletDataset',
    'create_dataset',
    'create_data_loaders',
    'create_2d_dataloaders',
    'create_dataloaders',
    'DATASET_REGISTRY'
]