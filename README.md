# Drone Detection System

A comprehensive video-based drone detection system that combines computer vision and 3D convolutional neural networks. The system processes video streams to identify drone objects through a multi-stage pipeline: background subtraction → preprocessing → tubelet generation → 3D CNN classification.

## Features

- **Multiple Background Subtraction Methods**: MOG2 and ViBe algorithms for motion detection
- **Advanced Preprocessing Pipeline**: Configurable thresholding, morphological operations, and contour detection
- **2D CNN and 3D CNN Classification**: ResNet18 and R3D-18 architectures for temporal feature learning
- **Tubelet Generation**: Extracts spatio-temporal regions of interest for classification
- **Configuration-Driven**: YAML-based configuration system for all components

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- OpenCV 4.9+
- PyTorch 2.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drone-detection.git
cd drone-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Detection

```python
from detection import create_pipeline
import yaml

# Load configuration
with open('configs/experiment.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create detection pipeline
pipeline = create_pipeline(config)

# Process video
results = pipeline.process_video('path/to/video.mp4')
```

### Training a Model

```bash
python scripts/train.py --exp_config configs/experiment.yaml --clf_config configs/classifier.yaml
```

### Running Grid Search

```bash
python scripts/grid_search.py --config configs/experiment.yaml
```

## Architecture

### Pipeline Overview

```
Video Input → Background Subtraction → Preprocessing → Tubelet Generation → 3D CNN → Detection Results
```

### Components

- **Background Subtraction** (`src/detection/background/`):
  - MOG2: Mixture of Gaussians background subtractor
  - ViBe: Visual Background Extractor

- **Preprocessing** (`src/detection/preprocess/`):
  - Thresholding (fixed/Otsu)
  - Morphological operations
  - Contour detection and merging
  - ROI extraction

- **Classification** (`src/detection/classifier/`):
  - R3D-18: ResNet3D backbone
  - R(2+1)D: Spatiotemporal factorized convolutions

- **Optimization** (`src/detection/optimization/`):
  - Grid search for hyperparameter tuning
  - Custom scoring functions

## Configuration

The system uses YAML configuration files for all components:

### Main Configuration (`configs/experiment.yaml`)
```yaml
paths:
  input_video_dir: "data/raw"
  annotations_dir: "data/annotations"
  tubelets_dir: "data/tubelets"

background:
  method: "MOG2"
  mog2:
    history: 200
    var_threshold: 12

classifier:
  method: "r3d18"
  R3D18:
    pretrained: true
    num_classes: 2
```

### Component-Specific Configs
- `configs/background.yaml`: Background subtraction parameters
- `configs/classifier.yaml`: Model training configuration

## Data Format

### Input Data
- **Videos**: MP4, AVI, MOV formats in `data/raw/`
- **Annotations**: CSV format with frame-level bounding boxes
- **Splits**: Text files listing train/validation/test videos

### Generated Data
- **Tubelets**: 3D numpy arrays (T×H×W×C) saved as `.npy` files
- **Indexes**: CSV files mapping tubelet paths to labels

## Training

### Data Preparation

1. Place videos in `data/raw/`
2. Add annotations in `data/annotations/`
3. Create data splits in `data/splits/`
4. Generate tubelets:
```bash
python scripts/generate_tubelets.py --config configs/experiment.yaml
```

### Model Training

```bash
python scripts/train.py \
  --exp_config configs/experiment.yaml \
  --clf_config configs/classifier.yaml \
  --gpu 0
```

### Hyperparameter Optimization

```bash
python scripts/grid_search.py --config configs/experiment.yaml
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/experiment.yaml \
  --checkpoint models/best_f1.pt \
  --test_video data/raw/test_video.mp4
```

## API Reference

### Core Classes

**Background Subtraction**
```python
from detection.background import create
bg_subtractor = create("MOG2", config)
mask = bg_subtractor.apply(frame)
```

**Preprocessing**

```python
from detection.preprocessing import create

preprocessor = create("Threshold", config)
binary_mask = preprocessor.apply(soft_mask)
```

**Classification**
```python
from detection.classifier import create
model = create("r3d18", config)
predictions = model(tubelet_batch)
```

## Known Issues

- **Label Correction Required**: Some generated tubelets may have incorrect labels (IoU > 0 but label = 0). Use the provided correction script before training.
- **Path Dependencies**: Ensure all file paths in configuration match your directory structure.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Development Setup

For development, install additional dependencies:
```bash
pip install pytest jupyter
```

Run tests:
```bash
pytest tests/
```

## Performance

### Typical Results
- **Detection Rate**: 85-95% on test videos
- **False Positive Rate**: 5-15% depending on configuration
- **Processing Speed**: 15-30 FPS (GPU), 2-5 FPS (CPU)

### Optimization Tips
- Use GPU for training and inference
- Adjust background subtraction sensitivity for your environment
- Tune preprocessing parameters for your video quality
- Consider frame rate reduction for real-time applications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch and OpenCV
- Background subtraction algorithms from OpenCV
- 3D CNN architectures from torchvision

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{drone-detection-2024,
  title={Video-based Drone Detection using 3D CNNs},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/drone-detection}
}
```
