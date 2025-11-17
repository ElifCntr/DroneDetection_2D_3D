# Drone Detection in Surveillance Videos: 2D vs 3D CNN Comparison

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive comparison of 2D and 3D CNN approaches for drone detection in surveillance videos.**

## 📋 Overview

This project implements and compares two deep learning approaches for detecting drones in surveillance video footage:
- **2D CNN (ResNet-18)**: Spatial feature extraction from individual frames
- **3D CNN (R3D-18)**: Spatiotemporal feature extraction from frame sequences

Our research demonstrates that 2D spatial models can outperform 3D temporal models for drone detection in static surveillance scenarios, achieving **95.54% F1-score** compared to 86.54% for 3D models.

## 🎯 Key Features

- **Dual Architecture Comparison**: Side-by-side evaluation of 2D and 3D CNN approaches
- **Comprehensive Pipeline**: Complete workflow from data preprocessing to evaluation
- **Production-Ready Code**: Modular, well-documented, and extensively tested
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Rich Visualization**: Comprehensive evaluation plots and metrics
- **GPU Accelerated**: Optimized for CUDA-enabled training and inference

## 📊 Results Summary

### Model Performance on Test Set (14,946 samples)

| Model | F1-Score | Accuracy | Precision | Recall | Training Time |
|-------|----------|----------|-----------|--------|---------------|
| **ResNet-18 (2D)** | **95.54%** | **95.38%** | **95.82%** | **95.38%** | 6 min/epoch |
| R3D-18 (3D) | 86.54% | 84.30% | 91.15% | 84.30% | 12 min/epoch |

**Key Findings:**
- 2D models achieve 9% higher F1-score than 3D models
- 2D models are 2× faster to train
- Spatial features alone are sufficient for static camera surveillance
- Lower false negative rate with 2D approach (210 vs 306)

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA 11.8+ (for GPU acceleration)
8GB+ GPU RAM recommended
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/DroneDetection_2D_3D.git
cd DroneDetection_2D_3D
```

2. **Create conda environment**
```bash
conda create -n drone-detection python=3.8
conda activate drone-detection
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python scikit-learn pandas matplotlib seaborn pyyaml tqdm
```

4. **Verify installation**
```bash
python test_imports.py
```

## 📁 Project Structure

```
DroneDetection_2D_3D/
├── src/                          # Source code
│   ├── datasets/                 # Dataset loaders
│   │   ├── frame_dataset.py     # Single frame dataset
│   │   ├── region_dataset.py    # 2D region dataset
│   │   └── tubelet_dataset.py   # 3D tubelet dataset
│   ├── models/                   # Model architectures
│   │   ├── resnet18.py          # 2D ResNet-18
│   │   └── r3d18.py             # 3D R3D-18
│   ├── training/                 # Training utilities
│   │   └── trainer.py           # Training loop
│   ├── evaluation/               # Evaluation tools
│   ├── inference/                # Inference pipeline
│   └── utils/                    # Utility functions
│       ├── visualization/        # Plotting tools
│       ├── checkpoint.py        # Model checkpointing
│       ├── config.py            # Config loading
│       ├── logging.py           # Logging utilities
│       └── metrics.py           # Evaluation metrics
├── scripts/                      # Executable scripts
│   ├── train_2d.py              # Train 2D model
│   ├── train_3d.py              # Train 3D model
│   ├── evaluate_2d.py           # Evaluate 2D model
│   └── evaluate_3d.py           # Evaluate 3D model
├── configs/                      # Configuration files
│   └── experiment.yaml          # Training configuration
├── data/                         # Data directory
│   ├── raw/                     # Raw video files
│   ├── annotations/             # Ground truth annotations
│   ├── 2d_regions/              # Extracted 2D regions
│   └── tubelets/                # Generated 3D tubelets
├── outputs/                      # Training outputs
│   ├── checkpoints/             # Saved models
│   ├── logs/                    # Training logs
│   └── evaluation/              # Evaluation results
├── test_imports.py              # Test installation
├── visualize_test_results.py   # Visualization script
└── README.md                    # This file
```

## 🎓 Usage

### 1. Data Preparation

Place your data in the following structure:
```bash
data/
├── raw/                    # Raw video files (.mp4)
├── annotations/            # Ground truth bounding boxes (.txt)
└── splits/
    ├── train_videos.txt   # Training video list
    ├── val_videos.txt     # Validation video list
    └── test_videos.txt    # Test video list
```

### 2. Training

#### Train 2D Model (ResNet-18)
```bash
python scripts/train_2d.py \
    --config configs/experiment.yaml \
    --train-csv data/2d_regions/train/train_2d_index.csv \
    --val-csv data/2d_regions/val/val_2d_index.csv \
    --epochs 50 \
    --batch-size 32 \
    --gpu 0 \
    --experiment-name resnet18_experiment
```

#### Train 3D Model (R3D-18)
```bash
python scripts/train_3d.py \
    --config configs/experiment.yaml \
    --train-csv data/tubelets/train_index.csv \
    --val-csv data/tubelets/val_index.csv \
    --epochs 50 \
    --batch-size 16 \
    --gpu 0 \
    --experiment-name r3d18_experiment
```

### 3. Evaluation

#### Evaluate 2D Model
```bash
python scripts/evaluate_2d.py \
    --config configs/experiment.yaml \
    --checkpoint outputs/checkpoints/resnet18_experiment/best.pth \
    --test-csv data/2d_regions/test/test_2d_index.csv \
    --output-dir outputs/evaluation/resnet18 \
    --gpu 0 \
    --save-predictions
```

#### Evaluate 3D Model
```bash
python scripts/evaluate_3d.py \
    --config configs/experiment.yaml \
    --checkpoint outputs/checkpoints/r3d18_experiment/best.pth \
    --test-csv data/tubelets/test_index.csv \
    --output-dir outputs/evaluation/r3d18 \
    --gpu 0 \
    --save-predictions
```

### 4. Visualization

Generate evaluation plots:
```bash
python visualize_test_results.py
```

Outputs:
- `confusion_matrix.png` - Confusion matrix heatmap
- `roc_curve.png` - ROC curve with AUC
- `pr_curve.png` - Precision-Recall curve
- `metrics_bars.png` - Metrics comparison
- `per_class_metrics.png` - Per-class performance

## 📊 Configuration

Edit `configs/experiment.yaml` to customize:

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001

model:
  name: resnet18  # or r3d18
  num_classes: 2
  pretrained: true
  freeze_backbone: true
  dropout: 0.5

evaluation:
  batch_size: 32
  save_predictions: true
  save_visualizations: true
```

## 🔬 Research Context

This work is part of a PhD research project at the **University of Sussex** investigating efficient drone detection methods for surveillance applications.

**Dataset**: 37 static surveillance videos from the Drone-vs-Bird Detection Challenge
- **Training**: 30,967 samples
- **Validation**: 10,756 samples  
- **Test**: 14,946 samples
- **Class distribution**: 10.5% positive (drone), 89.5% negative (background)

## 📈 Detailed Results

### Confusion Matrix (2D Model)

```
                 Predicted
              Background  Drone
Actual  
Background     12,892      480
Drone             210    1,364
```

### Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Background | 98.40% | 96.41% | 97.39% |
| Drone | 73.97% | 86.66% | 79.81% |

## 🛠️ Development

### Running Tests

```bash
# Test all imports
python test_imports.py

# Test all components
python test_all_components.py

# Test data loading
python test_data_loading.py
```

### Code Organization

- **Modular Design**: Clear separation of concerns
- **Factory Pattern**: Easy model/dataset creation
- **Configuration-Driven**: YAML-based configuration
- **Type Hints**: Full type annotations throughout
- **Documentation**: Comprehensive docstrings

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{drone_detection_2d_3d,
  author = {Elif Karaca},
  title = {Drone Detection in Surveillance Videos: 2D vs 3D CNN Comparison},
  year = {2025},
  institution = {University of Sussex},
  url = {https://github.com/yourusername/DroneDetection_2D_3D}
}
```

## 👥 Authors

**Elif Karaca**  
PhD Student, University of Sussex  
Supervisor: Dr. Phil Birch

Co-authors: Xudong Han, Yueying Tian

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- University of Sussex for computational resources
- YLSY scholarship funding
- Drone-vs-Bird Detection Challenge for the dataset
- PyTorch and torchvision teams for the frameworks

## 📧 Contact

For questions or collaborations:
- Email: [your.email@sussex.ac.uk]
- GitHub Issues: [Project Issues](https://github.com/yourusername/DroneDetection_2D_3D/issues)

---

**⭐ If you find this work useful, please star the repository!**
