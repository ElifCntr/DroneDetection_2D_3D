# Drone Detection: 2D vs 3D CNN Comparison

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Comparative study of 2D and 3D CNN approaches for drone detection in surveillance videos using adaptive background subtraction.**

## 📋 Overview

This project implements and compares two deep learning approaches for detecting drones in surveillance videos:
- **2D CNN (ResNet-18)**: Frame-by-frame spatial feature extraction
- **3D CNN (R3D-18)**: Spatiotemporal feature extraction from 3-frame sequences

Both approaches employ **adaptive background subtraction (BGS)** with MOG2 for motion-based candidate region proposal, followed by deep learning classification. Our research on the Drone-vs-Bird Detection Challenge dataset demonstrates that **2D spatial analysis outperforms 3D temporal modeling**, achieving **94.9% F1-score** compared to 83.8% for 3D.

## 🎯 Key Features

- **Dual Architecture Comparison**: Side-by-side evaluation of 2D vs 3D CNN approaches
- **Adaptive Background Subtraction**: MOG2 with scene-adaptive parameter tuning
- **Motion-Based Preprocessing**: Focus computational resources on motion regions
- **Comprehensive Evaluation**: Frame-level metrics, qualitative analysis, and visualization
- **Modular Design**: Clean separation of datasets, models, training, and evaluation
- **Production-Ready**: Extensively tested pipeline with safety checks
- **GPU Accelerated**: CUDA-optimized training and inference

## 📊 Results Summary

### Model Performance (Frame-Level Evaluation on Drone-vs-Bird Dataset)

| Model | F1-Score | Accuracy | Precision | Recall | Frame Coverage |
|-------|----------|----------|-----------|--------|----------------|
| **BGS + ResNet-18 (2D)** | **94.9%** | **94.7%** | **95.6%** | **94.7%** | 88.6% |
| BGS + ResNet3D-18 (3D) | 83.8% | 73.0% | 96.1% | 74.4% | 88.6% |

**Key Findings:**
- 2D achieves **11.1% higher F1-score** than 3D (94.9% vs 83.8%)
- 3D has slightly higher precision (96.1% vs 95.6%) but much lower recall (74.4% vs 94.7%)
- Background subtraction successfully captures **88.6% of drone frames**
- Spatial features with single-frame analysis outperform temporal modeling for static cameras
- T=3 temporal window insufficient for capturing meaningful drone motion patterns
- 3D struggles with hovering drones that exhibit minimal inter-frame motion

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
git clone https://github.com/ElifCntr/DroneDetection_2D_3D.git
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
├── src/                           # Source code
│   ├── datasets/                  # Dataset loaders
│   │   ├── frame_dataset.py      # Single frame dataset
│   │   ├── region_dataset.py     # 2D region dataset
│   │   └── tubelet_dataset.py    # 3D tubelet dataset
│   ├── models/                    # Model architectures
│   │   ├── resnet18.py           # 2D ResNet-18
│   │   └── r3d18.py              # 3D R3D-18
│   ├── training/                  # Training utilities
│   │   └── trainer.py            # Unified trainer
│   ├── evaluation/                # Evaluation system
│   │   ├── evaluators/           # Model evaluators
│   │   │   ├── evaluator_2d.py  # 2D evaluation
│   │   │   └── evaluator_3d.py  # 3D evaluation
│   │   └── analyzers/            # Analysis tools
│   │       ├── qualitative_analyzer.py
│   │       └── threshold_analyzer.py
│   ├── inference/                 # Inference pipeline
│   │   ├── video_processor.py    # Video processing
│   │   ├── predictors/           # Model predictors
│   │   │   ├── predictor_2d.py
│   │   │   └── predictor_3d.py
│   │   └── tubelets/             # Tubelet generation
│   │       ├── generator.py
│   │       └── generator_optimized.py
│   └── utils/                     # Utility functions
│       ├── visualization/         # Plotting tools
│       │   ├── image_viewer.py
│       │   ├── tubelet_viewer.py
│       │   └── interactive_detect.py
│       ├── checkpoint.py         # Model checkpointing
│       ├── config.py             # Config loading
│       ├── logging.py            # Logging utilities
│       └── metrics.py            # Evaluation metrics
├── scripts/                       # Executable scripts
│   ├── train_2d.py               # Train 2D model
│   ├── train_3d.py               # Train 3D model
│   ├── evaluate_2d.py            # Evaluate 2D model
│   ├── evaluate_3d.py            # Evaluate 3D model
│   └── extract_2d_from_tubelets.py
├── configs/                       # Configuration files
│   └── test_experiment.yaml      # Experiment config
├── data/                          # Data directory
│   ├── raw/                      # Raw video files
│   ├── annotations/              # Ground truth
│   ├── splits/                   # Train/val/test splits
│   ├── 2d_regions/               # Extracted 2D regions
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── tubelets/                 # 3D tubelet sequences
│       ├── train/
│       ├── val/
│       └── test/
├── results/                       # Analysis results
│   ├── 2d/                       # 2D model results
│   │   └── resnet18/
│   │       ├── TP_examples.csv
│   │       ├── FP_examples.csv
│   │       ├── FN_examples.csv
│   │       ├── TN_examples.csv
│   │       └── analysis_summary.json
│   └── r3d18_qualitative_analysis/  # 3D analysis
├── outputs/                       # Training outputs
│   ├── checkpoints/              # Saved models
│   ├── logs/                     # Training logs
│   └── evaluation/               # Eval results
├── test_imports.py               # Test installation
├── test_all_components.py        # Component tests
├── test_data_loading.py          # Data loading tests
├── safety_checker.py             # Safety validation
├── visualize_test_results.py    # Visualization script
└── README.md                     # This file
```

## 🎓 Usage

### 1. Data Preparation

Organize your data:
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
  batch_size: 32           # 32 for 2D, 16 for 3D
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001

model:
  name: resnet18           # or r3d18
  num_classes: 2
  pretrained: true
  freeze_backbone: true    # Transfer learning
  dropout: 0.5

background:
  method: MOG2
  var_threshold: 16        # Lower = more sensitive
  detect_shadows: false

evaluation:
  batch_size: 32
  save_predictions: true
  save_visualizations: true
```

## 🔬 Research Context

This work is part of PhD research at the **University of Sussex** investigating efficient drone detection methods for surveillance applications.

**Dataset**: Drone-vs-Bird Detection Challenge
- 37 static surveillance videos
- Training: 30,967 samples (61.5% positive)
- Validation: 10,756 samples (43.2% positive)  
- Test: 14,946 samples (10.5% positive)
- Challenging scenarios: water ripples, waving trees, birds, varying illumination

**Key Contributions**:
- Adaptive background subtraction with scene-adaptive parameters
- Comprehensive 2D vs 3D comparison for drone detection
- Frame-level evaluation methodology
- Modular, production-ready implementation

## 📈 Detailed Analysis

### Why 2D Outperforms 3D

1. **Static Camera Advantage**: Fixed surveillance cameras provide consistent viewpoints where spatial features are sufficient
2. **Temporal Window Limitation**: T=3 frames insufficient for capturing meaningful drone motion patterns
3. **Hovering Challenge**: 3D struggles with hovering drones exhibiting minimal inter-frame motion
4. **Visual Ambiguity**: Many drone samples are blurry and blend with backgrounds, making temporal feature extraction difficult
5. **BGS Warm-up Issue**: MOG2 requires warm-up period, but drones often appear in first frames of short sequences

### 3D Model Characteristics

**Strengths:**
- Highest precision (96.1%): When it predicts "drone," it's usually correct
- Reduced false positives through temporal consistency

**Weaknesses:**
- Lower recall (74.4%): Misses many drone instances
- Struggles with minimal motion scenarios
- Higher computational cost

### Performance Insights

| Metric | 2D (BGS + ResNet-18) | 3D (BGS + ResNet3D-18) | Analysis |
|--------|----------------------|------------------------|----------|
| **Frame Coverage** | 88.6% | 88.6% | BGS performs equally for both |
| **F1-Score** | 94.9% | 83.8% | 2D superior overall |
| **Precision** | 95.6% | 96.1% | 3D slightly better |
| **Recall** | 94.7% | 74.4% | 2D detects more drones |

## 🛠️ Development

### Running Tests

```bash
# Test all imports
python test_imports.py

# Test all components
python test_all_components.py

# Test data loading
python test_data_loading.py

# Safety checks
python safety_checker.py
```

### Code Organization

- **Modular Design**: Clear separation of concerns
- **Factory Pattern**: Easy model/dataset creation
- **Configuration-Driven**: YAML-based configuration
- **Type Hints**: Full type annotations
- **Comprehensive Testing**: Unit and integration tests

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{drone_detection_2d_3d_2025,
  author = {Elif Karaca and Phil Birch and Xudong Han and Yueying Tian},
  title = {Drone Detection: 2D vs 3D CNN Comparison for Surveillance Videos},
  year = {2025},
  publisher = {University of Sussex},
  url = {https://github.com/ElifCntr/DroneDetection_2D_3D}
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
- PyTorch and OpenCV teams

## 📧 Contact

For questions or collaborations:
- GitHub Issues: [Project Issues](https://github.com/ElifCntr/DroneDetection_2D_3D/issues)
- Email: e.ucurum@sussex.ac.uk

---

**⭐ If you find this work useful, please star the repository!**
