#!/usr/bin/env python3
"""
Quick test to verify all imports work correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("Testing imports...")

try:
    # Test dataset imports
    from datasets import create_dataset, DATASET_REGISTRY

    print("✅ datasets imports OK")
    print(f"   Available datasets: {list(DATASET_REGISTRY.keys())}")

    # Test model imports
    from models import create_model, list_models, MODEL_REGISTRY

    print("✅ models imports OK")
    print(f"   Available models: {list_models()}")

    # Test evaluation imports
    from evaluation.evaluators import create_evaluator_2d, create_evaluator_3d
    from evaluation.analyzers import create_qualitative_analyzer, create_threshold_analyzer

    print("✅ evaluation imports OK")

    # Test inference imports
    from inference.predictors import create_predictor_2d, create_predictor_3d

    print("✅ inference imports OK")

    # Test utils imports
    from utils.config import load_config
    from utils.logging import setup_logger

    print("✅ utils imports OK")

    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("\nYour project structure is ready to use!")

except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    import traceback

    traceback.print_exc()