"""
Test visualization imports after reorganization
"""

import sys

sys.path.insert(0, r'D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\src')

print("Testing visualization imports...\n")

try:
    # Test main visualization module
    import utils.visualization

    print("✅ utils.visualization imports OK")

    # Test individual imports
    from utils.visualization import EvaluationPlotter

    print("✅ EvaluationPlotter imported")

    from utils.visualization import visualize_single_tubelet, visualize_batch_tubelets

    print("✅ Tubelet viewer functions imported")

    from utils.visualization import display_images_from_csv, display_all_categories

    print("✅ Image viewer functions imported")

    from utils.visualization import interactive_process_video, run_interactive_detection

    print("✅ Interactive detection functions imported")

    # List all available functions
    print("\n📦 Available visualization tools:")
    available = [item for item in dir(utils.visualization) if not item.startswith('_')]
    for item in available:
        print(f"   - {item}")

    print("\n🎉 ALL VISUALIZATION IMPORTS SUCCESSFUL!")
    print("✅ Your visualization module is ready to use!")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    import traceback

    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    import traceback

    traceback.print_exc()