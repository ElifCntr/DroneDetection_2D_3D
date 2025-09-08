"""
Main execution script for R3D-18 frame-level evaluation.
Simple script that runs the complete evaluation pipeline.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.r3d18_evaluator import R3D18FrameEvaluator


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(description='R3D-18 Frame-Level Evaluation')

    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained R3D-18 model (.pth file)')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test_index.csv file')
    parser.add_argument('--annotations_dir', type=str, required=True,
                       help='Directory containing ground truth annotation files')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Confidence thresholds to evaluate (default: 0.3 0.4 0.5 0.6 0.7 0.8 0.9)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference (default: auto)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip creating visualization plots')
    parser.add_argument('--no_save', action='store_true',
                       help='Skip saving result files')

    args = parser.parse_args()

    print("🚀 Starting R3D-18 Frame-Level Evaluation")
    print("="*50)
    print(f"Model: {args.model_path}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Annotations: {args.annotations_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Device: {args.device}")
    print("="*50)

    # Create evaluator
    evaluator = R3D18FrameEvaluator(
        model_path=args.model_path,
        test_csv_path=args.test_csv,
        annotations_dir=args.annotations_dir,
        device=args.device
    )

    # Run evaluation
    try:
        results = evaluator.run_complete_evaluation(
            thresholds=args.thresholds,
            create_plots=not args.no_plots,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )

        # Print summary
        best_threshold = results['best_threshold']
        best_f1 = results['best_metrics']['f1_score']

        print(f"\n🎯 EVALUATION SUMMARY:")
        print(f"   Best Threshold: {best_threshold:.1f}")
        print(f"   Best F1-Score: {best_f1:.4f}")
        print(f"   Results saved to: {args.output_dir}")

        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def quick_evaluation_example():
    """Example function showing how to run evaluation programmatically."""

    # Example configuration - update these paths
    MODEL_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\models\r3d18_static_t3_20250828_011134\best.pth"
    TEST_CSV_PATH = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\tubelets\test_index.csv"
    ANNOTATIONS_DIR = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\data\annotations"

    print("Running quick evaluation example...")

    # Create evaluator (without test_videos_path for now)
    evaluator = R3D18FrameEvaluator(
        model_path=MODEL_PATH,
        test_csv_path=TEST_CSV_PATH,
        annotations_dir=ANNOTATIONS_DIR
    )

    # Manually set the test videos path
    evaluator.test_videos_path = r"D:\Elif\Sussex-PhD\Python_Projects\DroneDetection\splits\test_videos.txt"

    # Run with default settings
    results = evaluator.run_complete_evaluation()

    return results


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        main()
    else:
        print("No command line arguments provided.")
        print("Running quick evaluation example...")
        print("For full usage, run with --help")
        print("-" * 50)

        # Run example (you can comment this out if not needed)
        try:
            results = quick_evaluation_example()
        except Exception as e:
            print(f"Example failed: {e}")
            print("\nTo run with custom parameters, use:")
            print("python scripts/run_frame_evaluation.py --model_path MODEL --test_csv CSV --annotations_dir DIR")