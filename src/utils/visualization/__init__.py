"""
Visualization utilities for drone detection project.
Includes tools for plotting evaluation results, viewing tubelets,
displaying qualitative analysis, and interactive BGS testing.
"""

from .evaluation_plots import EvaluationPlotter
from .tubelet_viewer import (
    visualize_single_tubelet,
    visualize_batch_tubelets,
    compare_tubelets,
    quick_view,
    batch_save
)
from .image_viewer import (
    display_images_from_csv,
    display_all_categories,
    display_specific_csv,
    create_comparison_grid
)
from .interactive_detect import (
    interactive_process_video,
    run_interactive_detection,
    test_bgs_interactive
)

__all__ = [
    # Evaluation plotting
    'EvaluationPlotter',

    # Tubelet visualization
    'visualize_single_tubelet',
    'visualize_batch_tubelets',
    'compare_tubelets',
    'quick_view',
    'batch_save',

    # Image viewing
    'display_images_from_csv',
    'display_all_categories',
    'display_specific_csv',
    'create_comparison_grid',

    # Interactive detection
    'interactive_process_video',
    'run_interactive_detection',
    'test_bgs_interactive',
]