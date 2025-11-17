"""
Evaluation module for 2D and 3D drone detection models.
"""
from .evaluators import (
    Evaluator2D,
    Evaluator3D,
    create_evaluator_2d,
    create_evaluator_3d
)

__all__ = [
    'Evaluator2D',
    'Evaluator3D',
    'create_evaluator_2d',
    'create_evaluator_3d'
]