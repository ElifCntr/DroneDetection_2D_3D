"""
Predictors for 2D and 3D inference.
"""
from .predictor_2d import Predictor2D, create_predictor_2d
from .predictor_3d import Predictor3D, create_predictor_3d

__all__ = [
    'Predictor2D',
    'Predictor3D',
    'create_predictor_2d',
    'create_predictor_3d'
]