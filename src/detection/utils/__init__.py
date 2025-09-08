# src/detection/utils/__init__.py
"""
Initialization for the utils package. Provides a dynamic loader for utility modules.
"""
from importlib import import_module

from .metrics import iou, precision, recall, f1_score
from .ground_truth import load_ground_truth

__all__ = [
    "iou",
    "precision",
    "recall",
    "f1_score",
    "load_ground_truth",
    "create",
]

def create(util_name: str):
    """
    Dynamically import and return a utils submodule.

    :param util_name: Name of the util module (e.g., "ground_truth" or "metrics").
    :return: The imported module object.

    Usage:
        from detection.utils import create as ut_create
        gt_mod = ut_create("ground_truth")
        boxes = gt_mod.load_ground_truth(path)
    """
    module = import_module(f".{util_name.lower()}", package=__name__)
    return module
