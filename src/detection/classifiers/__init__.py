# src/detection/classifier/__init__.py
"""
Classifier factory for 3D CNNs.
Usage:
    from detection.classifier import create
    model = create("r3d18", cfg)   # returns a torch.nn.Module on correct device
"""

from importlib import import_module


def create(method: str, cfg: dict):
    """
    Factory: create("<method>", cfg)
    Supported (now): "r3d18"
    """
    name = method.strip().lower()
    mod = import_module(".models", package=__name__)

    # Map aliases to class names defined in models.py
    aliases = {
        "r3d18": "R3D18",
        "resnet3d18": "R3D18",
        "r2plus1d18": "R2Plus1D18",
        "r(2+1)d18": "R2Plus1D18",
    }

    if name not in aliases:
        raise ValueError(f"[classifier] Unknown method '{method}'. Supported: {list(aliases.keys())}")

    cls_name = aliases[name]
    cls = getattr(mod, cls_name)
    return cls.from_config(cfg)  # convention: class provides from_config(cfg)
