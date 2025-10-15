# src/detection/preprocess/__init__.py

from importlib import import_module


def create(method: str, cfg: dict):
    """
    Dynamically load and instantiate a preprocessor.
    E.g. pp = create("Threshold", cfg) → loads .threshold → Threshold(cfg)
    """
    name_norm = method.strip()
    mod = import_module(f".{name_norm.lower()}", package=__name__)
    cls = getattr(mod, name_norm if name_norm[0].isupper() else name_norm.capitalize())
    return cls(cfg)
