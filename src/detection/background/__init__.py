# src/detection/background/__init__.py
from importlib import import_module

def create(method: str, cfg: dict):
    """
    Factory:
        bg = background.create("mog2", cfg)
        bg = background.create("MOG2", cfg)
        bg = background.create("ViBe", cfg)
    Looks for file `<method>.py` and class `<METHOD>` (upper-case),
    but also tries TitleCase and the raw name for robustness.
    """
    method_norm = method.strip().lower()  # 'mog2' | 'vibe'
    mod = import_module(f".{method_norm}", package=__name__)
    candidates = [
        method_norm.upper(),   # 'MOG2', 'VIBE'
        method_norm.title(),   # 'Mog2', 'Vibe'
        method,                # whatever caller passed
    ]
    for name in candidates:
        if hasattr(mod, name):
            cls = getattr(mod, name)
            return cls(cfg)    # instance with .apply(frame) and .reset()
    raise AttributeError(f"[background] Could not find a class for method='{method}' in module {mod.__name__}")
