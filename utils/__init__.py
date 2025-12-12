from importlib import import_module
from typing import Any

__all__ = ["io", "visual", "dataset", "dcrnn", "baselines", "training", "stgformer"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        try:
            module = import_module(f"utils.{name}")
        except ModuleNotFoundError as exc:
            raise AttributeError(
                f"Optional dependency for 'utils.{name}' missing"
            ) from exc
        globals()[name] = module
        return module
    raise AttributeError(f"module 'utils' has no attribute '{name}'")
