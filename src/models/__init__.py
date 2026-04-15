"""Model factory and exports."""

from typing import Any

from .base import BaseClassifier
from .cnn import CNNModel
from .logreg import LogRegModel

_REGISTRY: dict[str, type[BaseClassifier]] = {
    "logreg": LogRegModel,
    "cnn": CNNModel,
}


def build_model(name: str, **kwargs: Any) -> BaseClassifier:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


__all__ = ["BaseClassifier", "CNNModel", "LogRegModel", "build_model"]
