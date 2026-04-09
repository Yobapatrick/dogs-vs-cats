"""YAML config loader with simple inheritance support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``. Override wins on conflicts."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving the optional top-level ``defaults`` key.

    Example:
        # cnn.yaml
        defaults: base.yaml
        model: { ... }

    The base file is loaded first, then keys from the child override.
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "defaults" in cfg:
        base_path = (path.parent / cfg.pop("defaults")).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg
