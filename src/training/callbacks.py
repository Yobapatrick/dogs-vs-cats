"""Training callbacks (early stopping, checkpoint, LR monitor)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)


def build_callbacks(experiment_name: str, cfg: dict[str, Any], checkpoint_dir: str | Path):
    """Build the standard set of callbacks from a config dict."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    es_cfg = cfg.get("early_stopping", {})
    ck_cfg = cfg.get("checkpoint", {})
    lr_cfg = cfg.get("lr_monitor", {})

    return [
        EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            patience=es_cfg.get("patience", 4),
            mode=es_cfg.get("mode", "min"),
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=ck_cfg.get("monitor", "val_acc"),
            mode=ck_cfg.get("mode", "max"),
            filename=f"{experiment_name}-best-{{val_acc:.3f}}",
            save_top_k=ck_cfg.get("save_top_k", 1),
        ),
        LearningRateMonitor(logging_interval=lr_cfg.get("logging_interval", "epoch")),
    ]
