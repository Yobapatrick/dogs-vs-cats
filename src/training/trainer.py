"""High-level training orchestration driven by a config dict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytorch_lightning as pl

from src.data import DogsVsCatsDataModule, download_dataset
from src.models import build_model
from src.training.callbacks import build_callbacks
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def _build_datamodule(cfg: dict[str, Any], data_dir: str | Path) -> DogsVsCatsDataModule:
    data_cfg = cfg["data"]
    norm = data_cfg.get("normalize", {})
    return DogsVsCatsDataModule(
        data_dir=data_dir,
        batch_size=data_cfg["batch_size"],
        img_size=data_cfg["img_size"],
        num_workers=data_cfg["num_workers"],
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        normalize_mean=tuple(norm.get("mean", [0.485, 0.456, 0.406])),
        normalize_std=tuple(norm.get("std", [0.229, 0.224, 0.225])),
        seed=cfg["experiment"]["seed"],
        augmentation=data_cfg.get("augmentation"),
    )


def _build_model(cfg: dict[str, Any]):
    mcfg = cfg["model"]
    model_type = mcfg["type"]

    if model_type == "logreg":
        return build_model(
            "logreg",
            img_size=cfg["data"]["img_size"],
            num_classes=mcfg["num_classes"],
            lr=mcfg["lr"],
        )
    if model_type == "cnn":
        return build_model(
            "cnn",
            num_classes=mcfg["num_classes"],
            lr=mcfg["lr"],
            weight_decay=mcfg.get("weight_decay", 1e-4),
            dropout=mcfg.get("dropout", 0.4),
            max_epochs_for_scheduler=cfg["trainer"]["max_epochs"],
        )
    raise ValueError(f"Unknown model type: {model_type}")


def run_experiment(cfg: dict[str, Any], data_dir: str | Path | None = None) -> dict[str, Any]:
    """Run a full train + test cycle from a config dict.

    Returns a dict with the test metrics, the best checkpoint path,
    and the per-epoch history captured during training.
    """
    set_seed(cfg["experiment"]["seed"])
    name = cfg["experiment"]["name"]
    logger.info("=== Experiment: %s ===", name)

    if data_dir is None:
        data_dir = download_dataset(cfg["data"]["dataset_name"])

    dm = _build_datamodule(cfg, data_dir)
    model = _build_model(cfg)

    callbacks = build_callbacks(name, cfg.get("callbacks", {}), cfg["paths"]["checkpoints"])

    tcfg = cfg["trainer"]
    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        accelerator=tcfg.get("accelerator", "auto"),
        precision=tcfg.get("precision", 32),
        log_every_n_steps=tcfg.get("log_every_n_steps", 20),
        callbacks=callbacks,
        enable_progress_bar=True,
        default_root_dir=cfg["paths"].get("reports", "reports"),
    )
    trainer.fit(model, dm)
    test_metrics = trainer.test(model, dm, verbose=True)[0]

    # Persist test metrics
    metrics_dir = Path(cfg["paths"]["reports"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{name}_test.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    logger.info("Test metrics saved to %s", out_path)

    best_ckpt = None
    for cb in callbacks:
        if hasattr(cb, "best_model_path"):
            best_ckpt = cb.best_model_path
            break

    return {
        "metrics": {k: float(v) for k, v in test_metrics.items()},
        "best_ckpt": best_ckpt,
        "history": model.history,
        "model": model,
        "datamodule": dm,
    }
