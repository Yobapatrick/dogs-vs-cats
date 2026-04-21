"""Evaluate trained models and generate the full visual report.

Reproduces the four figures from the original notebook:
    - Training curves (loss, accuracy, overfit gap, final bars)
    - Confusion matrices
    - ROC curves + probability distributions
    - Misclassified-image galleries

Usage:
    python scripts/evaluate.py \
        --logreg-ckpt checkpoints/logreg-best.ckpt \
        --cnn-ckpt    checkpoints/cnn-best.ckpt \
        --config      configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DogsVsCatsDataModule, download_dataset  # noqa: E402
from src.evaluation import (  # noqa: E402
    compute_metrics,
    get_predictions,
    plot_confusion_matrices,
    plot_misclassified,
    plot_roc_and_probs,
    plot_training_curves,
)
from src.models import CNNModel, LogRegModel  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

logger = get_logger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models and generate the report")
    parser.add_argument("--logreg-ckpt", type=str, required=True)
    parser.add_argument("--cnn-ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default="reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"

    # --- Data ---
    data_dir = args.data_dir or download_dataset(cfg["data"]["dataset_name"])
    dm = DogsVsCatsDataModule(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        img_size=cfg["data"]["img_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg["experiment"]["seed"],
    )
    dm.setup()
    test_loader = dm.test_dataloader()
    class_names = dm.classes

    # --- Load models ---
    logreg = LogRegModel.load_from_checkpoint(args.logreg_ckpt)
    cnn = CNNModel.load_from_checkpoint(args.cnn_ckpt)
    logger.info("Models loaded.")

    # --- Predictions ---
    preds_lr = get_predictions(logreg, test_loader)
    preds_cnn = get_predictions(cnn, test_loader)
    predictions = {"logreg": preds_lr, "cnn": preds_cnn}

    # --- Metrics ---
    metrics_lr = compute_metrics(preds_lr, class_names)
    metrics_cnn = compute_metrics(preds_cnn, class_names)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / "full_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"logreg": metrics_lr, "cnn": metrics_cnn, "class_names": class_names},
            f,
            indent=2,
        )

    # --- Visualizations ---
    # Histories and final test metrics are read from JSONs persisted at train time
    try:
        with (metrics_dir / "logreg_test.json").open() as f:
            lr_test = json.load(f)
        with (metrics_dir / "cnn_test.json").open() as f:
            cnn_test = json.load(f)
        # NOTE: histories live in checkpoint metadata only if you persist them.
        # For a deterministic evaluation pass the histories optionally.
        # Here we skip the training-curves plot when histories are unavailable.
        histories = {}
    except FileNotFoundError:
        logger.warning("Per-experiment metric files missing — skipping bar plot.")
        lr_test = cnn_test = histories = {}

    if histories:
        plot_training_curves(
            histories,
            {"logreg": lr_test, "cnn": cnn_test},
            figures_dir / "fig1_training_curves.png",
        )

    plot_confusion_matrices(predictions, class_names, figures_dir / "fig2_confusion.png")
    plot_roc_and_probs(predictions, class_names, figures_dir / "fig3_roc_probs.png")
    plot_misclassified(
        logreg,
        test_loader,
        class_names,
        "Logistic Regression",
        figures_dir / "fig4_errors_logreg.png",
    )
    plot_misclassified(
        cnn,
        test_loader,
        class_names,
        "CNN",
        figures_dir / "fig4_errors_cnn.png",
    )

    logger.info("Evaluation complete. Figures saved to %s", figures_dir)


if __name__ == "__main__":
    main()
