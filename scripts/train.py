"""Train a model from a YAML config file.

Usage:
    python scripts/train.py --config configs/cnn.yaml
    python scripts/train.py --config configs/logreg.yaml --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import run_experiment  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

logger = get_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dogs vs Cats classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to ImageFolder root. If omitted, will be downloaded from Kaggle.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    result = run_experiment(cfg, data_dir=args.data_dir)

    logger.info("Training complete.")
    logger.info("Best checkpoint: %s", result["best_ckpt"])
    logger.info("Test metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
