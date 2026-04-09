"""Download the Dogs vs Cats dataset from Kaggle Hub."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"


def find_imagefolder_root(base_path: str | Path) -> Path:
    """Find the directory that contains class subfolders (ImageFolder layout)."""
    base_path = Path(base_path)
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for root, dirs, _ in os.walk(base_path):
        root_path = Path(root)
        candidate_dirs = [d for d in dirs if not d.startswith(".")]
        if len(candidate_dirs) < 2:
            continue
        has_images = all(
            any(f.suffix.lower() in image_exts for f in (root_path / d).iterdir())
            for d in candidate_dirs
        )
        if has_images:
            logger.info("ImageFolder root detected: %s", root_path)
            logger.info("Detected classes: %s", sorted(candidate_dirs))
            return root_path
    return base_path


def download_dataset(dataset_name: str = DEFAULT_DATASET) -> Path:
    """Download a Kaggle dataset and return the path to the ImageFolder root."""
    import kagglehub  # imported lazily so tests don't require the package

    logger.info("Downloading dataset: %s", dataset_name)
    raw_path = kagglehub.dataset_download(dataset_name)
    logger.info("Raw dataset path: %s", raw_path)
    return find_imagefolder_root(raw_path)


if __name__ == "__main__":
    path = download_dataset()
    print(path)
