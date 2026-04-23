"""Unit tests for the LightningDataModule against a tiny synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data import DogsVsCatsDataModule


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a small ImageFolder with two classes."""
    root = tmp_path / "data"
    for cls in ("cat", "dog"):
        cls_dir = root / cls
        cls_dir.mkdir(parents=True)
        for i in range(12):  # need enough samples for splits + dataloader workers
            img = Image.new("RGB", (96, 96), color=(i * 20 % 255, 100, 100))
            img.save(cls_dir / f"{cls}_{i}.jpg")
    return root


def test_setup_classes_and_split_sizes(synthetic_dataset):
    dm = DogsVsCatsDataModule(synthetic_dataset, batch_size=4, num_workers=0)
    dm.setup()
    assert sorted(dm.classes) == ["cat", "dog"]
    total = len(dm.train_ds) + len(dm.val_ds) + len(dm.test_ds)
    assert total == 24


def test_train_dataloader_batches(synthetic_dataset):
    dm = DogsVsCatsDataModule(synthetic_dataset, batch_size=4, num_workers=0)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert x.ndim == 4
    assert x.shape[1] == 3
    assert x.shape[2] == dm.img_size == 64
    assert y.ndim == 1


def test_val_and_test_use_eval_transform(synthetic_dataset):
    """val_ds and test_ds should share the eval transform, not the train one."""
    dm = DogsVsCatsDataModule(synthetic_dataset, batch_size=4, num_workers=0)
    dm.setup()
    assert dm.val_ds.dataset.transform is dm.val_transform
    assert dm.test_ds.dataset.transform is dm.val_transform


def test_split_is_reproducible(synthetic_dataset):
    """Two setups with the same seed must yield the same split."""
    dm1 = DogsVsCatsDataModule(synthetic_dataset, batch_size=4, num_workers=0, seed=42)
    dm2 = DogsVsCatsDataModule(synthetic_dataset, batch_size=4, num_workers=0, seed=42)
    dm1.setup()
    dm2.setup()
    assert list(dm1.train_ds.indices) == list(dm2.train_ds.indices)
