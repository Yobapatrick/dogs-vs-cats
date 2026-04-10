"""LightningDataModule for the Dogs vs Cats classification task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DogsVsCatsDataModule(pl.LightningDataModule):
    """Loads images via ``torchvision.datasets.ImageFolder`` and splits them.

    The train transform applies augmentation; val/test use only resize +
    normalization to keep the evaluation deterministic.
    """

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 64,
        img_size: int = 64,
        num_workers: int = 2,
        val_split: float = 0.15,
        test_split: float = 0.15,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        seed: int = 42,
        augmentation: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_transform = self._build_train_transform(
            normalize_mean, normalize_std, augmentation or {}
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

        self.classes: list[str] | None = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _build_train_transform(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        aug: dict[str, Any],
    ) -> transforms.Compose:
        tfs: list[Any] = [transforms.Resize((self.img_size, self.img_size))]
        if aug.get("horizontal_flip", True):
            tfs.append(transforms.RandomHorizontalFlip())
        if (deg := aug.get("rotation_degrees", 10)) > 0:
            tfs.append(transforms.RandomRotation(deg))
        if (cj := aug.get("color_jitter")) is not None:
            tfs.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0.0),
                    contrast=cj.get("contrast", 0.0),
                    saturation=cj.get("saturation", 0.0),
                )
            )
        tfs.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return transforms.Compose(tfs)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        full = datasets.ImageFolder(self.data_dir, transform=self.train_transform)
        self.classes = full.classes

        n_total = len(full)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.val_split)
        n_train = n_total - n_test - n_val

        # Reproducible split via seeded generator
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full, [n_train, n_val, n_test], generator=generator
        )

        # Val and test must NOT use augmentation -> swap underlying transform
        val_pool = datasets.ImageFolder(self.data_dir, transform=self.val_transform)
        self.val_ds.dataset = val_pool
        self.test_ds.dataset = val_pool

        logger.info(
            "Classes: %s | Total: %d | Train: %d | Val: %d | Test: %d",
            self.classes,
            n_total,
            n_train,
            n_val,
            n_test,
        )

    def _make_loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_ds, shuffle=False)
