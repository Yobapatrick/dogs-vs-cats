"""Custom 4-block CNN for binary image classification."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseClassifier


class CNNModel(BaseClassifier):
    """4-block VGG-inspired CNN with BatchNorm and adaptive pooling.

    Architecture (input 3x64x64):
        Block 1:  Conv(3 -> 32)   + BN + ReLU + MaxPool   -> 32x32x32
        Block 2:  Conv(32 -> 64)  + BN + ReLU + MaxPool   -> 64x16x16
        Block 3:  Conv(64 -> 128) + BN + ReLU + MaxPool   -> 128x8x8
        Block 4:  Conv(128 -> 256)+ BN + ReLU + AdaptAvg  -> 256x1x1
        Head:     Linear(256 -> 128) + ReLU + Dropout + Linear(128 -> n)
    """

    def __init__(
        self,
        num_classes: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.4,
        max_epochs_for_scheduler: int = 15,
    ) -> None:
        super().__init__(lr=lr, num_classes=num_classes)
        self.save_hyperparameters()
        self.weight_decay = weight_decay
        self.max_epochs_for_scheduler = max_epochs_for_scheduler

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs_for_scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
