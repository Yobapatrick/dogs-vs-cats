"""Baseline: logistic regression on flattened pixels."""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseClassifier


class LogRegModel(BaseClassifier):
    """Single linear layer on the flattened image.

    This is the simplest possible baseline. Its purpose is to quantify
    how much *spatial reasoning* the CNN actually contributes versus
    raw pixel correlations.
    """

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 3,
        num_classes: int = 2,
        lr: float = 1e-3,
    ) -> None:
        super().__init__(lr=lr, num_classes=num_classes)
        self.save_hyperparameters()
        self.input_dim = in_channels * img_size * img_size
        self.model = nn.Linear(self.input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.size(0), -1))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
