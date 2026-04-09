"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import pytorch_lightning as pl
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every relevant RNG and (optionally) enforce deterministic cuDNN.

    Note: ``deterministic=True`` can slow down training. Use it for
    final reproducible runs, not for hyperparameter sweeps.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
