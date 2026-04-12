"""Shared Lightning base class for binary image classifiers."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall


class BaseClassifier(pl.LightningModule):
    """Base ``LightningModule`` factoring metrics and history tracking.

    Subclasses must implement :meth:`forward` and :meth:`configure_optimizers`.
    Logits of shape ``(N, num_classes)`` are expected.
    """

    def __init__(self, lr: float = 1e-3, num_classes: int = 2) -> None:
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        task = "binary" if num_classes == 2 else "multiclass"
        kwargs = {"task": task, "num_classes": num_classes} if task == "multiclass" else {"task": task}

        self.train_acc = Accuracy(**kwargs)
        self.val_acc = Accuracy(**kwargs)
        self.test_acc = Accuracy(**kwargs)
        self.test_f1 = F1Score(**kwargs)
        self.test_prec = Precision(**kwargs)
        self.test_rec = Recall(**kwargs)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    # --- Shared step ---
    def _step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        probs = logits.softmax(dim=-1)
        return loss, probs, y

    def training_step(self, batch, _):
        loss, probs, y = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "train_acc",
            self.train_acc(probs[:, 1], y),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, _):
        loss, probs, y = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(probs[:, 1], y), prog_bar=True)

    def test_step(self, batch, _):
        loss, probs, y = self._step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc(probs[:, 1], y))
        self.log("test_f1", self.test_f1(probs[:, 1], y))
        self.log("test_prec", self.test_prec(probs[:, 1], y))
        self.log("test_rec", self.test_rec(probs[:, 1], y))

    # --- History hooks ---
    def on_train_epoch_end(self) -> None:
        m = self.trainer.callback_metrics
        self.history["train_loss"].append(float(m.get("train_loss", torch.tensor(0.0))))
        self.history["train_acc"].append(float(m.get("train_acc", torch.tensor(0.0))))

    def on_validation_epoch_end(self) -> None:
        m = self.trainer.callback_metrics
        self.history["val_loss"].append(float(m.get("val_loss", torch.tensor(0.0))))
        self.history["val_acc"].append(float(m.get("val_acc", torch.tensor(0.0))))
