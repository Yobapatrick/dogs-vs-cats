"""Compute test-set predictions and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader


@dataclass
class PredictionResult:
    """Bundle of arrays returned by :func:`get_predictions`."""

    preds: np.ndarray
    labels: np.ndarray
    probs_pos: np.ndarray  # P(class == 1)


def get_predictions(model: torch.nn.Module, loader: DataLoader, device: str | None = None) -> PredictionResult:
    """Run a model in eval mode over a loader and return numpy arrays."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            probs = logits.softmax(dim=-1)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y)
            all_probs.append(probs[:, 1].cpu())

    return PredictionResult(
        preds=torch.cat(all_preds).numpy(),
        labels=torch.cat(all_labels).numpy(),
        probs_pos=torch.cat(all_probs).numpy(),
    )


def compute_metrics(result: PredictionResult, class_names: list[str]) -> dict:
    """Return a dict of metrics + the confusion matrix + ROC info."""
    fpr, tpr, _ = roc_curve(result.labels, result.probs_pos)
    cm = confusion_matrix(result.labels, result.preds)
    report = classification_report(
        result.labels, result.preds, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
    }
