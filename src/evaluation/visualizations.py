"""Plotting routines that mirror the four notebook figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

from .metrics import PredictionResult

# --- Visual style ---
PALETTE = {"blue": "#3b82f6", "orange": "#f97316", "red": "#ef4444", "gray": "#94a3b8"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#f8fafc",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _trim_zeros(values: list[float]) -> list[float]:
    return [v for v in values if v != 0]


def plot_training_curves(
    histories: dict[str, dict[str, list[float]]],
    test_results: dict[str, dict[str, float]],
    save_path: str | Path,
) -> Path:
    """Reproduces Figure 1: 2x2 grid (loss, accuracy, overfit gap, final bars)."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves — Dogs vs Cats", fontsize=15, fontweight="bold")

    colors = {"logreg": PALETTE["blue"], "cnn": PALETTE["orange"]}
    pretty = {"logreg": "LogReg", "cnn": "CNN"}

    # 1A — Loss
    ax = axes[0, 0]
    for name, hist in histories.items():
        tl = _trim_zeros(hist["train_loss"])
        vl = _trim_zeros(hist["val_loss"])
        n = min(len(tl), len(vl))
        x = range(1, n + 1)
        ax.plot(x, tl[:n], color=colors[name], lw=2, label=f"{pretty[name]} Train")
        ax.plot(x, vl[:n], color=colors[name], lw=2, ls="--", alpha=0.65, label=f"{pretty[name]} Val")
    ax.set_title("Loss — Train vs Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=9)

    # 1B — Accuracy
    ax = axes[0, 1]
    for name, hist in histories.items():
        ta = _trim_zeros(hist["train_acc"])
        va = _trim_zeros(hist["val_acc"])
        n = min(len(ta), len(va))
        x = range(1, n + 1)
        ax.plot(x, ta[:n], color=colors[name], lw=2, label=f"{pretty[name]} Train")
        ax.plot(x, va[:n], color=colors[name], lw=2, ls="--", alpha=0.65, label=f"{pretty[name]} Val")
    ax.set_title("Accuracy — Train vs Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)

    # 1C — Overfitting gap
    ax = axes[1, 0]
    for name, hist in histories.items():
        ta = _trim_zeros(hist["train_acc"])
        va = _trim_zeros(hist["val_acc"])
        n = min(len(ta), len(va))
        gap = [ta[i] - va[i] for i in range(n)]
        x = range(1, n + 1)
        ax.fill_between(x, gap, alpha=0.25, color=colors[name])
        ax.plot(x, gap, color=colors[name], lw=2, label=pretty[name])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Overfitting Gap (Train − Val Accuracy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.legend()

    # 1D — Final metrics
    ax = axes[1, 1]
    labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    keys = ["test_acc", "test_f1", "test_prec", "test_rec"]
    x_pos = np.arange(len(labels))
    w = 0.32
    for i, name in enumerate(["logreg", "cnn"]):
        vals = [test_results[name].get(k, 0) for k in keys]
        offset = -w / 2 if i == 0 else w / 2
        bars = ax.bar(
            x_pos + offset, vals, w, color=colors[name], alpha=0.85, label=pretty[name], zorder=3
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{b.get_height():.3f}",
                ha="center",
                fontsize=8.5,
                color=colors[name],
                fontweight="bold",
            )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.18)
    ax.set_title("Final Metrics — Test Set")
    ax.set_ylabel("Score")
    ax.legend()

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrices(
    predictions: dict[str, PredictionResult],
    class_names: list[str],
    save_path: str | Path,
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold")

    style_for = {
        "logreg": ("Logistic Regression", "Blues", PALETTE["blue"]),
        "cnn": ("CNN", "Oranges", PALETTE["orange"]),
    }
    for ax, (name, pred) in zip(axes, predictions.items(), strict=False):
        title, cmap, color = style_for[name]
        cm = confusion_matrix(pred.labels, pred.preds)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annots = np.array(
            [[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
        )
        sns.heatmap(
            cm_pct,
            annot=annots,
            fmt="",
            cmap=cmap,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=1.5,
            linecolor="white",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "% of true class"},
        )
        ax.set_title(title, fontweight="bold", fontsize=12, color=color)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_and_probs(
    predictions: dict[str, PredictionResult],
    class_names: list[str],
    save_path: str | Path,
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Prediction Probability Analysis", fontsize=14, fontweight="bold")

    # ROC
    ax = axes[0]
    for name, color in [("logreg", PALETTE["blue"]), ("cnn", PALETTE["orange"])]:
        pred = predictions[name]
        fpr, tpr, _ = roc_curve(pred.labels, pred.probs_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{name.upper()} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    # Probability distributions
    ax = axes[1]
    distrib_style = {
        "logreg": ("LogReg", "#93c5fd", "#fdba74", 0.55, 1),
        "cnn": ("CNN", "#3b82f6", "#f97316", 0.85, 2),
    }
    for name, (label, c_pos, c_neg, alpha, lw) in distrib_style.items():
        pred = predictions[name]
        ax.hist(
            pred.probs_pos[pred.labels == 1],
            bins=40,
            density=True,
            alpha=alpha,
            color=c_pos,
            label=f"{label} — {class_names[1]}",
            histtype="stepfilled",
            lw=lw,
            edgecolor=c_pos,
        )
        ax.hist(
            pred.probs_pos[pred.labels == 0],
            bins=40,
            density=True,
            alpha=alpha,
            color=c_neg,
            label=f"{label} — {class_names[0]}",
            histtype="stepfilled",
            lw=lw,
            edgecolor=c_neg,
        )
    ax.axvline(0.5, color="black", lw=1.5, ls="--", label="Threshold = 0.5")
    ax.set_xlabel("Predicted probability (positive class)")
    ax.set_ylabel("Density")
    ax.set_title("Probability distribution by true class")
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_misclassified(
    model: torch.nn.Module,
    loader,
    class_names: list[str],
    title: str,
    save_path: str | Path,
    n: int = 12,
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Path | None:
    """Gallery of misclassified images for qualitative error analysis."""
    apply_style()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    wrong_imgs, wrong_pred, wrong_true = [], [], []
    mean = torch.tensor(normalize_mean).view(3, 1, 1)
    std = torch.tensor(normalize_std).view(3, 1, 1)

    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(dim=1).cpu()
            mask = preds != y
            if mask.any():
                wrong_imgs.extend(x[mask])
                wrong_pred.extend(preds[mask].tolist())
                wrong_true.extend(y[mask].tolist())
            if len(wrong_imgs) >= n:
                break

    n = min(n, len(wrong_imgs))
    if n == 0:
        return None

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    fig.suptitle(f"Misclassified Images — {title}", fontsize=13, fontweight="bold")
    axes = np.atleast_2d(axes).flatten()

    for i in range(n):
        img = wrong_imgs[i] * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {class_names[wrong_true[i]]}\nPredicted: {class_names[wrong_pred[i]]}",
            color=PALETTE["red"],
            fontsize=9,
            fontweight="bold",
        )
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path
