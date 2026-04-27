# Methodology

This document explains the modelling choices, the experimental setup, and how to read the results. The README is the user-facing entry point; this is the technical companion.

## 1. Problem framing

Binary image classification: given a photo, decide whether it shows a **cat** or a **dog**. The dataset is roughly balanced, so we treat the two classes symmetrically and report multi-axis metrics (accuracy, precision, recall, F1, ROC-AUC) rather than a single number.

## 2. Why two models?

We train two deliberately different models and compare them head-to-head:

1. **Logistic Regression on flattened pixels** — `nn.Linear(3·64·64, 2)`. No spatial structure, no hierarchy of features. This is the *floor*. It tells us what a model can learn from raw pixel intensities alone.
2. **Custom 4-block CNN** — convolutions, BatchNorm, ReLU, pooling, dropout, adaptive average pooling. This is the *target architecture*: same input, same loss, same trainer, but with the inductive bias that matters for images (translation equivariance, local connectivity, hierarchical features).

The gap between the two quantifies how much the CNN's architecture, **not just more parameters**, contributes to the task.

## 3. Data pipeline

- **Source**: Kaggle dataset `bhavikjikadara/dog-and-cat-classification-dataset`, fetched via `kagglehub` (see `src/data/download.py`).
- **Splits**: 70 / 15 / 15 (train / val / test), seeded with `torch.Generator().manual_seed(42)` for reproducibility.
- **Resolution**: 64×64. Deliberately small to keep training cheap on a single GPU; a clear next step is to increase to 224×224 with transfer learning.
- **Normalization**: ImageNet statistics (mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]`). Even though we don't pretrain on ImageNet, this is a sane default; in particular, it sets the data to roughly zero-mean unit-variance per channel.
- **Augmentation** (train only):
  - `RandomHorizontalFlip` — cats and dogs don't have a privileged left/right.
  - `RandomRotation(10°)` — small jitter to break exact orientation memorisation.
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` — robustness to lighting.
- Validation and test sets use the **eval transform only** (resize + normalize); no augmentation, no leakage.

## 4. Model architectures

### Logistic Regression baseline

```
Flatten(3×64×64=12288)  ->  Linear(12288 -> 2)
Optimizer: Adam, lr=1e-3
```

12,290 trainable parameters. No non-linearities, no regularisation other than implicit weight decay through Adam. This is the simplest possible learner.

### CNN

```
Block 1:  Conv(3 -> 32)   + BN + ReLU + MaxPool(2)    -> 32×32×32
Block 2:  Conv(32 -> 64)  + BN + ReLU + MaxPool(2)    -> 64×16×16
Block 3:  Conv(64 -> 128) + BN + ReLU + MaxPool(2)    -> 128×8×8
Block 4:  Conv(128 -> 256)+ BN + ReLU + AdaptiveAvg   -> 256×1×1
Head:     Flatten + Linear(256 -> 128) + ReLU + Dropout(0.4) + Linear(128 -> 2)
Optimizer: AdamW (weight_decay=1e-4), CosineAnnealingLR scheduler
```

~390k parameters. Design rationale:

- **BatchNorm after every conv** stabilises training and acts as a mild regulariser.
- **Adaptive Average Pooling** at the end means the head is decoupled from input resolution; you can swap in 96×96 images without changing the classifier.
- **Dropout(0.4)** in the head and **weight decay** in AdamW give two independent regularisers — important here because the CNN visibly overfits early (see Figure 1 in the README).
- **Cosine annealing** decays the LR smoothly over the schedule rather than via discrete steps; combined with early stopping (patience = 4 on val_loss), the model picks its own stopping point.

## 5. Training protocol

Both models share the same Lightning `Trainer`:

- Max 15 epochs, batch size 64
- Early stopping on `val_loss` (mode=min, patience=4)
- `ModelCheckpoint` saving the best `val_acc` epoch
- `LearningRateMonitor` for the cosine schedule

The shared `BaseClassifier` (`src/models/base.py`) factors the metric stack (`torchmetrics`) and the per-epoch history tracking that feeds the visualisations. This is what keeps the LogReg vs CNN comparison **strictly apples-to-apples**: same metrics object, same loss function (`F.cross_entropy`), same training loop. Only the model itself differs.

## 6. Reading the results

The four figures in `reports/figures/` tell complementary stories:

| Figure | What to look for |
|---|---|
| `fig1_training_curves.png` | LogReg train and val accuracy are stuck ~55%, hovering just above chance. The val_loss spike at epoch 5 (1.74) confirms the linear model is *unstable* on this task. The CNN reaches 90%+ accuracy by epoch 13. |
| `fig1_training_curves.png` (gap subplot) | The CNN's train−val gap spikes to ~28% at epoch 8, then *re-converges* — classic post-warmup behaviour, not pathological overfitting. The LogReg gap stays near zero because it can't learn enough to overfit. |
| `fig2_confusion.png` | LogReg confuses dogs as cats in ~47% of cases — barely above random. The CNN is symmetric (9.2% / 10.7% error rates), suggesting balanced learning. |
| `fig3_roc_probs.png` | AUC 0.965 vs 0.623. The probability histograms show the CNN's calibration: confident, bimodal at the extremes. LogReg outputs cluster near 0.5 — the linear model is unsure on most inputs. |

## 7. Reproducibility checklist

- [x] Fixed seed (`pl.seed_everything(42, workers=True)`) — see `src/utils/seed.py`
- [x] Deterministic split (seeded `torch.Generator`)
- [x] Locked dependencies in `requirements.txt`
- [x] Configs externalised in YAML
- [x] Best checkpoint saved automatically
- [ ] Deterministic CUDA ops — opt-in via `set_seed(deterministic=True)` (slower)

## 8. Known limitations and future work

1. **Single dataset, single split.** A k-fold CV pass would tighten the variance estimates on the test metrics.
2. **No transfer learning.** A frozen ResNet-18 backbone would likely push past 97% with less compute than this CNN required.
3. **No calibration analysis.** AUC 0.965 is great but the model could still be miscalibrated — a reliability diagram is a quick next step.
4. **No adversarial robustness testing.** A few simple corruptions (blur, noise, occlusion) would show how brittle the CNN really is.
5. **No deployment artifact.** Exporting to ONNX or TorchScript and serving via FastAPI is a natural extension and is sketched in the README roadmap.
