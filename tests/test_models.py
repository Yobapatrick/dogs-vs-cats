"""Unit tests for model forward passes and output shapes."""

from __future__ import annotations

import pytest
import torch

from src.models import CNNModel, LogRegModel, build_model

BATCH = 4
IMG_SIZE = 64
NUM_CLASSES = 2


@pytest.fixture
def dummy_batch():
    """Random images and labels."""
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, NUM_CLASSES, (BATCH,))
    return x, y


def test_logreg_forward_shape(dummy_batch):
    x, _ = dummy_batch
    model = LogRegModel(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    logits = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES)


def test_cnn_forward_shape(dummy_batch):
    x, _ = dummy_batch
    model = CNNModel(num_classes=NUM_CLASSES)
    logits = model(x)
    assert logits.shape == (BATCH, NUM_CLASSES)


def test_cnn_output_is_finite(dummy_batch):
    x, _ = dummy_batch
    model = CNNModel(num_classes=NUM_CLASSES)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert torch.isfinite(logits).all()


def test_logreg_gradient_flow(dummy_batch):
    x, y = dummy_batch
    model = LogRegModel(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)


def test_cnn_gradient_flow(dummy_batch):
    x, y = dummy_batch
    model = CNNModel(num_classes=NUM_CLASSES)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)


def test_factory_known_models():
    assert isinstance(build_model("cnn"), CNNModel)
    assert isinstance(build_model("logreg"), LogRegModel)


def test_factory_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("transformer")
