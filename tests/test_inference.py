"""Unit tests for the Predictor inference API."""

from __future__ import annotations

from PIL import Image

from src.inference import Predictor
from src.models import CNNModel


def test_predictor_returns_valid_output(tmp_path):
    model = CNNModel(num_classes=2)
    predictor = Predictor(model=model, class_names=("cat", "dog"))

    img_path = tmp_path / "dummy.jpg"
    Image.new("RGB", (200, 200), color=(180, 100, 50)).save(img_path)

    out = predictor.predict(img_path)

    assert out.label in {"cat", "dog"}
    assert 0.0 <= out.confidence <= 1.0
    assert set(out.probabilities) == {"cat", "dog"}
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-4


def test_predictor_batch(tmp_path):
    model = CNNModel(num_classes=2)
    predictor = Predictor(model=model, class_names=("cat", "dog"))

    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (200, 200), color=(i * 50, 100, 100)).save(p)
        paths.append(p)

    outputs = predictor.predict_batch(paths)
    assert len(outputs) == 3
    assert all(o.label in {"cat", "dog"} for o in outputs)


def test_predictor_accepts_pil_image():
    model = CNNModel(num_classes=2)
    predictor = Predictor(model=model, class_names=("cat", "dog"))
    img = Image.new("RGB", (200, 200), color=(0, 200, 50))
    out = predictor.predict(img)
    assert out.label in {"cat", "dog"}
