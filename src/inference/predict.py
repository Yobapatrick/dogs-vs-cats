"""Single and batch inference API for trained models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models import CNNModel, LogRegModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_IMG_SIZE = 64
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_CLASS_NAMES = ("cat", "dog")


@dataclass
class PredictionOutput:
    """Result of a single image prediction."""

    label: str
    label_idx: int
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "label_idx": self.label_idx,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


class Predictor:
    """Wraps a trained model for inference on raw images.

    Usage:
        predictor = Predictor.from_checkpoint("checkpoints/cnn-best.ckpt", model_type="cnn")
        result = predictor.predict("my_photo.jpg")
        print(result.label, result.confidence)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        img_size: int = DEFAULT_IMG_SIZE,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
        normalize_mean: tuple[float, float, float] = DEFAULT_MEAN,
        normalize_std: tuple[float, float, float] = DEFAULT_STD,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        self.class_names = class_names
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, model_type: str, **kwargs
    ) -> "Predictor":
        """Load a model from a Lightning checkpoint."""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if model_type.lower() == "cnn":
            model = CNNModel.load_from_checkpoint(ckpt_path)
        elif model_type.lower() == "logreg":
            model = LogRegModel.load_from_checkpoint(ckpt_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        logger.info("Loaded %s from %s", model_type, ckpt_path)
        return cls(model=model, **kwargs)

    def predict(self, image: str | Path | Image.Image) -> PredictionOutput:
        """Predict the class of a single image."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(dim=-1).squeeze().cpu()

        conf, idx = probs.max(dim=0)
        return PredictionOutput(
            label=self.class_names[idx.item()],
            label_idx=int(idx.item()),
            confidence=float(conf.item()),
            probabilities={name: float(p) for name, p in zip(self.class_names, probs.tolist(), strict=False)},
        )

    def predict_batch(self, images: list[str | Path | Image.Image]) -> list[PredictionOutput]:
        """Predict labels for a list of images."""
        return [self.predict(img) for img in images]
