"""Run inference on one or more images.

Usage:
    python scripts/predict.py --image dog.jpg --ckpt checkpoints/cnn-best.ckpt --model cnn
    python scripts/predict.py --image dir/ --ckpt checkpoints/cnn-best.ckpt --model cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import Predictor  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("predict")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def gather_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict cat/dog on image(s)")
    parser.add_argument("--image", type=str, required=True, help="Path to an image or directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "logreg"])
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(args.ckpt, model_type=args.model, img_size=args.img_size)
    images = gather_images(Path(args.image))
    logger.info("Running inference on %d image(s)", len(images))

    results = []
    for img_path in images:
        out = predictor.predict(img_path)
        results.append({"path": str(img_path), **out.as_dict()})
        print(f"{img_path.name:30s} -> {out.label:5s}  conf={out.confidence:.3f}")

    if args.output:
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
