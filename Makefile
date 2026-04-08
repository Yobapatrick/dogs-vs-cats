.PHONY: help install install-dev download train-logreg train-cnn train-all evaluate predict test lint format clean

help:
	@echo "Dogs vs Cats — available targets:"
	@echo "  install        Install runtime dependencies"
	@echo "  install-dev    Install dev + test dependencies"
	@echo "  download       Download dataset from Kaggle"
	@echo "  train-logreg   Train baseline logistic regression"
	@echo "  train-cnn      Train CNN model"
	@echo "  train-all      Train both models sequentially"
	@echo "  evaluate       Run evaluation and generate report"
	@echo "  predict        Run inference on an image (IMG=path)"
	@echo "  test           Run unit tests with coverage"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with black + ruff"
	@echo "  clean          Remove caches, logs and build artifacts"

install:
	pip install -r requirements.txt

install-dev: install
	pre-commit install

download:
	python -m src.data.download

train-logreg:
	python scripts/train.py --config configs/logreg.yaml

train-cnn:
	python scripts/train.py --config configs/cnn.yaml

train-all: train-logreg train-cnn

evaluate:
	python scripts/evaluate.py \
		--logreg-ckpt checkpoints/logreg-best.ckpt \
		--cnn-ckpt    checkpoints/cnn-best.ckpt

predict:
	python scripts/predict.py --image $(IMG) --model cnn

test:
	pytest

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

clean:
	rm -rf lightning_logs/ __pycache__/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
