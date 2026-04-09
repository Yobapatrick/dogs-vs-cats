"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "dogs_vs_cats", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stdout with a clean format."""
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid double handlers on re-import
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
