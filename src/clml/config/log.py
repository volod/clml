"""Logging configuration for clml.

Usage in any module:
    from clml.config.log import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
import warnings
from pathlib import Path

_THIRD_PARTY_LOGGERS = (
    "mlflow",
    "optuna",
)


def configure_logging(level: str = "WARNING", log_file: Path | None = None) -> None:
    """Set up root logger for the clml package."""
    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="y contains no unlabeled samples",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="Stochastic Optimizer: Maximum iterations",
    )

    root = logging.getLogger("clml")
    root.setLevel(level.upper())
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
