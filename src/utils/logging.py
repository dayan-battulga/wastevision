"""Logging configuration for the WasteVision project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "wastevision",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_file: Optional path to write logs to a file.

    Returns:
        Configured logger instance.
    """
    # TODO: Implement
    raise NotImplementedError
