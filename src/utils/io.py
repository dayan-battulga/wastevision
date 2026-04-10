"""File I/O helpers for reading/writing images, labels, and datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image from disk as a BGR numpy array.

    Args:
        path: Path to the image file.

    Returns:
        Image as a numpy array in BGR format, or None if the file
        cannot be decoded.
    """
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("Failed to read image: %s", path)
    return img


def write_image(image: np.ndarray, path: str | Path) -> None:
    """Write a numpy array image to disk.

    Args:
        image: Image as a numpy array (BGR).
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def read_yolo_label(path: str | Path) -> list[list[float]]:
    """Read a YOLO-format label file.

    Each line contains: class_id center_x center_y width height (normalized).

    Args:
        path: Path to the .txt label file.

    Returns:
        List of [class_id, cx, cy, w, h] entries.  Empty list if the
        file is empty or missing.
    """
    path = Path(path)
    if not path.exists():
        return []

    entries: list[list[float]] = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            entries.append([float(p) for p in parts])
    return entries


def write_yolo_label(annotations: list[list[float]], path: str | Path) -> None:
    """Write annotations in YOLO format to a .txt file.

    Args:
        annotations: List of [class_id, cx, cy, w, h] entries.
        path: Destination .txt file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}"
        for ann in annotations
    ]
    path.write_text("\n".join(lines) + "\n" if lines else "")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
