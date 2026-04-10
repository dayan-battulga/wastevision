"""Shared test fixtures for WasteVision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.convert import CLASS_NAMES


@pytest.fixture
def class_names() -> list[str]:
    """The 9-class WasteVision taxonomy."""
    return CLASS_NAMES


@pytest.fixture
def sample_image() -> np.ndarray:
    """A dummy 640x640 BGR image for testing."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_yolo_labels() -> list[list[float]]:
    """Sample YOLO-format annotations (class_id, cx, cy, w, h)."""
    return [
        [0, 0.5, 0.5, 0.3, 0.4],   # cardboard
        [1, 0.2, 0.3, 0.1, 0.15],   # food_organics
        [8, 0.8, 0.7, 0.2, 0.25],   # vegetation
    ]


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset directory structure for testing."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    return tmp_path
