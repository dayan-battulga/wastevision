#!/usr/bin/env python3
"""Generate Gate 1 visualizations: 10 random samples per class with bboxes.

For each of the 9 classes, picks 10 random images from data/processed/,
draws bounding boxes, and saves a tiled grid to notebooks/gate1_samples/.

Usage:
    python scripts/visualize_gate1.py
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.convert import CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "gate1_samples"
SAMPLES_PER_CLASS = 10
SEED = 42
TILE_SIZE = 320

COLORS: list[tuple[int, int, int]] = [
    (230, 159, 0),   # cardboard — orange
    (86, 180, 233),  # food_organics — sky blue
    (0, 158, 115),   # glass — teal
    (170, 170, 170), # metal — grey
    (128, 128, 0),   # misc_trash — olive
    (240, 228, 66),  # paper — yellow
    (0, 114, 178),   # plastic — blue
    (213, 94, 0),    # textile_trash — vermilion
    (0, 200, 83),    # vegetation — green
]


def _draw_yolo_boxes(
    image: np.ndarray,
    labels: list[list[float]],
    class_names: list[str],
) -> np.ndarray:
    """Draw YOLO bounding boxes on a copy of the image.

    Args:
        image: BGR image.
        labels: List of [class_id, cx, cy, w, h] normalized entries.
        class_names: Class name strings indexed by class_id.

    Returns:
        Copy of the image with boxes and labels drawn.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    for entry in labels:
        cls_id = int(entry[0])
        cx, cy, bw, bh = entry[1], entry[2], entry[3], entry[4]

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label_text = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def _make_grid(images: list[np.ndarray], cols: int = 5) -> np.ndarray:
    """Tile images into a grid, resizing each to TILE_SIZE x TILE_SIZE.

    Args:
        images: List of BGR images.
        cols: Number of columns.

    Returns:
        Single BGR image with all tiles arranged.
    """
    tiles = [cv2.resize(img, (TILE_SIZE, TILE_SIZE)) for img in images]
    rows = math.ceil(len(tiles) / cols)
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8))
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(tiles[r * cols : (r + 1) * cols]))
    return np.vstack(grid_rows)


def _collect_images_by_class(
    data_dir: Path,
) -> dict[int, list[tuple[Path, Path]]]:
    """Group image/label pairs by their primary class.

    Args:
        data_dir: Root with images/ and labels/ subdirs.

    Returns:
        Dict mapping class_id -> list of (image_path, label_path).
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    by_class: dict[int, list[tuple[Path, Path]]] = {i: [] for i in range(len(CLASS_NAMES))}

    for lbl_path in sorted(labels_dir.iterdir()):
        if lbl_path.suffix != ".txt":
            continue
        text = lbl_path.read_text().strip()
        if not text:
            continue

        first_line = text.splitlines()[0].strip().split()
        if len(first_line) != 5:
            continue
        class_id = int(float(first_line[0]))

        stem = lbl_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is not None and 0 <= class_id < len(CLASS_NAMES):
            by_class[class_id].append((img_path, lbl_path))

    return by_class


def main() -> None:
    """Generate and save visualization grids."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    by_class = _collect_images_by_class(PROCESSED_DIR)

    for class_id, name in enumerate(CLASS_NAMES):
        candidates = by_class[class_id]
        if not candidates:
            logger.warning("No images for class %d (%s), skipping", class_id, name)
            continue

        samples = rng.sample(candidates, min(SAMPLES_PER_CLASS, len(candidates)))

        annotated_images: list[np.ndarray] = []
        for img_path, lbl_path in samples:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            labels: list[list[float]] = []
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])

            vis = _draw_yolo_boxes(image, labels, CLASS_NAMES)
            annotated_images.append(vis)

        if not annotated_images:
            continue

        grid = _make_grid(annotated_images, cols=5)
        out_path = OUTPUT_DIR / f"{class_id}_{name}.jpg"
        cv2.imwrite(str(out_path), grid)
        logger.info("Saved %d samples for %s -> %s", len(annotated_images), name, out_path)

    logger.info("All visualizations saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
