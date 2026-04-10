"""Preview augmented samples with bounding box overlays."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.convert import CLASS_NAMES

COLORS: list[tuple[int, int, int]] = [
    (230, 159, 0),   # cardboard
    (86, 180, 233),  # food_organics
    (0, 158, 115),   # glass
    (170, 170, 170), # metal
    (128, 128, 0),   # misc_trash
    (240, 228, 66),  # paper
    (0, 114, 178),   # plastic
    (213, 94, 0),    # textile_trash
    (0, 200, 83),    # vegetation
]


def _draw_boxes(
    image: np.ndarray,
    bboxes: list[list[float]],
    class_labels: list[int],
    class_names: list[str],
) -> np.ndarray:
    """Draw YOLO bounding boxes on a copy of the image.

    Args:
        image: BGR image.
        bboxes: List of [cx, cy, w, h] normalized entries.
        class_labels: Class index for each bbox.
        class_names: Class name strings.

    Returns:
        Copy with boxes drawn.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    for bbox, raw_cls in zip(bboxes, class_labels):
        cls_id = int(raw_cls)
        cx, cy, bw, bh = bbox
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def preview_augmented(
    image: np.ndarray,
    bboxes: list[list[float]],
    class_names: list[str],
    num_samples: int = 5,
    transform: Any | None = None,
) -> list[np.ndarray]:
    """Generate and visualize multiple augmented versions of a single image.

    Args:
        image: Source image as BGR numpy array.
        bboxes: YOLO-format bounding boxes [[class_id, cx, cy, w, h], ...].
        class_names: List of class name strings (index matches class_id).
        num_samples: Number of augmented versions to generate.
        transform: Albumentations Compose object. If None, draws original.

    Returns:
        List of augmented images with bounding boxes drawn.
    """
    class_labels = [int(b[0]) for b in bboxes]
    raw_bboxes = [[b[1], b[2], b[3], b[4]] for b in bboxes]

    results: list[np.ndarray] = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for _ in range(num_samples):
        if transform is not None:
            try:
                out = transform(
                    image=image_rgb,
                    bboxes=raw_bboxes,
                    class_labels=class_labels,
                )
                aug_bgr = cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR)
                aug_bboxes = out["bboxes"]
                aug_labels = [int(c) for c in out["class_labels"]]
            except Exception:
                aug_bgr = image.copy()
                aug_bboxes = raw_bboxes
                aug_labels = class_labels
        else:
            aug_bgr = image.copy()
            aug_bboxes = raw_bboxes
            aug_labels = class_labels

        vis = _draw_boxes(aug_bgr, aug_bboxes, aug_labels, class_names)
        results.append(vis)

    return results


def save_preview_grid(
    images: list[np.ndarray],
    output_path: str | Path,
    cols: int = 3,
    tile_size: int = 320,
) -> None:
    """Arrange images in a grid and save to disk.

    Args:
        images: List of images to tile.
        output_path: File path to save the grid image.
        cols: Number of columns in the grid.
        tile_size: Width and height of each tile in pixels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = [cv2.resize(img, (tile_size, tile_size)) for img in images]
    rows = math.ceil(len(tiles) / cols)

    while len(tiles) < rows * cols:
        tiles.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(tiles[r * cols : (r + 1) * cols]))

    grid = np.vstack(grid_rows)
    cv2.imwrite(str(output_path), grid)
