"""Batch processing for large-scale inference (10k+ images)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def batch_predict(
    model_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 32,
    confidence_threshold: float = 0.25,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run inference on a directory of images in batches.

    Args:
        model_path: Path to the model weights.
        image_dir: Directory containing images to process.
        output_dir: Directory to save result JSON and annotated images.
        batch_size: Number of images per batch.
        confidence_threshold: Minimum detection confidence.
        device: 'cpu' or 'cuda:0'.

    Returns:
        Summary dict with total_images, total_detections, per_class_counts, elapsed_seconds.
    """
    # TODO: Implement — iterate in batches, run inference, aggregate results
    raise NotImplementedError


def collect_image_paths(image_dir: str | Path, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    """Gather all image file paths from a directory.

    Args:
        image_dir: Directory to scan.
        extensions: Accepted file extensions.

    Returns:
        Sorted list of image paths.
    """
    # TODO: Implement
    raise NotImplementedError
