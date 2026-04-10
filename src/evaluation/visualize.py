"""Prediction visualization: draw bounding boxes and confidence on images."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def draw_predictions(
    image: np.ndarray,
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    class_names: list[str],
    confidence_threshold: float = 0.25,
) -> np.ndarray:
    """Draw bounding boxes with labels and scores on an image.

    Args:
        image: BGR numpy array.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        labels: Class indices for each box.
        scores: Confidence scores for each box.
        class_names: List of class name strings.
        confidence_threshold: Minimum score to draw a box.

    Returns:
        Image with drawn predictions.
    """
    # TODO: Implement
    raise NotImplementedError


def save_prediction_grid(
    images: list[np.ndarray],
    output_path: str | Path,
    cols: int = 4,
) -> None:
    """Arrange prediction images in a grid and save to disk.

    Args:
        images: List of annotated images.
        output_path: Destination file path.
        cols: Number of grid columns.
    """
    # TODO: Implement
    raise NotImplementedError


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array from metrics.confusion_matrix().
        class_names: List of class name strings.
        output_path: File path to save the plot.
        normalize: Whether to normalize values to percentages.
    """
    # TODO: Implement
    raise NotImplementedError
