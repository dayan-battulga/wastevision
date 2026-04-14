"""Compute mAP, F1, precision, recall, and confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def compute_map(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mean Average Precision (mAP) at a given IoU threshold.

    Args:
        predictions: List of prediction dicts with keys: boxes, scores, labels.
        ground_truths: List of ground truth dicts with keys: boxes, labels.
        iou_threshold: IoU threshold for a true positive match.

    Returns:
        Dict with 'mAP', per-class 'AP', and 'mAP@50:95'.
    """
    # TODO: Implement
    raise NotImplementedError


def compute_f1(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute F1 score, precision, and recall per class.

    Args:
        predictions: List of prediction dicts.
        ground_truths: List of ground truth dicts.
        iou_threshold: IoU threshold for matching.

    Returns:
        Dict with per-class and macro F1, precision, recall.
    """
    # TODO: Implement
    raise NotImplementedError


def confusion_matrix(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    num_classes: int = 9,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Build a confusion matrix from predictions and ground truths.

    Args:
        predictions: List of prediction dicts.
        ground_truths: List of ground truth dicts.
        num_classes: Number of classes (9 for DyrtyVision).
        iou_threshold: IoU threshold for matching.

    Returns:
        Confusion matrix as a (num_classes+1 x num_classes+1) numpy array
        (extra row/col for background).
    """
    # TODO: Implement
    raise NotImplementedError
