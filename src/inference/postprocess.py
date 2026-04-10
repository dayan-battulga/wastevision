"""Non-Maximum Suppression and result formatting."""

from __future__ import annotations

import numpy as np


def non_max_suppression(
    predictions: np.ndarray,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[dict[str, any]]:
    """Apply NMS to raw model output.

    Args:
        predictions: Raw model output tensor.
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for suppression.

    Returns:
        List of detection dicts with keys: bbox, confidence, class_id.
    """
    # TODO: Implement
    raise NotImplementedError


def scale_boxes(
    boxes: np.ndarray,
    original_shape: tuple[int, int],
    preprocessed_shape: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    """Rescale bounding boxes from preprocessed coordinates to original image.

    Args:
        boxes: Array of [x1, y1, x2, y2] in preprocessed image coords.
        original_shape: (height, width) of the original image.
        preprocessed_shape: (height, width) of the preprocessed image.
        padding: (pad_w, pad_h) applied during letterbox.

    Returns:
        Rescaled boxes in original image coordinates.
    """
    # TODO: Implement
    raise NotImplementedError


def format_results(
    detections: list[dict],
    class_names: list[str],
) -> list[dict[str, any]]:
    """Format raw detections into the API response structure.

    Args:
        detections: Raw detection dicts from NMS.
        class_names: List of class name strings.

    Returns:
        Formatted detection dicts with class_name included.
    """
    # TODO: Implement
    raise NotImplementedError
