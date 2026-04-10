"""OpenCV preprocessing pipeline for inference."""

from __future__ import annotations

import numpy as np


def preprocess_image(
    image: np.ndarray,
    target_size: int = 640,
) -> np.ndarray:
    """Resize and normalize an image for YOLOv8 inference.

    Args:
        image: Raw BGR image as numpy array.
        target_size: Target square dimension.

    Returns:
        Preprocessed image tensor (CHW, float32, normalized).
    """
    # TODO: Implement — letterbox resize, BGR->RGB, normalize, HWC->CHW
    raise NotImplementedError


def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with letterboxing (aspect-ratio-preserving padding).

    Args:
        image: Input BGR image.
        target_size: Target square dimension.
        color: Padding color (BGR).

    Returns:
        Tuple of (padded_image, scale_ratio, (pad_w, pad_h)).
    """
    # TODO: Implement
    raise NotImplementedError
