"""Export trained YOLOv8 models to ONNX and TensorRT formats."""

from __future__ import annotations

from pathlib import Path


def export_onnx(
    weights_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
    dynamic: bool = True,
    simplify: bool = True,
) -> Path:
    """Export a YOLOv8 model to ONNX format.

    Args:
        weights_path: Path to the .pt weights file.
        output_path: Destination for the .onnx file. Defaults to same dir as weights.
        imgsz: Input image size.
        dynamic: Whether to use dynamic batch dimensions.
        simplify: Whether to simplify the ONNX graph.

    Returns:
        Path to the exported .onnx file.
    """
    # TODO: Implement
    raise NotImplementedError


def export_tensorrt(
    weights_path: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
    half: bool = True,
) -> Path:
    """Export a YOLOv8 model to TensorRT engine format.

    Args:
        weights_path: Path to the .pt weights file.
        output_path: Destination for the .engine file. Defaults to same dir as weights.
        imgsz: Input image size.
        half: Use FP16 quantization.

    Returns:
        Path to the exported .engine file.
    """
    # TODO: Implement
    raise NotImplementedError
