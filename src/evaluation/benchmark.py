"""Latency and throughput benchmarking for inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def benchmark_latency(
    model_path: str | Path,
    imgsz: int = 640,
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: str = "cpu",
) -> dict[str, float]:
    """Measure single-image inference latency.

    Args:
        model_path: Path to the model weights (.pt, .onnx, or .engine).
        imgsz: Input image size.
        num_warmup: Warmup iterations (excluded from timing).
        num_iterations: Timed iterations.
        device: 'cpu' or 'cuda:0'.

    Returns:
        Dict with 'mean_ms', 'median_ms', 'p95_ms', 'p99_ms', 'fps'.
    """
    # TODO: Implement
    raise NotImplementedError


def benchmark_throughput(
    model_path: str | Path,
    batch_sizes: list[int] | None = None,
    imgsz: int = 640,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Measure throughput at various batch sizes.

    Args:
        model_path: Path to the model weights.
        batch_sizes: List of batch sizes to test. Defaults to [1, 4, 8, 16, 32].
        imgsz: Input image size.
        device: 'cpu' or 'cuda:0'.

    Returns:
        List of dicts with 'batch_size', 'images_per_second', 'latency_ms'.
    """
    # TODO: Implement
    raise NotImplementedError
