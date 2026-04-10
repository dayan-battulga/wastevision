"""Hyperparameter search using Ultralytics built-in tuner or Ray Tune."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_search_space() -> dict[str, Any]:
    """Define the hyperparameter search space.

    Returns:
        Dictionary mapping parameter names to search ranges/distributions.
    """
    # TODO: Implement — define ranges for lr0, batch_size, augmentation params, etc.
    raise NotImplementedError


def run_hyperparam_search(
    config: dict[str, Any],
    n_trials: int = 20,
    output_dir: str | Path = "runs/tune",
) -> dict[str, Any]:
    """Run hyperparameter search and return the best configuration.

    Args:
        config: Base training config to override.
        n_trials: Number of search trials.
        output_dir: Directory to save search results.

    Returns:
        Best hyperparameter configuration found.
    """
    # TODO: Implement — use model.tune() or Ray Tune
    raise NotImplementedError
