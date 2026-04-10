"""Download TACO, TrashNet, and Kaggle waste datasets."""

from __future__ import annotations

from pathlib import Path


def download_taco(output_dir: str | Path) -> Path:
    """Download the TACO (Trash Annotations in Context) dataset.

    Args:
        output_dir: Directory to save the downloaded dataset.

    Returns:
        Path to the downloaded dataset root.
    """
    # TODO: Implement — clone TACO repo and download images via TACO API
    raise NotImplementedError


def download_trashnet(output_dir: str | Path) -> Path:
    """Download the TrashNet dataset (Gary Thung & Mindy Yang).

    Args:
        output_dir: Directory to save the downloaded dataset.

    Returns:
        Path to the downloaded dataset root.
    """
    # TODO: Implement — download from GitHub release or Google Drive
    raise NotImplementedError


def download_kaggle(output_dir: str | Path, dataset_slug: str = "techsash/waste-classification-data") -> Path:
    """Download a waste classification dataset from Kaggle.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.

    Args:
        output_dir: Directory to save the downloaded dataset.
        dataset_slug: Kaggle dataset identifier (owner/dataset-name).

    Returns:
        Path to the downloaded dataset root.
    """
    # TODO: Implement — use kaggle API to download and extract
    raise NotImplementedError


def download_all(output_dir: str | Path = "datasets/raw") -> dict[str, Path]:
    """Download all supported datasets.

    Args:
        output_dir: Root directory for all raw dataset downloads.

    Returns:
        Dictionary mapping dataset name to its download path.
    """
    # TODO: Implement — orchestrate all download functions
    raise NotImplementedError


if __name__ == "__main__":
    download_all()
