"""YOLOv8 data.yaml and model configuration generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.data.convert import CLASS_NAMES

logger = logging.getLogger(__name__)


def generate_data_yaml(
    dataset_dir: str | Path,
    output_path: str | Path = "configs/data.yaml",
) -> Path:
    """Generate a YOLOv8-compatible data.yaml file.

    Creates the YAML file that Ultralytics expects, referencing
    train/val/test image paths and the 9-class taxonomy.

    Args:
        dataset_dir: Root directory of the processed dataset
            (must contain images/{train,val,test} subdirs).
        output_path: Where to write the data.yaml file.

    Returns:
        Path to the generated data.yaml file.
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }

    with open(output_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Generated data.yaml at %s (path=%s, nc=%d)", output_path, dataset_dir, len(CLASS_NAMES))
    return output_path


def get_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the model kwargs for YOLO() initialization.

    Args:
        config: Parsed model.yaml configuration dict (the 'model' key contents).

    Returns:
        Dict of keyword arguments for ultralytics.YOLO().
    """
    model_section = config.get("model", config)

    arch = model_section.get("architecture", "yolov8m")
    pretrained = model_section.get("pretrained", True)

    model_file = f"{arch}.pt" if pretrained else f"{arch}.yaml"

    return {
        "model": model_file,
        "nc": model_section.get("nc", len(CLASS_NAMES)),
    }
