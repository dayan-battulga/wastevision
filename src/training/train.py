"""Main YOLOv8 2-stage training script.

Stage 1: Freeze backbone for N epochs (warm up detection head).
Stage 2: Unfreeze all layers for full fine-tuning with cosine annealing.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.config import load_config, merge_configs

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict[str, Any]) -> Path:
    """Run a 2-stage YOLOv8 training session.

    Stage 1: Backbone frozen for ``freeze_epochs`` epochs.
    Stage 2: All layers unfrozen for remaining epochs.

    Args:
        config: Merged configuration from model.yaml + train.yaml.

    Returns:
        Path to the best weights file from the final stage.
    """
    from ultralytics import YOLO

    tc = config.get("training", config)
    paths = config.get("paths", {})

    seed = tc.get("seed", 42)
    _set_seeds(seed)

    data_yaml = str(paths.get("data_yaml", "configs/data.yaml"))
    project = str(paths.get("project", "runs/train"))
    name = str(paths.get("name", "wastevision_v1"))

    model_cfg = config.get("model", {})
    arch = model_cfg.get("architecture", "yolov8m")
    model_file = f"{arch}.pt" if model_cfg.get("pretrained", True) else f"{arch}.yaml"

    total_epochs = tc.get("epochs", 200)
    freeze_epochs = tc.get("freeze_epochs", 5)
    freeze_layers = tc.get("freeze", 10)

    shared_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "imgsz": tc.get("imgsz", 640),
        "batch": tc.get("batch_size", 16),
        "optimizer": tc.get("optimizer", "SGD"),
        "lr0": tc.get("lr0", 0.01),
        "lrf": tc.get("lrf", 0.01),
        "momentum": tc.get("momentum", 0.937),
        "weight_decay": tc.get("weight_decay", 0.0005),
        "warmup_epochs": tc.get("warmup_epochs", 3),
        "warmup_momentum": tc.get("warmup_momentum", 0.8),
        "patience": tc.get("patience", 30),
        "amp": tc.get("amp", True),
        "mosaic": tc.get("mosaic", 1.0),
        "mixup": tc.get("mixup", 0.15),
        "close_mosaic": tc.get("close_mosaic", 20),
        "device": tc.get("device", "0"),
        "workers": tc.get("workers", 8),
        "seed": seed,
        "exist_ok": True,
        "verbose": True,
    }

    # ------------------------------------------------------------------
    # Stage 1: Frozen backbone
    # ------------------------------------------------------------------
    logger.info("=== Stage 1: Frozen backbone (%d epochs, freeze=%d layers) ===", freeze_epochs, freeze_layers)

    model = YOLO(model_file)

    from src.training.callbacks import register_callbacks
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        register_callbacks(model)

    stage1_results = model.train(
        epochs=freeze_epochs,
        freeze=freeze_layers,
        project=project,
        name=f"{name}_stage1",
        **shared_kwargs,
    )

    # Use the actual save directory from Ultralytics rather than guessing the path
    stage1_dir = Path(stage1_results.save_dir)
    stage1_best = stage1_dir / "weights" / "best.pt"
    stage1_last = stage1_dir / "weights" / "last.pt"
    checkpoint = stage1_best if stage1_best.exists() else stage1_last

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found. Checked: {stage1_best}, {stage1_last}"
        )

    logger.info("Stage 1 complete. Checkpoint: %s", checkpoint)

    # ------------------------------------------------------------------
    # Stage 2: Full fine-tuning
    # ------------------------------------------------------------------
    remaining_epochs = total_epochs - freeze_epochs
    logger.info("=== Stage 2: Full fine-tuning (%d epochs, all layers unfrozen) ===", remaining_epochs)

    model = YOLO(str(checkpoint))

    if wandb_cfg.get("enabled", False):
        register_callbacks(model)

    stage2_results = model.train(
        epochs=remaining_epochs,
        freeze=0,
        project=project,
        name=f"{name}_stage2",
        **shared_kwargs,
    )

    # Use the actual save directory from Ultralytics
    stage2_dir = Path(stage2_results.save_dir)
    stage2_best = stage2_dir / "weights" / "best.pt"
    stage2_last = stage2_dir / "weights" / "last.pt"
    final_best = stage2_best if stage2_best.exists() else stage2_last

    experiments_dir = Path("experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    final_dst = experiments_dir / "best.pt"
    shutil.copy2(final_best, final_dst)
    logger.info("Training complete. Best weights copied to %s", final_dst)

    return final_dst


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train WasteVision YOLOv8 model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Training config path")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Model config path")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    train_config = load_config(args.config)
    model_config = load_config(args.model_config)
    config = merge_configs(train_config, model_config)

    train(config)


if __name__ == "__main__":
    main()

