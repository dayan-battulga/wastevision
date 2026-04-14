"""Custom Weights & Biases callbacks for YOLOv8 training.

All wandb calls are guarded so training works without wandb installed.
Images are logged every 10 epochs to avoid throttling (per CLAUDE.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOG_IMAGE_INTERVAL = 10


def _get_wandb() -> Any | None:
    """Import wandb if available, return None otherwise."""
    try:
        import wandb

        if wandb.run is None:
            return None
        return wandb
    except ImportError:
        return None


def on_train_epoch_end(trainer: Any) -> None:
    """Log custom metrics at the end of each training epoch.

    Args:
        trainer: Ultralytics trainer object.
    """
    wandb = _get_wandb()
    if wandb is None:
        return

    epoch = trainer.epoch
    metrics: dict[str, Any] = {}

    if hasattr(trainer, "lr") and trainer.lr:
        for i, lr_val in enumerate(trainer.lr.values()) if isinstance(trainer.lr, dict) else enumerate(trainer.lr):
            metrics[f"lr/pg{i}"] = lr_val

    if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        for name, val in zip(loss_names, trainer.loss_items):
            metrics[f"train/{name}"] = float(val)

    if metrics:
        wandb.log(metrics, step=epoch)


def on_val_end(validator: Any) -> None:
    """Log validation results and sample predictions.

    Args:
        validator: Ultralytics validator object.
    """
    wandb = _get_wandb()
    if wandb is None:
        return

    metrics: dict[str, Any] = {}

    if hasattr(validator, "metrics"):
        m = validator.metrics
        if hasattr(m, "box"):
            metrics["val/mAP50"] = float(m.box.map50)
            metrics["val/mAP50-95"] = float(m.box.map)
            metrics["val/precision"] = float(m.box.mp)
            metrics["val/recall"] = float(m.box.mr)

            if hasattr(m.box, "ap_class_index") and hasattr(m.box, "ap50"):
                for cls_idx, ap_val in zip(m.box.ap_class_index, m.box.ap50):
                    metrics[f"val/AP50_class{int(cls_idx)}"] = float(ap_val)

    if metrics:
        wandb.log(metrics)

    trainer = getattr(validator, "trainer", None)
    epoch = getattr(trainer, "epoch", 0) if trainer else 0

    if epoch > 0 and epoch % _LOG_IMAGE_INTERVAL == 0:
        _log_confusion_matrix(validator)


def _log_confusion_matrix(validator: Any) -> None:
    """Log confusion matrix as an image artifact."""
    wandb = _get_wandb()
    if wandb is None:
        return

    save_dir = getattr(validator, "save_dir", None)
    if save_dir is None:
        return

    cm_path = Path(save_dir) / "confusion_matrix_normalized.png"
    if not cm_path.exists():
        cm_path = Path(save_dir) / "confusion_matrix.png"

    if cm_path.exists():
        wandb.log({"val/confusion_matrix": wandb.Image(str(cm_path))})


def on_train_end(trainer: Any) -> None:
    """Log final artifacts when training completes.

    Args:
        trainer: Ultralytics trainer object.
    """
    wandb = _get_wandb()
    if wandb is None:
        return

    save_dir = Path(trainer.save_dir) if hasattr(trainer, "save_dir") else None
    if save_dir is None:
        return

    best_pt = save_dir / "weights" / "best.pt"
    if best_pt.exists():
        artifact = wandb.Artifact(
            name=f"dyrtyvision-best-{wandb.run.id}",
            type="model",
        )
        artifact.add_file(str(best_pt))
        wandb.log_artifact(artifact)
        logger.info("Logged best.pt as wandb artifact")

    for plot_name in ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png"]:
        plot_path = save_dir / plot_name
        if plot_path.exists():
            wandb.log({f"results/{plot_name}": wandb.Image(str(plot_path))})


def register_callbacks(model: Any) -> None:
    """Attach all custom callbacks to a YOLO model.

    Args:
        model: Ultralytics YOLO model instance.
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning("wandb not installed, skipping callback registration")
        return

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_end", on_train_end)
    logger.info("Registered wandb callbacks")
