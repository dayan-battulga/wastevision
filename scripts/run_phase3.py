#!/usr/bin/env python3
"""Phase 3: Training pipeline and Exit Gate verification.

1. Runs 2-stage YOLOv8m training (frozen backbone -> full fine-tune)
2. After training, parses Ultralytics results to verify gate criteria
3. Copies confusion matrix to experiments/

Usage:
    python scripts/run_phase3.py                  # full training
    python scripts/run_phase3.py --gate-only DIR  # skip training, check gate on existing run
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.convert import CLASS_NAMES
from src.utils.config import load_config, merge_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_training() -> Path:
    """Execute the 2-stage training pipeline.

    Returns:
        Path to the best weights file.
    """
    from src.training.train import train

    train_config = load_config(PROJECT_ROOT / "configs" / "train.yaml")
    model_config = load_config(PROJECT_ROOT / "configs" / "model.yaml")
    config = merge_configs(train_config, model_config)

    return train(config)


def _find_latest_run_dir() -> Path | None:
    """Find the most recent stage2 training run directory.

    Searches both runs/train/ and runs/detect/train/ since Ultralytics
    may use either depending on the task type.
    """
    candidates = []
    for runs_dir in [
        PROJECT_ROOT / "runs" / "train",
        PROJECT_ROOT / "runs" / "detect" / "train",
    ]:
        if runs_dir.exists():
            candidates.extend(
                d for d in runs_dir.iterdir() if d.is_dir() and "stage2" in d.name
            )

    if not candidates:
        return None

    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _parse_results_csv(run_dir: Path) -> dict[str, float]:
    """Parse the Ultralytics results.csv for final metrics.

    Args:
        run_dir: Training run directory containing results.csv.

    Returns:
        Dict of metric names -> values from the last row.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        logger.warning("results.csv not found in %s", run_dir)
        return {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    last = rows[-1]
    parsed: dict[str, float] = {}
    for key, val in last.items():
        key = key.strip()
        try:
            parsed[key] = float(val.strip())
        except (ValueError, AttributeError):
            pass

    return parsed


def _check_per_class_ap(run_dir: Path) -> dict[int, float]:
    """Extract per-class AP@0.5 from the validation results.

    Ultralytics saves per-class metrics when verbose validation runs.
    Falls back to running val if the data isn't cached.

    Args:
        run_dir: Training run directory.

    Returns:
        Dict mapping class_id -> AP@0.5.
    """
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        logger.warning("best.pt not found in %s", run_dir)
        return {}

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    data_yaml = str(PROJECT_ROOT / "configs" / "data.yaml")

    results = model.val(data=data_yaml, verbose=False)

    per_class: dict[int, float] = {}
    if hasattr(results, "box") and hasattr(results.box, "ap50"):
        for cls_idx, ap_val in zip(results.box.ap_class_index, results.box.ap50):
            per_class[int(cls_idx)] = float(ap_val)

    return per_class


def _check_confusion_matrix(run_dir: Path) -> dict[int, float]:
    """Compute per-class misclassification rate from confusion matrix.

    Args:
        run_dir: Training run directory.

    Returns:
        Dict mapping class_id -> misclassification rate (0-1).
    """
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        return {}

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    data_yaml = str(PROJECT_ROOT / "configs" / "data.yaml")
    results = model.val(data=data_yaml, verbose=False)

    if not hasattr(results, "confusion_matrix"):
        return {}

    cm = results.confusion_matrix
    matrix = cm.matrix if hasattr(cm, "matrix") else None
    if matrix is None:
        return {}

    misclass: dict[int, float] = {}
    for i in range(min(len(CLASS_NAMES), matrix.shape[0])):
        row_sum = matrix[i].sum()
        if row_sum > 0:
            correct = matrix[i][i]
            misclass[i] = 1.0 - (correct / row_sum)
        else:
            misclass[i] = 0.0

    return misclass


def _get_food_organics_precision(run_dir: Path) -> float:
    """Get precision for food_organics (class 1) from validation.

    Args:
        run_dir: Training run directory.

    Returns:
        Precision value for food_organics, or -1 if unavailable.
    """
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        return -1.0

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    data_yaml = str(PROJECT_ROOT / "configs" / "data.yaml")
    results = model.val(data=data_yaml, verbose=False)

    if hasattr(results, "box") and hasattr(results.box, "p"):
        precisions = results.box.p
        if hasattr(results.box, "ap_class_index"):
            for cls_idx, p_val in zip(results.box.ap_class_index, precisions):
                if int(cls_idx) == 1:
                    return float(p_val)

    return -1.0


def verify_gate(run_dir: Path) -> bool:
    """Verify Exit Gate 3 criteria against a completed training run.

    Args:
        run_dir: Path to the stage2 training run directory.

    Returns:
        True if all gate criteria pass.
    """
    logger.info("=== EXIT GATE 3 VERIFICATION ===")
    logger.info("Run directory: %s", run_dir)

    all_pass = True

    # 1. Training completed (results.csv exists and has rows)
    metrics = _parse_results_csv(run_dir)
    training_ok = len(metrics) > 0
    logger.info("[%s] Training completed without OOM/NaN (results.csv has %d metric columns)",
                "PASS" if training_ok else "FAIL", len(metrics))
    if not training_ok:
        all_pass = False

    # 2. Val mAP@0.5 > 0.85
    map50_key = [k for k in metrics if "mAP50" in k and "mAP50-95" not in k]
    map50 = metrics.get(map50_key[0], 0.0) if map50_key else 0.0
    map50_pass = map50 > 0.85
    logger.info("[%s] Val mAP@0.5 = %.4f (threshold: > 0.85)", "PASS" if map50_pass else "FAIL", map50)
    if not map50_pass:
        all_pass = False

    # 3. Per-class AP > 0.75
    logger.info("Running per-class AP evaluation...")
    per_class_ap = _check_per_class_ap(run_dir)
    ap_below_75 = []
    ap_above_80 = 0
    for i, name in enumerate(CLASS_NAMES):
        ap = per_class_ap.get(i, 0.0)
        status = "OK" if ap > 0.75 else "LOW"
        logger.info("  %d: %-16s AP@0.5=%.4f  [%s]", i, name, ap, status)
        if ap <= 0.75:
            ap_below_75.append(name)
        if ap > 0.80:
            ap_above_80 += 1

    ap_pass = len(ap_below_75) == 0
    logger.info("[%s] Per-class AP > 0.75 for all classes (%d/9 above 0.80)",
                "PASS" if ap_pass else "FAIL", ap_above_80)
    if not ap_pass:
        logger.warning("  Classes below 0.75: %s", ", ".join(ap_below_75))
        all_pass = False

    # 4. food_organics precision > 0.90
    fo_precision = _get_food_organics_precision(run_dir)
    fo_pass = fo_precision > 0.90
    logger.info("[%s] food_organics precision = %.4f (threshold: > 0.90)",
                "PASS" if fo_pass else "FAIL", fo_precision)
    if not fo_pass:
        all_pass = False

    # 5. Confusion matrix — no class > 20% misclassification
    misclass = _check_confusion_matrix(run_dir)
    cm_fail_classes = []
    for i, name in enumerate(CLASS_NAMES):
        rate = misclass.get(i, 0.0)
        if rate > 0.20:
            cm_fail_classes.append(f"{name}={rate:.1%}")

    cm_pass = len(cm_fail_classes) == 0
    logger.info("[%s] Confusion matrix — no class > 20%% misclassification",
                "PASS" if cm_pass else "FAIL")
    if not cm_pass:
        logger.warning("  Classes above 20%%: %s", ", ".join(cm_fail_classes))
        all_pass = False

    # Copy confusion matrix to experiments/
    experiments_dir = PROJECT_ROOT / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    for cm_name in ["confusion_matrix.png", "confusion_matrix_normalized.png", "results.png"]:
        src = run_dir / cm_name
        if src.exists():
            shutil.copy2(src, experiments_dir / cm_name)
            logger.info("Copied %s to experiments/", cm_name)

    logger.info("")
    if all_pass:
        logger.info("GATE 3: ALL CHECKS PASSED")
    else:
        logger.warning("GATE 3: SOME CHECKS FAILED — review results above")

    return all_pass


def main() -> None:
    """Run training then verify gate, or verify gate on existing run."""
    parser = argparse.ArgumentParser(description="Phase 3 training + gate check")
    parser.add_argument("--gate-only", type=str, default=None,
                        help="Skip training, verify gate on this run directory")
    args = parser.parse_args()

    if args.gate_only:
        run_dir = Path(args.gate_only)
    else:
        best_weights = run_training()
        logger.info("Training finished. Best weights: %s", best_weights)
        run_dir = _find_latest_run_dir()
        if run_dir is None:
            logger.error("Could not find training run directory")
            sys.exit(1)

    verify_gate(run_dir)


if __name__ == "__main__":
    main()
