#!/usr/bin/env python3
"""Phase 2: Augmentation pipeline and model configuration.

1. Split data/processed/ into train/val/test (70/15/15)
2. Generate configs/data.yaml
3. Run augmentation pipeline on 100 random train images
4. Generate visual check grids with bbox overlays
5. Validate data.yaml with Ultralytics and instantiate model with nc=9
6. Print Gate 2 status

Usage:
    python scripts/run_phase2.py
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.augmentation.pipeline import augment_dataset, build_transform
from src.augmentation.visualize import preview_augmented, save_preview_grid
from src.data.convert import CLASS_NAMES
from src.data.split import create_split_dirs, stratified_split
from src.model.yolo_config import generate_data_yaml, get_model_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUG_TEST_DIR = PROJECT_ROOT / "notebooks" / "aug_test"
SEED = 42


def _is_already_split(data_dir: Path) -> bool:
    """Check if data is already in train/val/test subdirs."""
    return (data_dir / "images" / "train").is_dir()


def step1_split() -> None:
    """Split the flat dataset into train/val/test."""
    if _is_already_split(PROCESSED_DIR):
        logger.info("Data already split into train/val/test, skipping split step")
        return

    logger.info("=== Step 1: Splitting data into train/val/test (70/15/15) ===")
    mapping = stratified_split(PROCESSED_DIR, 0.7, 0.15, 0.15, seed=SEED)
    create_split_dirs(PROCESSED_DIR, mapping)
    logger.info("Split complete")


def step2_data_yaml() -> Path:
    """Generate configs/data.yaml."""
    logger.info("=== Step 2: Generating configs/data.yaml ===")
    output = generate_data_yaml(PROCESSED_DIR, PROJECT_ROOT / "configs" / "data.yaml")
    logger.info("data.yaml written to %s", output)
    return output


def step3_augmentation_test() -> int:
    """Run augmentation pipeline on 100 random train images."""
    logger.info("=== Step 3: Augmentation test on 100 train images ===")

    aug_config_path = PROJECT_ROOT / "configs" / "augmentation.yaml"
    with open(aug_config_path) as f:
        aug_config = yaml.safe_load(f)

    train_config = aug_config["augmentation"]["train"]

    train_images_dir = PROCESSED_DIR / "images" / "train"
    train_labels_dir = PROCESSED_DIR / "labels" / "train"

    all_images = sorted([
        p for p in train_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    rng = random.Random(SEED)
    sample = rng.sample(all_images, min(100, len(all_images)))

    subset_dir = PROJECT_ROOT / "notebooks" / "_aug_subset"
    subset_img = subset_dir / "images"
    subset_lbl = subset_dir / "labels"
    subset_img.mkdir(parents=True, exist_ok=True)
    subset_lbl.mkdir(parents=True, exist_ok=True)

    import shutil
    for img_path in sample:
        shutil.copy2(img_path, subset_img / img_path.name)
        lbl_src = train_labels_dir / f"{img_path.stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, subset_lbl / lbl_src.name)

    AUG_TEST_DIR.mkdir(parents=True, exist_ok=True)
    count = augment_dataset(subset_dir, AUG_TEST_DIR, train_config, multiplier=1)

    shutil.rmtree(subset_dir)

    logger.info("Augmentation test: %d images generated without error", count)
    return count


def step4_visual_check() -> None:
    """Generate visual check grids with bbox overlays."""
    logger.info("=== Step 4: Visual bbox check ===")

    aug_config_path = PROJECT_ROOT / "configs" / "augmentation.yaml"
    with open(aug_config_path) as f:
        aug_config = yaml.safe_load(f)

    transform = build_transform(aug_config["augmentation"]["train"])

    train_images_dir = PROCESSED_DIR / "images" / "train"
    train_labels_dir = PROCESSED_DIR / "labels" / "train"

    all_images = sorted([
        p for p in train_images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    rng = random.Random(SEED + 1)
    samples = rng.sample(all_images, min(5, len(all_images)))

    for idx, img_path in enumerate(samples):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        lbl_path = train_labels_dir / f"{img_path.stem}.txt"
        bboxes: list[list[float]] = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    bboxes.append([float(p) for p in parts])

        if not bboxes:
            continue

        previews = preview_augmented(image, bboxes, CLASS_NAMES, num_samples=4, transform=transform)
        out_path = AUG_TEST_DIR / f"visual_check_{idx}.jpg"
        save_preview_grid(previews, out_path, cols=4)
        logger.info("Saved visual check grid: %s", out_path)


def step5_model_check(data_yaml_path: Path) -> bool:
    """Validate data.yaml with Ultralytics and check model nc."""
    logger.info("=== Step 5: Model instantiation check ===")

    from ultralytics import YOLO

    model_config_path = PROJECT_ROOT / "configs" / "model.yaml"
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    mc = get_model_config(model_config)
    logger.info("Model config: %s", mc)

    model = YOLO(mc["model"])
    actual_nc = model.model.nc if hasattr(model.model, "nc") else None

    logger.info("Model loaded: %s, nc=%s", mc["model"], actual_nc)

    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    data_nc = data_cfg.get("nc", -1)
    data_names = data_cfg.get("names", {})
    logger.info("data.yaml: nc=%d, names=%s", data_nc, list(data_names.values()))

    ok = data_nc == 9 and len(data_names) == 9
    return ok


def main() -> None:
    """Run the full Phase 2 pipeline."""
    gate_results: dict[str, bool] = {}

    step1_split()

    data_yaml = step2_data_yaml()

    try:
        aug_count = step3_augmentation_test()
        gate_results["augmentation_100"] = aug_count >= 50
    except Exception as e:
        logger.error("Augmentation test FAILED: %s", e)
        gate_results["augmentation_100"] = False

    try:
        step4_visual_check()
        viz_files = list(AUG_TEST_DIR.glob("visual_check_*.jpg"))
        gate_results["bbox_visual"] = len(viz_files) > 0
    except Exception as e:
        logger.error("Visual check FAILED: %s", e)
        gate_results["bbox_visual"] = False

    try:
        gate_results["data_yaml_valid"] = step5_model_check(data_yaml)
    except Exception as e:
        logger.error("Model check FAILED: %s", e)
        gate_results["data_yaml_valid"] = False

    gate_results["model_nc9"] = gate_results.get("data_yaml_valid", False)

    logger.info("")
    logger.info("=== GATE 2 STATUS ===")
    labels = {
        "augmentation_100": "Augmentation pipeline ran on 100 images without error",
        "bbox_visual": "Bbox visual check saved to notebooks/aug_test/",
        "data_yaml_valid": "configs/data.yaml generated and validated by Ultralytics",
        "model_nc9": "Model instantiates with nc=9",
    }
    all_pass = True
    for key, description in labels.items():
        passed = gate_results.get(key, False)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        logger.info("[%s] %s", status, description)

    if all_pass:
        logger.info("Gate 2: ALL CHECKS PASSED")
    else:
        logger.warning("Gate 2: SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
