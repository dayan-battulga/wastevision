#!/usr/bin/env python3
"""Convert all raw datasets to YOLO format in data/processed/.

Runs conversion for RealWaste, Kaggle Garbage V2, and TACO, then
validates the result and reports class distribution.  If any class
has fewer than 400 annotations, lightweight oversampling is applied
by duplicating existing images until the threshold is met.

Usage:
    python scripts/run_conversion.py
"""

from __future__ import annotations

import logging
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.convert import CLASS_NAMES, convert_dataset
from src.data.validate import check_class_distribution, validate_annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASETS: list[dict[str, str]] = [
    {
        "name": "realwaste",
        "raw_dir": "data/realwaste-main/RealWaste",
    },
    {
        "name": "kaggle_v2",
        "raw_dir": "data/Kaggle_Garbage_V2/Garbage classification/Garbage classification",
    },
    {
        "name": "taco",
        "raw_dir": "data/TACO",
    },
]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MIN_SAMPLES_PER_CLASS = 800
SEED = 42


def _oversample_class(
    data_dir: Path,
    class_id: int,
    current_count: int,
    target_count: int,
) -> int:
    """Duplicate random images of a class until the target count is reached.

    Args:
        data_dir: Root with images/ and labels/ subdirs.
        class_id: The class index to oversample.
        current_count: Current annotation count for this class.
        target_count: Desired minimum annotation count.

    Returns:
        Number of new images created.
    """
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    candidates: list[tuple[Path, Path]] = []
    for lbl_path in labels_dir.iterdir():
        if lbl_path.suffix != ".txt":
            continue
        text = lbl_path.read_text().strip()
        if not text:
            continue
        first_class = int(float(text.splitlines()[0].split()[0]))
        if first_class == class_id:
            stem = lbl_path.stem
            for ext in (".jpg", ".jpeg", ".png"):
                img_path = images_dir / f"{stem}{ext}"
                if img_path.exists():
                    candidates.append((img_path, lbl_path))
                    break

    if not candidates:
        logger.warning("No images found for class %d to oversample", class_id)
        return 0

    rng = random.Random(SEED)
    added = 0
    idx = 0
    while current_count + added < target_count:
        src_img, src_lbl = rng.choice(candidates)
        new_stem = f"{src_lbl.stem}_dup{idx}"
        shutil.copy2(src_img, images_dir / f"{new_stem}{src_img.suffix}")
        shutil.copy2(src_lbl, labels_dir / f"{new_stem}.txt")
        added += 1
        idx += 1

    return added


def main() -> None:
    """Run the full conversion pipeline."""
    if PROCESSED_DIR.exists():
        logger.info("Removing existing data/processed/ ...")
        shutil.rmtree(PROCESSED_DIR)

    total_images = 0
    for ds in RAW_DATASETS:
        raw_dir = PROJECT_ROOT / ds["raw_dir"]
        logger.info("Converting %s from %s ...", ds["name"], raw_dir)
        count = convert_dataset(
            raw_dir=raw_dir,
            output_dir=PROCESSED_DIR,
            dataset_name=ds["name"],
        )
        logger.info("  -> %s: %d images converted", ds["name"], count)
        total_images += count

    logger.info("Total images in data/processed/: %d", total_images)

    logger.info("--- Class Distribution ---")
    dist = check_class_distribution(PROCESSED_DIR)
    below_threshold: list[tuple[int, str, int]] = []
    for class_id in range(len(CLASS_NAMES)):
        name = CLASS_NAMES[class_id]
        count = dist.get(class_id, 0)
        status = "OK" if count >= MIN_SAMPLES_PER_CLASS else "LOW"
        logger.info("  %d: %-16s %5d  [%s]", class_id, name, count, status)
        if count < MIN_SAMPLES_PER_CLASS:
            below_threshold.append((class_id, name, count))

    for class_id, name, count in below_threshold:
        deficit = MIN_SAMPLES_PER_CLASS - count
        logger.info("Oversampling %s (%d -> %d) ...", name, count, MIN_SAMPLES_PER_CLASS)
        added = _oversample_class(PROCESSED_DIR, class_id, count, MIN_SAMPLES_PER_CLASS)
        logger.info("  -> Added %d duplicate images for %s", added, name)

    if below_threshold:
        logger.info("--- Updated Distribution ---")
        dist = check_class_distribution(PROCESSED_DIR)
        for class_id in range(len(CLASS_NAMES)):
            name = CLASS_NAMES[class_id]
            count = dist.get(class_id, 0)
            logger.info("  %d: %-16s %5d", class_id, name, count)

    logger.info("--- Annotation Validation ---")
    issues = validate_annotations(PROCESSED_DIR)
    if not issues:
        logger.info("PASSED — all annotations valid")
    else:
        for issue_type, entries in issues.items():
            logger.warning("  %s: %d issues", issue_type, len(entries))
            for entry in entries[:5]:
                logger.warning("    %s", entry)
            if len(entries) > 5:
                logger.warning("    ... and %d more", len(entries) - 5)

    logger.info("--- Conversion complete ---")


if __name__ == "__main__":
    main()
