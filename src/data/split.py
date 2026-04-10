"""Train/val/test split with class-stratified sampling."""

from __future__ import annotations

import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.data.convert import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def stratified_split(
    data_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Split a YOLO-format dataset into train/val/test with stratification.

    Ensures each split maintains the same class distribution as the full
    dataset. The primary class for each image is taken from the first
    annotation line in its label file.

    Args:
        data_dir: Root directory containing images/ and labels/ subdirs.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to lists of image paths.
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    by_class: dict[int, list[Path]] = defaultdict(list)

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        text = lbl_path.read_text().strip()
        if not text:
            continue

        first_line = text.splitlines()[0].strip().split()
        if len(first_line) != 5:
            continue

        class_id = int(float(first_line[0]))
        by_class[class_id].append(img_path)

    rng = random.Random(seed)
    split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    for class_id in sorted(by_class.keys()):
        items = by_class[class_id]
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        split["train"].extend(items[:n_train])
        split["val"].extend(items[n_train : n_train + n_val])
        split["test"].extend(items[n_train + n_val :])

    for key in split:
        rng.shuffle(split[key])

    logger.info(
        "Stratified split: train=%d, val=%d, test=%d",
        len(split["train"]), len(split["val"]), len(split["test"]),
    )
    return split


def create_split_dirs(
    data_dir: str | Path,
    split_mapping: dict[str, list[Path]],
) -> None:
    """Move images and labels into train/val/test subdirectories.

    After this, the layout will be:
      data_dir/images/{train,val,test}/
      data_dir/labels/{train,val,test}/

    Args:
        data_dir: Root dataset directory.
        split_mapping: Output of stratified_split().
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    for split_name, img_paths in split_mapping.items():
        dst_images = images_dir / split_name
        dst_labels = labels_dir / split_name
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for img_path in img_paths:
            lbl_path = labels_dir / f"{img_path.stem}.txt"

            shutil.move(str(img_path), str(dst_images / img_path.name))
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(dst_labels / lbl_path.name))

        logger.info("[%s] Moved %d image/label pairs", split_name, len(img_paths))
