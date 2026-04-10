"""Compose Albumentations transforms from YAML config."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np

from src.data.convert import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def build_transform(config: list[dict[str, Any]]) -> A.Compose:
    """Build an Albumentations Compose pipeline from a config list.

    Args:
        config: List of transform dicts, each with 'name', 'p', and
            optional kwargs.  Matches the structure under
            configs/augmentation.yaml -> augmentation.train.

    Returns:
        Albumentations Compose object with bbox_params for YOLO format.
    """
    transforms: list[A.BasicTransform] = []

    for entry in config:
        name = entry["name"]
        p = entry.get("p", 1.0)

        kwargs = {k: v for k, v in entry.items() if k not in ("name", "p")}
        kwargs["p"] = p

        transform_cls = getattr(A, name, None)
        if transform_cls is None:
            logger.warning("Unknown Albumentations transform: %s, skipping", name)
            continue

        transforms.append(transform_cls(**kwargs))

    pipeline = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )

    logger.info("Built augmentation pipeline with %d transforms", len(transforms))
    return pipeline


def augment_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    config: list[dict[str, Any]],
    multiplier: int = 3,
) -> int:
    """Apply augmentation pipeline to a dataset, generating new samples.

    Args:
        data_dir: Source directory with images/ and labels/ subdirs.
        output_dir: Destination for augmented images and labels.
        config: Augmentation config list (the 'train' list from YAML).
        multiplier: Number of augmented copies per original image.

    Returns:
        Total number of augmented images generated.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    images_in = data_dir / "images"
    labels_in = data_dir / "labels"
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    transform = build_transform(config)
    generated = 0

    for img_path in sorted(images_in.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        lbl_path = labels_in / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Corrupt image, skipping: %s", img_path)
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        text = lbl_path.read_text().strip()
        if not text:
            continue

        bboxes: list[list[float]] = []
        class_labels: list[int] = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bboxes.append([cx, cy, w, h])
            class_labels.append(cls_id)

        if not bboxes:
            continue

        for i in range(multiplier):
            try:
                result = transform(
                    image=image_rgb,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
            except Exception:
                logger.debug("Transform failed for %s (copy %d), skipping", img_path.name, i)
                continue

            aug_img = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
            aug_bboxes = result["bboxes"]
            aug_labels = result["class_labels"]

            if not aug_bboxes:
                continue

            stem = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(images_out / f"{stem}.jpg"), aug_img)

            lines: list[str] = []
            for bbox, cls_id in zip(aug_bboxes, aug_labels):
                cx, cy, w, h = bbox
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (labels_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

            generated += 1

    logger.info("Generated %d augmented images in %s", generated, output_dir)
    return generated
