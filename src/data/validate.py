"""Verify annotation integrity and image-label consistency."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from src.data.convert import CLASS_NAMES, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

NUM_CLASSES = len(CLASS_NAMES)


def validate_annotations(data_dir: str | Path) -> dict[str, list[str]]:
    """Run all validation checks on a YOLO-format dataset.

    Args:
        data_dir: Root directory containing images/ and labels/ subdirs.

    Returns:
        Dict of issue type -> list of affected file paths.
        Empty dict means the dataset is clean.
    """
    data_dir = Path(data_dir)
    issues: dict[str, list[str]] = {}

    orphans = check_image_label_pairs(data_dir)
    if orphans:
        issues["orphaned_files"] = orphans

    labels_dir = data_dir / "labels"
    if labels_dir.is_dir():
        format_errors: list[str] = []
        for lbl_path in sorted(labels_dir.iterdir()):
            if lbl_path.suffix != ".txt":
                continue
            errs = check_label_format(lbl_path)
            for e in errs:
                format_errors.append(f"{lbl_path.name}: {e}")
        if format_errors:
            issues["label_format_errors"] = format_errors

    corrupt = _check_corrupt_images(data_dir)
    if corrupt:
        issues["corrupt_images"] = corrupt

    total_issues = sum(len(v) for v in issues.values())
    if total_issues == 0:
        logger.info("Validation passed — 0 issues found in %s", data_dir)
    else:
        logger.warning("Validation found %d issues in %s", total_issues, data_dir)

    return issues


def check_image_label_pairs(data_dir: str | Path) -> list[str]:
    """Find images missing labels and labels missing images.

    Args:
        data_dir: Root dataset directory with images/ and labels/ subdirs.

    Returns:
        List of orphaned file paths (relative descriptions).
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        logger.error("Expected images/ and labels/ subdirs in %s", data_dir)
        return []

    image_stems: set[str] = set()
    for f in images_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            image_stems.add(f.stem)

    label_stems: set[str] = set()
    for f in labels_dir.iterdir():
        if f.suffix == ".txt":
            label_stems.add(f.stem)

    orphans: list[str] = []
    for stem in sorted(image_stems - label_stems):
        orphans.append(f"image missing label: {stem}")
    for stem in sorted(label_stems - image_stems):
        orphans.append(f"label missing image: {stem}")

    return orphans


def check_label_format(label_path: str | Path) -> list[str]:
    """Validate that a single label file has correct YOLO format.

    Checks: non-empty, 5 values per line, class_id in [0, nc), coords in [0, 1].

    Args:
        label_path: Path to a .txt label file.

    Returns:
        List of error messages (empty if valid).
    """
    label_path = Path(label_path)
    errors: list[str] = []

    text = label_path.read_text().strip()
    if not text:
        errors.append("empty label file")
        return errors

    for line_num, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            errors.append(f"line {line_num}: expected 5 values, got {len(parts)}")
            continue

        try:
            values = [float(p) for p in parts]
        except ValueError:
            errors.append(f"line {line_num}: non-numeric values")
            continue

        class_id = int(values[0])
        if class_id < 0 or class_id >= NUM_CLASSES:
            errors.append(f"line {line_num}: class_id {class_id} out of range [0, {NUM_CLASSES})")

        for i, name in enumerate(["cx", "cy", "w", "h"], start=1):
            if values[i] < 0.0 or values[i] > 1.0:
                errors.append(f"line {line_num}: {name}={values[i]:.4f} outside [0, 1]")

    return errors


def check_class_distribution(data_dir: str | Path) -> dict[int, int]:
    """Count the number of annotations per class across the dataset.

    Args:
        data_dir: Root dataset directory with a labels/ subdir.

    Returns:
        Dict mapping class_id -> annotation count.
    """
    data_dir = Path(data_dir)
    labels_dir = data_dir / "labels"

    counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}

    if not labels_dir.is_dir():
        logger.error("No labels/ directory in %s", data_dir)
        return counts

    for lbl_path in labels_dir.iterdir():
        if lbl_path.suffix != ".txt":
            continue
        text = lbl_path.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(float(parts[0]))
                if 0 <= class_id < NUM_CLASSES:
                    counts[class_id] += 1

    return counts


def _check_corrupt_images(data_dir: Path) -> list[str]:
    """Find images that OpenCV cannot read.

    Args:
        data_dir: Root dataset directory with an images/ subdir.

    Returns:
        List of corrupt image filenames.
    """
    images_dir = data_dir / "images"
    if not images_dir.is_dir():
        return []

    corrupt: list[str] = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if cv2.imread(str(img_path)) is None:
            corrupt.append(img_path.name)

    return corrupt
