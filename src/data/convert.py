"""Remap heterogeneous dataset labels to the unified 9-class YOLO taxonomy.

Supports RealWaste (1:1), Kaggle Garbage V2, and
TACO (60-category COCO → 9-class YOLO).

Classification datasets (RealWaste, Kaggle V2) use a pretrained YOLOv8n
to generate bounding-box proposals. TACO already has COCO annotations.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 9-class taxonomy
# ---------------------------------------------------------------------------

CLASS_NAMES: list[str] = [
    "cardboard",       # 0
    "food_organics",   # 1
    "glass",           # 2
    "metal",           # 3
    "misc_trash",      # 4
    "paper",           # 5
    "plastic",         # 6
    "textile_trash",   # 7
    "vegetation",      # 8
]

CLASS_MAPPING: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Per-dataset label maps
# ---------------------------------------------------------------------------

REALWASTE_LABEL_MAP: dict[str, int] = {
    "Cardboard": 0,
    "Food Organics": 1,
    "Glass": 2,
    "Metal": 3,
    "Miscellaneous Trash": 4,
    "Paper": 5,
    "Plastic": 6,
    "Textile Trash": 7,
    "Vegetation": 8,
}

KAGGLE_V2_LABEL_MAP: dict[str, int] = {
    "cardboard": 0,  # cardboard
    "glass": 2,      # glass
    "metal": 3,      # metal
    "paper": 5,      # paper
    "plastic": 6,    # plastic
    "trash": 4,      # misc_trash
}

# TACO uses integer category IDs (0-59). Map each to one of the 9 classes.
TACO_LABEL_MAP: dict[int, int] = {
    # --- metal (3) ---
    0: 3,   # Aluminium foil
    2: 3,   # Aluminium blister pack
    8: 3,   # Metal bottle cap
    10: 3,  # Food Can
    11: 3,  # Aerosol
    12: 3,  # Drink can
    28: 3,  # Metal lid
    50: 3,  # Pop tab
    52: 3,  # Scrap metal
    # --- plastic (6) ---
    4: 6,   # Other plastic bottle
    5: 6,   # Clear plastic bottle
    7: 6,   # Plastic bottle cap
    21: 6,  # Disposable plastic cup
    22: 6,  # Foam cup
    24: 6,  # Other plastic cup
    27: 6,  # Plastic lid
    29: 6,  # Other plastic
    36: 6,  # Plastic film
    37: 6,  # Six pack rings
    38: 6,  # Garbage bag
    39: 6,  # Other plastic wrapper
    40: 6,  # Single-use carrier bag
    41: 6,  # Polypropylene bag
    42: 6,  # Crisp packet
    43: 6,  # Spread tub
    44: 6,  # Tupperware
    45: 6,  # Disposable food container
    46: 6,  # Foam food container
    47: 6,  # Other plastic container
    48: 6,  # Plastic glooves
    49: 6,  # Plastic utensils
    54: 6,  # Squeezable tube
    55: 6,  # Plastic straw
    # --- glass (2) ---
    6: 2,   # Glass bottle
    9: 2,   # Broken glass
    23: 2,  # Glass cup
    26: 2,  # Glass jar
    # --- cardboard (0) ---
    3: 0,   # Carded blister pack
    13: 0,  # Toilet tube
    14: 0,  # Other carton
    15: 0,  # Egg carton
    16: 0,  # Drink carton
    17: 0,  # Corrugated carton
    18: 0,  # Meal carton
    19: 0,  # Pizza box
    # --- paper (5) ---
    20: 5,  # Paper cup
    30: 5,  # Magazine paper
    31: 5,  # Tissues
    32: 5,  # Wrapping paper
    33: 5,  # Normal paper
    34: 5,  # Paper bag
    35: 5,  # Plastified paper bag
    56: 5,  # Paper straw
    # --- food_organics (1) ---
    25: 1,  # Food waste
    # --- misc_trash (4) ---
    1: 4,   # Battery
    51: 4,  # Rope & strings
    57: 4,  # Styrofoam piece
    58: 4,  # Unlabeled litter
    59: 4,  # Cigarette
    # --- textile_trash (7) ---
    53: 7,  # Shoe
}

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def remap_label(source_label: str | int, label_map: dict[str | int, int]) -> int:
    """Map a source-dataset label to the unified class index.

    Args:
        source_label: Original label from the source dataset (string or int).
        label_map: Mapping from source labels to unified class indices.

    Returns:
        Unified class index (0-8).

    Raises:
        KeyError: If source_label is not found in the mapping.
    """
    return label_map[source_label]


def convert_coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert a COCO-format bbox to YOLO-format.

    COCO: [x_min, y_min, width, height] in pixels.
    YOLO: [x_center, y_center, width, height] normalized to [0, 1].

    Args:
        bbox: COCO bbox [x_min, y_min, w, h] in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        [x_center, y_center, width, height] normalized to [0, 1].
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / image_width
    y_center = (y_min + h / 2.0) / image_height
    norm_w = w / image_width
    norm_h = h / image_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return [x_center, y_center, norm_w, norm_h]


def convert_to_yolo(
    annotation: dict[str, Any],
    image_width: int,
    image_height: int,
    label_map: dict[int, int],
) -> list[float] | None:
    """Convert a single COCO annotation to a YOLO label line.

    Args:
        annotation: COCO annotation dict with 'bbox' and 'category_id'.
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        label_map: Mapping from COCO category_id to unified class index.

    Returns:
        [class_id, x_center, y_center, width, height] or None if the
        category is not in the label map.
    """
    cat_id = annotation["category_id"]
    if cat_id not in label_map:
        logger.warning("Unknown TACO category_id=%d, skipping annotation", cat_id)
        return None

    class_id = label_map[cat_id]
    bbox = annotation["bbox"]
    yolo_bbox = convert_coco_bbox_to_yolo(bbox, image_width, image_height)
    return [float(class_id), *yolo_bbox]


# ---------------------------------------------------------------------------
# Per-dataset converters
# ---------------------------------------------------------------------------

def _xyxy_to_yolo(
    xyxy: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert a single xyxy pixel-coord box to YOLO normalized format.

    Args:
        xyxy: Array of [x1, y1, x2, y2] in pixels.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        [x_center, y_center, width, height] clamped to [0, 1].
    """
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    w = (x2 - x1) / image_width
    h = (y2 - y1) / image_height
    return [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    ]


_FULL_IMAGE_BBOX = "0.500000 0.500000 1.000000 1.000000"


def _convert_with_bbox_proposals(
    raw_dir: Path,
    output_dir: Path,
    label_map: dict[str, int],
    dataset_name: str,
    confidence_threshold: float = 0.3,
) -> int:
    """Convert a folder-based classification dataset using YOLOv8n bbox proposals.

    A pretrained YOLOv8n (COCO weights) is run on every image to locate
    objects.  The COCO class predictions are discarded -- the class label
    comes from the folder name.  If the detector produces zero boxes above
    ``confidence_threshold``, a full-image fallback bbox is written instead.

    Args:
        raw_dir: Root with class-name subdirectories containing images.
        output_dir: Destination (images/ and labels/ subdirs will be created).
        label_map: Folder-name to class-index mapping.
        dataset_name: Name used for logging.
        confidence_threshold: Minimum detector confidence to keep a proposal.

    Returns:
        Number of images successfully converted.
    """
    from ultralytics import YOLO

    detector = YOLO("yolov8n.pt")
    logger.info("[%s] Loaded YOLOv8n for bbox proposals (conf=%.2f)", dataset_name, confidence_threshold)

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    proposals_used = 0
    fallbacks_used = 0

    for folder_name, class_id in label_map.items():
        class_dir = raw_dir / folder_name
        if not class_dir.is_dir():
            logger.warning("[%s] Missing class folder: %s", dataset_name, class_dir)
            continue

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("[%s] Corrupt image, skipping: %s", dataset_name, img_path)
                continue

            img_h, img_w = image.shape[:2]

            results = detector.predict(
                source=str(img_path),
                conf=confidence_threshold,
                verbose=False,
            )
            boxes = results[0].boxes

            yolo_lines: list[str] = []
            if len(boxes) > 0:
                for xyxy_tensor in boxes.xyxy:
                    xyxy = xyxy_tensor.cpu().numpy()
                    cx, cy, w, h = _xyxy_to_yolo(xyxy, img_w, img_h)
                    yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                proposals_used += 1
            else:
                yolo_lines.append(f"{class_id} {_FULL_IMAGE_BBOX}")
                fallbacks_used += 1

            stem = f"{dataset_name}_{folder_name}_{img_path.stem}"
            dst_img = images_out / f"{stem}{img_path.suffix}"
            dst_lbl = labels_out / f"{stem}.txt"

            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(yolo_lines) + "\n")
            converted += 1

    logger.info(
        "[%s] Converted %d images (%d with proposals, %d with fallback bbox)",
        dataset_name, converted, proposals_used, fallbacks_used,
    )
    return converted


def _convert_taco(
    raw_dir: Path,
    output_dir: Path,
) -> int:
    """Convert the TACO dataset (COCO annotations) to YOLO format.

    Args:
        raw_dir: TACO repo root containing data/annotations.json and batch dirs.
        output_dir: Destination (images/ and labels/ subdirs will be created).

    Returns:
        Number of images successfully converted.
    """
    ann_path = raw_dir / "data" / "annotations.json"
    if not ann_path.exists():
        logger.error("[taco] annotations.json not found at %s", ann_path)
        return 0

    with open(ann_path) as f:
        coco = json.load(f)

    id_to_image: dict[int, dict[str, Any]] = {img["id"]: img for img in coco["images"]}

    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    for image_id, img_info in id_to_image.items():
        file_name = img_info["file_name"]
        img_path = raw_dir / "data" / file_name
        if not img_path.exists():
            logger.warning("[taco] Image not found: %s", img_path)
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        annotations = anns_by_image.get(image_id, [])
        if not annotations:
            continue

        yolo_lines: list[str] = []
        for ann in annotations:
            yolo_entry = convert_to_yolo(ann, img_w, img_h, TACO_LABEL_MAP)
            if yolo_entry is None:
                continue
            cls, cx, cy, w, h = yolo_entry
            yolo_lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not yolo_lines:
            continue

        stem = f"taco_{img_path.stem}"
        dst_img = images_out / f"{stem}{img_path.suffix}"
        dst_lbl = labels_out / f"{stem}.txt"

        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines) + "\n")
        converted += 1

    logger.info("[taco] Converted %d images", converted)
    return converted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    confidence_threshold: float = 0.3,
) -> int:
    """Convert a raw dataset to YOLO-format labels and images.

    For classification datasets (realwaste, kaggle_v2) a pretrained YOLOv8n
    generates bounding-box proposals; the class label comes from the folder
    name.  For TACO the existing COCO annotations are converted directly.

    Args:
        raw_dir: Root directory of the raw downloaded dataset.
        output_dir: Directory to write converted images/ and labels/.
        dataset_name: One of 'realwaste', 'kaggle_v2', 'taco'.
        confidence_threshold: Min detector confidence for bbox proposals
            (only used by realwaste/kaggle_v2).

    Returns:
        Number of images successfully converted.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    classification_datasets: dict[str, dict[str, int]] = {
        "realwaste": REALWASTE_LABEL_MAP,
        "kaggle_v2": KAGGLE_V2_LABEL_MAP,
    }

    if dataset_name in classification_datasets:
        return _convert_with_bbox_proposals(
            raw_dir,
            output_dir,
            classification_datasets[dataset_name],
            dataset_name,
            confidence_threshold=confidence_threshold,
        )
    if dataset_name == "taco":
        return _convert_taco(raw_dir, output_dir)

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Expected one of: realwaste, kaggle_v2, taco"
    )
