"""Tests for src.data modules (download, convert, split, validate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.convert import (
    CLASS_MAPPING,
    CLASS_NAMES,
    KAGGLE_V2_LABEL_MAP,
    REALWASTE_LABEL_MAP,
    TACO_LABEL_MAP,
    convert_coco_bbox_to_yolo,
    remap_label,
)


class TestClassTaxonomy:
    """Verify the class taxonomy is correctly defined."""

    def test_class_count(self, class_names: list[str]) -> None:
        assert len(class_names) == 9

    def test_class_mapping_matches_names(self) -> None:
        for idx, name in enumerate(CLASS_NAMES):
            assert CLASS_MAPPING[name] == idx

    def test_expected_classes_present(self) -> None:
        expected = {
            "cardboard", "food_organics", "glass", "metal", "misc_trash",
            "paper", "plastic", "textile_trash", "vegetation",
        }
        assert set(CLASS_NAMES) == expected


class TestLabelMaps:
    """Verify that all label maps produce valid class indices."""

    def test_realwaste_values_in_range(self) -> None:
        for class_id in REALWASTE_LABEL_MAP.values():
            assert 0 <= class_id <= 8

    def test_kaggle_v2_values_in_range(self) -> None:
        for class_id in KAGGLE_V2_LABEL_MAP.values():
            assert 0 <= class_id <= 8

    def test_taco_values_in_range(self) -> None:
        for class_id in TACO_LABEL_MAP.values():
            assert 0 <= class_id <= 8

    def test_taco_covers_all_60_categories(self) -> None:
        assert len(TACO_LABEL_MAP) == 60
        assert set(TACO_LABEL_MAP.keys()) == set(range(60))


class TestRemapLabel:
    """Tests for the remap_label function."""

    def test_remap_known_kaggle_v2_label(self) -> None:
        assert remap_label("trash", KAGGLE_V2_LABEL_MAP) == 4

    def test_remap_known_realwaste_label(self) -> None:
        assert remap_label("Food Organics", REALWASTE_LABEL_MAP) == 1

    def test_remap_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            remap_label("nonexistent", KAGGLE_V2_LABEL_MAP)


class TestConvertCocoBbox:
    """Tests for COCO-to-YOLO bbox conversion."""

    def test_full_image_bbox(self) -> None:
        result = convert_coco_bbox_to_yolo([0, 0, 100, 200], 100, 200)
        assert result == pytest.approx([0.5, 0.5, 1.0, 1.0])

    def test_quarter_bbox(self) -> None:
        result = convert_coco_bbox_to_yolo([0, 0, 50, 100], 100, 200)
        assert result == pytest.approx([0.25, 0.25, 0.5, 0.5])

    def test_values_clamped_to_unit(self) -> None:
        result = convert_coco_bbox_to_yolo([0, 0, 200, 400], 100, 200)
        for val in result:
            assert 0.0 <= val <= 1.0


class TestSplit:
    """Tests for dataset splitting."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_stratified_split_ratios(self, tmp_dataset: Path) -> None:
        # TODO: Test that split ratios sum to 1 and files are distributed correctly
        pass


class TestValidate:
    """Tests for annotation validation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_check_image_label_pairs(self, tmp_dataset: Path) -> None:
        # TODO: Test orphan detection with mismatched files
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_check_label_format_valid(self) -> None:
        # TODO: Test that valid YOLO labels pass validation
        pass
