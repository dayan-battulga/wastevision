"""Tests for src.model modules."""

from __future__ import annotations

import pytest


class TestYoloConfig:
    """Tests for YOLOv8 config generation."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_generate_data_yaml_creates_file(self, tmp_path) -> None:
        # TODO: Test that generate_data_yaml writes a valid YAML with correct nc and names
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_get_model_config_keys(self) -> None:
        # TODO: Test that returned config dict has required keys
        pass


class TestExport:
    """Tests for model export."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_export_onnx(self) -> None:
        # TODO: Test ONNX export produces a valid .onnx file
        pass
