"""Tests for src.inference modules."""

from __future__ import annotations

import numpy as np
import pytest


class TestPreprocess:
    """Tests for image preprocessing."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_letterbox_preserves_aspect_ratio(self, sample_image: np.ndarray) -> None:
        # TODO: Test that letterbox output is square with correct padding
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_preprocess_output_shape(self, sample_image: np.ndarray) -> None:
        # TODO: Test CHW output shape and float32 dtype
        pass


class TestPostprocess:
    """Tests for NMS and result formatting."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_non_max_suppression_reduces_boxes(self) -> None:
        # TODO: Test that NMS correctly suppresses overlapping boxes
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_scale_boxes_to_original(self) -> None:
        # TODO: Test coordinate rescaling from preprocessed to original image
        pass


class TestServer:
    """Tests for the FastAPI inference endpoint."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_health_endpoint(self) -> None:
        # TODO: Test GET /health returns 200 with {"status": "ok"}
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_predict_endpoint_accepts_image(self) -> None:
        # TODO: Test POST /predict with a sample image file
        pass
