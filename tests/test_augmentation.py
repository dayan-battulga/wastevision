"""Tests for src.augmentation modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class TestPipeline:
    """Tests for the augmentation pipeline builder."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_build_transform_returns_compose(self) -> None:
        # TODO: Test that build_transform returns an Albumentations Compose object
        pass

    @pytest.mark.skip(reason="Not implemented yet")
    def test_augmented_image_shape_preserved(self, sample_image: np.ndarray) -> None:
        # TODO: Test that augmented images maintain expected dimensions
        pass


class TestVisualize:
    """Tests for augmentation preview."""

    @pytest.mark.skip(reason="Not implemented yet")
    def test_preview_augmented_count(self, sample_image: np.ndarray) -> None:
        # TODO: Test that preview_augmented returns the requested number of samples
        pass
