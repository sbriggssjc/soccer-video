"""Tests for soccer_highlights.colors."""
from __future__ import annotations

import numpy as np
import pytest

from soccer_highlights.config import ColorsConfig, HSVRange
from soccer_highlights.colors import (
    HSVTolerance,
    combine_masks,
    hsv_mask,
    mask_ratio,
    pitch_mask,
    team_mask,
)


# --- HSVTolerance defaults ---------------------------------------------------

class TestHSVTolerance:
    def test_defaults(self):
        tol = HSVTolerance()
        assert tol.h == 12.0
        assert tol.s == 60.0
        assert tol.v == 60.0


# --- mask_ratio ---------------------------------------------------------------

class TestMaskRatio:
    def test_all_white(self):
        mask = np.full((10, 10), 255, dtype=np.uint8)
        assert mask_ratio(mask) == pytest.approx(1.0)

    def test_all_black(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert mask_ratio(mask) == pytest.approx(0.0)

    def test_half(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 255
        assert mask_ratio(mask) == pytest.approx(0.5)


# --- combine_masks ------------------------------------------------------------

class TestCombineMasks:
    def test_single_mask(self):
        mask = np.full((5, 5), 128, dtype=np.uint8)
        result = combine_masks([mask])
        assert result.shape == (5, 5)
        assert np.all(result == 128)

    def test_two_masks_or(self):
        m1 = np.zeros((4, 4), dtype=np.uint8)
        m1[0, 0] = 255
        m2 = np.zeros((4, 4), dtype=np.uint8)
        m2[3, 3] = 255
        result = combine_masks([m1, m2])
        assert result[0, 0] == 255
        assert result[3, 3] == 255
        assert result[1, 1] == 0

    def test_empty_iterable(self):
        result = combine_masks([])
        assert result.shape == (1, 1)
        assert result[0, 0] == 0

    def test_three_masks(self):
        m1 = np.array([[255, 0]], dtype=np.uint8)
        m2 = np.array([[0, 255]], dtype=np.uint8)
        m3 = np.array([[0, 0]], dtype=np.uint8)
        result = combine_masks([m1, m2, m3])
        assert result[0, 0] == 255
        assert result[0, 1] == 255


# --- hsv_mask -----------------------------------------------------------------

class TestHSVMask:
    def test_green_frame_detects_pitch(self):
        """A pure green frame should be detected by the default pitch HSV range."""
        # Create a 10x10 green frame (BGR)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :] = (0, 128, 0)  # BGR green
        center = HSVRange(h=60.0, s=128.0, v=64.0)  # Green in HSV
        tol = HSVTolerance(h=30.0, s=128.0, v=128.0)
        mask = hsv_mask(frame, center, tol)
        assert mask.shape == (10, 10)
        # Most pixels should be detected
        ratio = mask_ratio(mask)
        assert ratio > 0.5

    def test_black_frame_no_mask(self):
        """A black frame shouldn't match typical pitch colors."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        center = HSVRange(h=90.0, s=80.0, v=80.0)
        mask = hsv_mask(frame, center)
        ratio = mask_ratio(mask)
        assert ratio == pytest.approx(0.0)


# --- pitch_mask / team_mask --------------------------------------------------

class TestPitchTeamMask:
    def test_pitch_mask_returns_ndarray(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        config = ColorsConfig()
        mask = pitch_mask(frame, config)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (10, 10)

    def test_team_mask_returns_ndarray(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        config = ColorsConfig()
        mask = team_mask(frame, config)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (10, 10)
