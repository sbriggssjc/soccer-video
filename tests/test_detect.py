"""Tests for soccer_highlights.detect."""
from __future__ import annotations

import numpy as np
import pytest

from soccer_highlights.detect import _normalize, _expand_window


# --- _normalize ---------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        arr = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        result = _normalize(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_all_same(self):
        arr = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        result = _normalize(arr)
        # All identical → all zero after subtracting min, max is 0 → no division
        assert np.allclose(result, 0.0)

    def test_empty(self):
        arr = np.zeros(0, dtype=np.float32)
        result = _normalize(arr)
        assert result.size == 0

    def test_single_element(self):
        arr = np.array([7.0], dtype=np.float32)
        result = _normalize(arr)
        assert result[0] == pytest.approx(0.0)

    def test_negative_values(self):
        arr = np.array([-10.0, 0.0, 10.0], dtype=np.float32)
        result = _normalize(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_output_dtype(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = _normalize(arr)
        assert result.dtype == np.float32

    def test_two_elements(self):
        arr = np.array([2.0, 8.0], dtype=np.float32)
        result = _normalize(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)


# --- _expand_window -----------------------------------------------------------

class TestExpandWindow:
    def test_basic(self):
        start, end = _expand_window(5, 10, pre=2.0, post=3.0, total_duration=60.0)
        assert start == pytest.approx(3.0)
        assert end == pytest.approx(14.0)  # (10 + 1) + 3.0

    def test_clamp_start(self):
        start, end = _expand_window(0, 2, pre=5.0, post=1.0, total_duration=60.0)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(4.0)  # (2 + 1) + 1.0

    def test_clamp_end(self):
        start, end = _expand_window(55, 59, pre=1.0, post=5.0, total_duration=60.0)
        assert start == pytest.approx(54.0)
        assert end == pytest.approx(60.0)  # clamped to total_duration

    def test_zero_pre_post(self):
        start, end = _expand_window(10, 15, pre=0.0, post=0.0, total_duration=100.0)
        assert start == pytest.approx(10.0)
        assert end == pytest.approx(16.0)  # 15 + 1

    def test_single_bin(self):
        start, end = _expand_window(5, 5, pre=1.0, post=1.0, total_duration=30.0)
        assert start == pytest.approx(4.0)
        assert end == pytest.approx(7.0)  # (5 + 1) + 1.0

    def test_both_clamp(self):
        start, end = _expand_window(0, 0, pre=5.0, post=10.0, total_duration=5.0)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(5.0)
