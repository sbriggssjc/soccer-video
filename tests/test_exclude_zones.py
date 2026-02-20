"""Tests for YOLO exclusion zone filtering in ball_telemetry."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tools.ball_telemetry import (
    BallSample,
    ExcludeZone,
    fuse_yolo_and_centroid,
    load_exclude_zones,
)


# ---------------------------------------------------------------------------
# ExcludeZone.contains
# ---------------------------------------------------------------------------


class TestExcludeZoneContains:
    def test_inside(self):
        z = ExcludeZone(x_min=100, x_max=200, y_min=50, y_max=150)
        assert z.contains(150, 100, 0)

    def test_outside_x(self):
        z = ExcludeZone(x_min=100, x_max=200, y_min=50, y_max=150)
        assert not z.contains(250, 100, 0)

    def test_outside_y(self):
        z = ExcludeZone(x_min=100, x_max=200, y_min=50, y_max=150)
        assert not z.contains(150, 200, 0)

    def test_frame_range_inside(self):
        z = ExcludeZone(x_min=0, x_max=400, frame_start=100, frame_end=200)
        assert z.contains(200, 500, 150)

    def test_frame_range_outside(self):
        z = ExcludeZone(x_min=0, x_max=400, frame_start=100, frame_end=200)
        assert not z.contains(200, 500, 50)
        assert not z.contains(200, 500, 200)  # frame_end is exclusive

    def test_boundary_inclusive(self):
        z = ExcludeZone(x_min=100, x_max=200, y_min=50, y_max=150)
        assert z.contains(100, 50, 0)   # min edges inclusive
        assert z.contains(200, 150, 0)  # max edges inclusive

    def test_defaults_match_everything(self):
        z = ExcludeZone()
        assert z.contains(500, 500, 999)


# ---------------------------------------------------------------------------
# load_exclude_zones
# ---------------------------------------------------------------------------


class TestLoadExcludeZones:
    def test_load_valid_json(self, tmp_path):
        data = [
            {"x_min": 0, "x_max": 400, "y_min": 0, "y_max": 1080,
             "frame_start": 1100, "frame_end": 1500,
             "note": "adjacent field ball"},
        ]
        p = tmp_path / "exclude.json"
        p.write_text(json.dumps(data))
        zones = load_exclude_zones(p)
        assert len(zones) == 1
        assert zones[0].x_min == 0
        assert zones[0].x_max == 400
        assert zones[0].frame_start == 1100
        assert zones[0].frame_end == 1500

    def test_load_single_object(self, tmp_path):
        """A single zone object (not wrapped in a list) should also work."""
        data = {"x_min": 100, "x_max": 300}
        p = tmp_path / "exclude.json"
        p.write_text(json.dumps(data))
        zones = load_exclude_zones(p)
        assert len(zones) == 1
        assert zones[0].x_min == 100
        assert zones[0].x_max == 300

    def test_missing_file_returns_empty(self, tmp_path):
        zones = load_exclude_zones(tmp_path / "nonexistent.json")
        assert zones == []

    def test_defaults_for_omitted_keys(self, tmp_path):
        data = [{"x_min": 50}]
        p = tmp_path / "exclude.json"
        p.write_text(json.dumps(data))
        zones = load_exclude_zones(p)
        assert zones[0].x_min == 50
        assert zones[0].x_max == float("inf")
        assert zones[0].y_min == 0
        assert zones[0].y_max == float("inf")
        assert zones[0].frame_start == 0


# ---------------------------------------------------------------------------
# fuse_yolo_and_centroid with exclude_zones
# ---------------------------------------------------------------------------


def _make_sample(frame: int, x: float, y: float, conf: float = 0.8) -> BallSample:
    return BallSample(t=frame / 30.0, frame=frame, x=x, y=y, conf=conf)


class TestFuseWithExcludeZones:
    def test_no_zones_passes_all(self):
        """Without exclusion zones, all YOLO detections are used."""
        yolo = [_make_sample(i, 500.0 + i, 400.0) for i in range(10)]
        positions, mask, conf, src = fuse_yolo_and_centroid(
            yolo, [], 10, 1920.0, 1080.0, exclude_zones=None,
        )
        # All 10 frames should have YOLO-sourced positions
        yolo_source = (src == 1)  # FUSE_YOLO = 1
        assert yolo_source.sum() == 10

    def test_zone_filters_detections(self):
        """Detections inside an exclusion zone are dropped."""
        yolo = [
            _make_sample(0, 500.0, 400.0),   # outside zone → kept
            _make_sample(1, 200.0, 400.0),   # inside zone → dropped
            _make_sample(2, 500.0, 400.0),   # outside zone → kept
        ]
        zone = ExcludeZone(x_min=0, x_max=300, y_min=0, y_max=1080)
        positions, mask, conf, src = fuse_yolo_and_centroid(
            yolo, [], 3, 1920.0, 1080.0, exclude_zones=[zone],
        )
        # Frame 1 should NOT be YOLO-sourced (it was excluded)
        assert src[0] == 1  # FUSE_YOLO
        assert src[1] != 1  # excluded → not YOLO
        assert src[2] == 1  # FUSE_YOLO

    def test_frame_bounded_zone(self):
        """A zone with frame bounds only filters within that range."""
        yolo = [
            _make_sample(0, 200.0, 400.0),    # in zone region but frame < start → kept
            _make_sample(10, 200.0, 400.0),   # in zone region and frame in range → dropped
            _make_sample(20, 200.0, 400.0),   # in zone region but frame >= end → kept
        ]
        zone = ExcludeZone(x_min=0, x_max=300, y_min=0, y_max=1080,
                           frame_start=5, frame_end=15)
        positions, mask, conf, src = fuse_yolo_and_centroid(
            yolo, [], 25, 1920.0, 1080.0, exclude_zones=[zone],
        )
        assert src[0] == 1   # FUSE_YOLO (before zone frame range)
        assert src[10] != 1  # excluded
        assert src[20] == 1  # FUSE_YOLO (after zone frame range)

    def test_multiple_zones(self):
        """Multiple zones can each filter different regions/times."""
        yolo = [
            _make_sample(0, 100.0, 400.0),    # in zone A → dropped
            _make_sample(1, 1800.0, 900.0),   # in zone B → dropped
            _make_sample(2, 960.0, 540.0),    # in neither → kept
        ]
        zone_a = ExcludeZone(x_min=0, x_max=300)
        zone_b = ExcludeZone(x_min=1600, x_max=1920, y_min=700, y_max=1080)
        positions, mask, conf, src = fuse_yolo_and_centroid(
            yolo, [], 3, 1920.0, 1080.0, exclude_zones=[zone_a, zone_b],
        )
        assert src[0] != 1  # excluded by zone A
        assert src[1] != 1  # excluded by zone B
        assert src[2] == 1  # FUSE_YOLO

    def test_empty_zones_list(self):
        """An empty zones list has no effect."""
        yolo = [_make_sample(0, 200.0, 400.0)]
        positions, mask, conf, src = fuse_yolo_and_centroid(
            yolo, [], 1, 1920.0, 1080.0, exclude_zones=[],
        )
        assert src[0] == 1  # FUSE_YOLO, not filtered
