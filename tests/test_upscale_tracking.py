"""Tests for upscale.py catalog tracking integration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.upscale import _out_path, _track_upscale


class TestOutPath:
    def test_lanczos(self):
        src = Path("clip.mp4")
        out = _out_path(src, 2, "")
        assert out.stem == "clip__x2"
        assert out.suffix == ".mp4"

    def test_with_method_tag(self):
        src = Path("clip.mp4")
        out = _out_path(src, 4, "realesrgan")
        assert out.stem == "clip__x4__realesrgan"

    def test_scale_in_name(self):
        src = Path("001__GOAL__t10-t20.mp4")
        out = _out_path(src, 2, "")
        assert "__x2" in out.stem


class TestTrackUpscale:
    def test_track_calls_mark_upscaled(self, tmp_path, monkeypatch):
        """_track_upscale should call catalog.mark_upscaled."""
        import tools.catalog as cat

        # Set up isolated catalog dirs
        catalog_dir = tmp_path / "catalog"
        sidecar_dir = catalog_dir / "sidecar"
        monkeypatch.setattr(cat, "ROOT", tmp_path)
        monkeypatch.setattr(cat, "OUT_DIR", tmp_path / "out")
        monkeypatch.setattr(cat, "CATALOG_DIR", catalog_dir)
        monkeypatch.setattr(cat, "SIDE_CAR_DIR", sidecar_dir)
        monkeypatch.setattr(cat, "PIPELINE_STATUS_PATH", catalog_dir / "pipeline_status.csv")

        src = tmp_path / "clip.mp4"
        src.touch()
        out = tmp_path / "clip__x2.mp4"
        out.touch()

        # Should not raise
        _track_upscale(src, out, scale=2, model="lanczos")

        # Verify sidecar was created
        sidecar_file = sidecar_dir / "clip.json"
        assert sidecar_file.exists()

    def test_track_handles_errors_gracefully(self):
        """_track_upscale should not raise even if catalog import fails."""
        with patch("tools.upscale._track_upscale") as mock_track:
            mock_track.side_effect = None  # no-op
            # Should not raise
            _track_upscale(Path("nonexistent.mp4"), None, scale=2, model="lanczos")

    def test_track_with_error_message(self, tmp_path, monkeypatch):
        """_track_upscale with error should record the error."""
        import tools.catalog as cat

        catalog_dir = tmp_path / "catalog"
        sidecar_dir = catalog_dir / "sidecar"
        monkeypatch.setattr(cat, "ROOT", tmp_path)
        monkeypatch.setattr(cat, "OUT_DIR", tmp_path / "out")
        monkeypatch.setattr(cat, "CATALOG_DIR", catalog_dir)
        monkeypatch.setattr(cat, "SIDE_CAR_DIR", sidecar_dir)
        monkeypatch.setattr(cat, "PIPELINE_STATUS_PATH", catalog_dir / "pipeline_status.csv")

        src = tmp_path / "clip.mp4"
        src.touch()

        _track_upscale(src, None, scale=2, model="lanczos", error="ffmpeg failed")

        import json
        sidecar = sidecar_dir / "clip.json"
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["steps"]["upscale"]["done"] is False
        assert data["steps"]["upscale"]["error"] == "ffmpeg failed"
