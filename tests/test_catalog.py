"""Integration test for the atomic clips catalog pipeline.

Creates synthetic video clips, runs catalog.py operations, and verifies
that indexing, dedup, and pipeline status tracking all work correctly.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.catalog import (
    ATOMIC_HEADERS,
    ClipRecord,
    DuplicateRecord,
    MasterRecord,
    _normalize_timestamps,
    choose_canonical,
    compute_duplicate_groups,
    compute_overlap_ratio,
    format_float,
    gather_clip_record,
    is_canonical_rel,
    load_duplicates,
    load_sidecar,
    mark_branded,
    mark_upscaled,
    normalize_tree,
    parse_timestamps,
    probe_video,
    rebuild_atomic_index,
    save_sidecar,
    scan_atomic_clips,
    sha1_64,
    to_repo_relative,
    update_pipeline_status,
    write_atomic_index,
    write_catalog,
    write_duplicates_from_index,
)


# --- Helpers ------------------------------------------------------------------

_clip_counter = 0

def _make_clip(path: Path, width: int = 640, height: int = 360, fps: float = 30.0, duration: float = 2.0) -> Path:
    """Generate a minimal synthetic MP4 clip with unique content per call."""
    global _clip_counter
    _clip_counter += 1
    seed = _clip_counter
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot open MP4 writer")
    n_frames = int(fps * duration)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (34, 139, 34)  # green pitch
        cx = int(50 + (width - 100) * (i / max(n_frames - 1, 1)))
        cy = height // 2 + seed * 10  # vary per clip to produce unique hashes
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# --- parse_timestamps ---------------------------------------------------------

class TestParseTimestamps:
    def test_with_second_t(self):
        start, end = parse_timestamps("001__GOAL__t155.50-t166.40")
        assert start == pytest.approx(155.50)
        assert end == pytest.approx(166.40)

    def test_without_second_t(self):
        start, end = parse_timestamps("001__SHOT__t100.0-200.0")
        assert start == pytest.approx(100.0)
        assert end == pytest.approx(200.0)

    def test_integer_timestamps(self):
        start, end = parse_timestamps("005__SHOT__t7320-t7860")
        assert start == pytest.approx(7320.0)
        assert end == pytest.approx(7860.0)

    def test_no_match(self):
        start, end = parse_timestamps("random_filename")
        assert start is None
        assert end is None

    def test_underscore_suffix_breaks_match(self):
        start, end = parse_timestamps("001__SHOT__t100-t200_portrait_POST")
        assert start is None  # _portrait_POST prevents match (underscore suffix)

    def test_dot_debug_suffix_matches(self):
        start, end = parse_timestamps(
            "006__2025-10-04__TSC_vs_FC__BUILD_UP_GOAL__t40440.00-t43020.00.__DEBUG"
        )
        assert start == pytest.approx(674.0)   # 40440/60
        assert end == pytest.approx(717.0)     # 43020/60

    def test_dot_debug_final_portrait_suffix(self):
        start, end = parse_timestamps(
            "002__2025-09-13__TSC_vs_NEOFC__GOAL__t180.80-t191.20.__DEBUG_FINAL_portrait_FINAL"
        )
        assert start == pytest.approx(180.80)
        assert end == pytest.approx(191.20)

    def test_game_prefix(self):
        start, end = parse_timestamps("003__2025-11-01__Team_A_vs_B__SHOT__t580.10-t592.30")
        assert start == pytest.approx(580.10)
        assert end == pytest.approx(592.30)

    def test_normalize_60x(self):
        # 18900/60 = 315, 20340/60 = 339
        start, end = parse_timestamps("001__GAME__SHOT__t18900.00-t20340.00")
        assert start == pytest.approx(315.0)
        assert end == pytest.approx(339.0)

    def test_normalize_1440x(self):
        # 2937600/1440 = 2040, 3002400/1440 = 2085
        start, end = parse_timestamps("027__GAME__SHOT__t2937600.00-t3002400.00")
        assert start == pytest.approx(2040.0)
        assert end == pytest.approx(2085.0, rel=0.01)

    def test_plausible_values_untouched(self):
        # Values under 10800 should be returned as-is
        start, end = parse_timestamps("001__GAME__SHOT__t3500.00-t3520.00")
        assert start == pytest.approx(3500.0)
        assert end == pytest.approx(3520.0)

    def test_dot_thousands_separator(self):
        # t1.855.00 = 1855.00 seconds (dot as thousands separator)
        start, end = parse_timestamps(
            "005__GOAL__t1.855.00-t1.870.00.__CINEMATIC"
        )
        assert start == pytest.approx(1855.0)
        assert end == pytest.approx(1870.0)

    def test_dot_thousands_separator_larger(self):
        # t2.694.00 = 2694.00 seconds
        start, end = parse_timestamps(
            "010__PRESSURE__t2.694.00-t2.706.00.__CINEMATIC"
        )
        assert start == pytest.approx(2694.0)
        assert end == pytest.approx(2706.0)

    def test_dot_thousands_no_false_positive(self):
        # Normal decimal t640.00 should NOT be affected
        start, end = parse_timestamps("004__PRESSURE__t640.00-t663.00")
        assert start == pytest.approx(640.0)
        assert end == pytest.approx(663.0)

    def test_overlay_suffix_with_hyphens(self):
        # Hyphens in OVERLAY parameters (sh-1, dx-64) should not break parsing
        start, end = parse_timestamps(
            "004__PRESSURE__t640.00-t663.00"
            ".__OVERLAY_plan_sh-1_dx-64_dy-128"
            "__localTW_dt-0.02_dx-6_dy-2"
        )
        assert start == pytest.approx(640.0)
        assert end == pytest.approx(663.0)

    def test_overlay_suffix_with_cinematic_chain(self):
        start, end = parse_timestamps(
            "004__PRESSURE__t640.00-t663.00"
            ".__OVERLAY_plan_sh-1_dx-64_dy-128"
            "__localTW_dt-0.02_dx-6_dy-2"
            ".__CINEMATIC.__CINEMATIC"
        )
        assert start == pytest.approx(640.0)
        assert end == pytest.approx(663.0)

    def test_navy_hms_format(self):
        # 0_06_37 = 397s, 0_07_10 = 430s
        start, end = parse_timestamps(
            "001__Pressure & Build Up with a Great Cross__0_06_37-0_07_10"
        )
        assert start == pytest.approx(397.0)
        assert end == pytest.approx(430.0)

    def test_navy_hms_with_cinematic(self):
        start, end = parse_timestamps(
            "010__Build Up, Cross, Shot & Goal__0_55_26-0_56_02.__CINEMATIC"
        )
        assert start == pytest.approx(3326.0)
        assert end == pytest.approx(3362.0)

    def test_navy_hms_hour_plus(self):
        # 1_00_43 = 3643s, 1_01_22 = 3682s
        start, end = parse_timestamps(
            "011__Build Up__1_00_43-1_01_22"
        )
        assert start == pytest.approx(3643.0)
        assert end == pytest.approx(3682.0)


# --- format_float -------------------------------------------------------------

class TestFormatFloat:
    def test_none(self):
        assert format_float(None) == ""

    def test_number(self):
        assert format_float(10.5) == "10.500"

    def test_string_passthrough(self):
        assert format_float("already") == "already"

    def test_nan(self):
        assert format_float(float("nan")) == ""


# --- compute_overlap_ratio ---------------------------------------------------

class TestOverlapRatio:
    def test_full_overlap(self):
        assert compute_overlap_ratio(0, 10, 0, 10) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_overlap_ratio(0, 5, 10, 15) == pytest.approx(0.0)

    def test_partial(self):
        ratio = compute_overlap_ratio(0, 10, 5, 15)
        assert ratio == pytest.approx(5.0 / 15.0)

    def test_none_values(self):
        assert compute_overlap_ratio(None, 10, 0, 10) is None

    def test_containment_low_iou(self):
        """A short clip entirely within a longer one has low IoU but high containment."""
        # Clip A: 2040-2058 (18s), Clip B: 2040-2085 (45s)
        # IoU = 18/45 = 0.4, Containment = 18/18 = 1.0
        ratio = compute_overlap_ratio(2040, 2058, 2040, 2085)
        assert ratio == pytest.approx(18 / 45)  # IoU only
        # But the soft dupe detection should still catch this via containment


class TestContainmentDupe:
    """Verify that soft dupe detection catches containment cases."""

    def test_contained_clip_is_soft_dupe(self):
        """A clip fully contained in another should be flagged as soft dupe."""
        records = [
            ClipRecord(
                clip_path=Path("a.mp4"), clip_rel="out/atomic_clips/g/a.mp4",
                clip_name="a.mp4", clip_stem="a",
                t_start_s=2040, t_end_s=2058, master_rel="m",
                sha1_64="aaa", duration_s=18, width=1920, height=1080,
                fps="30", created_at_utc="2025-01-01", created_ts=0,
            ),
            ClipRecord(
                clip_path=Path("b.mp4"), clip_rel="out/atomic_clips/g/b.mp4",
                clip_name="b.mp4", clip_stem="b",
                t_start_s=2040, t_end_s=2085, master_rel="m",
                sha1_64="bbb", duration_s=45, width=1920, height=1080,
                fps="30", created_at_utc="2025-01-01", created_ts=0,
            ),
        ]
        _, _, soft = compute_duplicate_groups(records)
        assert soft >= 1


# --- to_repo_relative ---------------------------------------------------------

class TestToRepoRelative:
    def test_path_under_root(self, tmp_path, monkeypatch):
        import tools.catalog as cat
        monkeypatch.setattr(cat, "ROOT", tmp_path)
        p = tmp_path / "out" / "clips" / "file.mp4"
        p.parent.mkdir(parents=True)
        p.touch()
        assert to_repo_relative(p) == "out/clips/file.mp4"

    def test_symlink_stays_relative(self, tmp_path, monkeypatch):
        """Symlinked directories should still produce repo-relative paths."""
        import tools.catalog as cat
        monkeypatch.setattr(cat, "ROOT", tmp_path)
        real_dir = tmp_path / "real_data"
        real_dir.mkdir()
        (real_dir / "clip.mp4").touch()
        link_dir = tmp_path / "out" / "clips"
        link_dir.parent.mkdir(parents=True)
        link_dir.symlink_to(real_dir)
        clip = link_dir / "clip.mp4"
        rel = to_repo_relative(clip)
        assert rel == "out/clips/clip.mp4"


# --- is_canonical_rel ---------------------------------------------------------

class TestIsCanonicalRel:
    def test_canonical(self):
        assert is_canonical_rel("out/atomic_clips/2025-09-13__TSC/clip.mp4")

    def test_not_canonical(self):
        assert not is_canonical_rel("out/atomic_clips/clip.mp4")
        assert not is_canonical_rel("other/clip.mp4")


# --- sha1_64 ------------------------------------------------------------------

class TestSha1_64:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h1 = sha1_64(f)
        h2 = sha1_64(f)
        assert h1 == h2
        assert len(h1) == 16  # 64-bit = 16 hex chars


# --- probe_video (OpenCV fallback) --------------------------------------------

class TestProbeVideo:
    def test_valid_clip(self, tmp_path):
        clip = _make_clip(tmp_path / "test.mp4", 320, 240, 25.0, 1.0)
        meta = probe_video(clip)
        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["stream_exists"] is True
        assert meta["duration_s"] is not None
        assert meta["duration_s"] > 0.5

    def test_nonexistent(self, tmp_path):
        meta = probe_video(tmp_path / "nope.mp4")
        assert meta["stream_exists"] is False


# --- Full pipeline integration (uses tmp directories) -------------------------

@pytest.fixture
def catalog_env(tmp_path, monkeypatch):
    """Set up isolated catalog directories for testing."""
    import tools.catalog as cat

    atomic_dir = tmp_path / "out" / "atomic_clips"
    catalog_dir = tmp_path / "out" / "catalog"
    sidecar_dir = catalog_dir / "sidecar"

    monkeypatch.setattr(cat, "ROOT", tmp_path)
    monkeypatch.setattr(cat, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(cat, "ATOMIC_DIR", atomic_dir)
    monkeypatch.setattr(cat, "GAMES_DIR", tmp_path / "out" / "games")
    monkeypatch.setattr(cat, "CATALOG_DIR", catalog_dir)
    monkeypatch.setattr(cat, "SIDE_CAR_DIR", sidecar_dir)
    monkeypatch.setattr(cat, "ATOMIC_INDEX_PATH", catalog_dir / "atomic_index.csv")
    monkeypatch.setattr(cat, "PIPELINE_STATUS_PATH", catalog_dir / "pipeline_status.csv")
    monkeypatch.setattr(cat, "DUPLICATES_PATH", catalog_dir / "duplicates.csv")
    monkeypatch.setattr(cat, "MASTERS_INDEX_PATH", catalog_dir / "masters_index.csv")
    monkeypatch.setattr(cat, "TRASH_ROOT", tmp_path / "out" / "_trash" / "atomic_dupes")
    monkeypatch.setattr(cat, "CLEANUP_LOG_PATH", catalog_dir / "cleanup_log.txt")

    return {
        "root": tmp_path,
        "atomic_dir": atomic_dir,
        "catalog_dir": catalog_dir,
        "sidecar_dir": sidecar_dir,
    }


class TestCatalogPipeline:
    def test_scan_empty(self, catalog_env):
        records, masters, failures = scan_atomic_clips()
        assert records == []
        assert masters == {}
        assert failures == 0

    def test_scan_with_clips(self, catalog_env):
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10.0-t20.0.mp4", duration=2.0)
        _make_clip(game_dir / "002__SHOT__t50.0-t60.0.mp4", duration=1.5)

        records, masters, failures = scan_atomic_clips()
        assert len(records) == 2
        assert all(r.width == 640 for r in records)
        assert all(r.height == 360 for r in records)
        assert all(r.sha1_64 for r in records)

    def test_timestamps_extracted(self, catalog_env):
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t155.50-t166.40.mp4")

        records, _, _ = scan_atomic_clips()
        assert len(records) == 1
        assert records[0].t_start_s == pytest.approx(155.50)
        assert records[0].t_end_s == pytest.approx(166.40)

    def test_write_and_read_index(self, catalog_env):
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        _make_clip(game_dir / "002__SHOT__t30-t40.mp4")

        records, _, _ = scan_atomic_clips()
        changed = write_atomic_index(records)
        assert changed == 2

        index_path = catalog_env["catalog_dir"] / "atomic_index.csv"
        assert index_path.exists()
        with index_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert all(h in rows[0] for h in ATOMIC_HEADERS)

    def test_rebuild_full(self, catalog_env):
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        _make_clip(game_dir / "002__SHOT__t30-t40.mp4")
        _make_clip(game_dir / "003__SHOT__t50-t60.mp4")

        result = rebuild_atomic_index()
        assert result.scanned == 3
        assert result.indexed == 3
        assert result.changed == 3
        assert result.hard_dupes == 0

    def test_sidecars_created(self, catalog_env):
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")

        rebuild_atomic_index()

        sidecar_files = list(catalog_env["sidecar_dir"].glob("*.json"))
        assert len(sidecar_files) >= 1
        data = json.loads(sidecar_files[0].read_text())
        assert "meta" in data
        assert data["meta"]["width"] == 640
        assert data["meta"]["height"] == 360

    def test_duplicate_detection_hard(self, catalog_env):
        """Two identical clips should be detected as hard duplicates."""
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        # Copy to create exact duplicate
        import shutil
        dup_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        shutil.copy2(clip1, dup_dir / "001__GOAL__t10-t20.mp4")

        records, _, _ = scan_atomic_clips()
        dupes, hard, soft = compute_duplicate_groups(records)
        assert hard >= 1

    def test_pipeline_status_tracking(self, catalog_env):
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")

        mark_upscaled(clip, clip.parent / "001__x2.mp4", scale=2, model="RealESRGAN_x2plus")
        data = load_sidecar(clip)
        assert data["steps"]["upscale"]["done"] is True
        assert data["steps"]["upscale"]["scale"] == 2

        mark_branded(clip, clip.parent / "001__FINAL.mp4", brand="TSC")
        data = load_sidecar(clip)
        assert data["steps"]["follow_crop_brand"]["done"] is True
        assert data["steps"]["follow_crop_brand"]["brand"] == "TSC"

    def test_report(self, catalog_env):
        from tools.catalog import generate_report
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        rebuild_atomic_index()

        report = generate_report()
        assert report["total_clips"] == 1
        assert report["upscaled"] == 0
        assert report["branded"] == 0

    def test_write_duplicates_from_index(self, catalog_env):
        """write_duplicates_from_index should detect hard dupes from the index."""
        import shutil
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        dup_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        shutil.copy2(clip1, dup_dir / "001__GOAL__t10-t20.mp4")

        # First rebuild to populate atomic_index.csv
        rebuild_atomic_index()

        hard, soft = write_duplicates_from_index()
        assert hard >= 1
        dupes_csv = catalog_env["catalog_dir"] / "duplicates.csv"
        assert dupes_csv.exists()

    def test_normalize_tree_dry_run(self, catalog_env):
        """Dry-run should preview moves without modifying files."""
        import shutil
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        dup_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        dup_path = dup_dir / "001__GOAL__t10-t20.mp4"
        shutil.copy2(clip1, dup_path)

        rebuild_atomic_index()

        result = normalize_tree(dry_run=True, force=False, purge=False)
        # Dry run should not move anything
        assert result["moved"] == 0
        # But duplicate file should still exist
        assert dup_path.exists()

    def test_normalize_tree_force(self, catalog_env):
        """Force mode should move duplicate files to trash."""
        import shutil
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        dup_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        dup_path = dup_dir / "001__GOAL__t10-t20.mp4"
        shutil.copy2(clip1, dup_path)

        rebuild_atomic_index()

        result = normalize_tree(dry_run=False, force=True, purge=False)
        assert result["moved"] >= 1
        # Duplicate should no longer exist at original location
        assert not dup_path.exists()
        # Canonical should still exist
        assert clip1.exists()

    def test_normalize_tree_requires_force(self, catalog_env):
        """Without --force (and not --dry-run), should raise CatalogError."""
        import shutil
        from tools.catalog import CatalogError
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        dup_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        shutil.copy2(clip1, dup_dir / "001__GOAL__t10-t20.mp4")

        rebuild_atomic_index()

        with pytest.raises(CatalogError, match="--force"):
            normalize_tree(dry_run=False, force=False, purge=False)

    def test_normalize_tree_no_dupes(self, catalog_env):
        """When no duplicates exist, normalize_tree returns zeros."""
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        _make_clip(game_dir / "002__SHOT__t30-t40.mp4")

        rebuild_atomic_index()

        result = normalize_tree(dry_run=True, force=False, purge=False)
        assert result["moved"] == 0

    def test_load_duplicates(self, catalog_env):
        """load_duplicates should return records from duplicates.csv."""
        import shutil
        game_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game"
        clip1 = _make_clip(game_dir / "001__GOAL__t10-t20.mp4")
        dup_dir = catalog_env["atomic_dir"] / "2025-01-01__Test_Game_copy"
        dup_dir.mkdir(parents=True)
        shutil.copy2(clip1, dup_dir / "001__GOAL__t10-t20.mp4")

        rebuild_atomic_index()

        dupes = load_duplicates()
        assert len(dupes) >= 1
        assert dupes[0].reason == "hard"
        assert dupes[0].overlap_ratio == 1.0
