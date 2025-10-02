from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from soccer_highlights.ball_tracker import BallTrack


_AUTOFRAME_SPEC = importlib.util.spec_from_file_location("autoframe", _ROOT / "autoframe.py")
if _AUTOFRAME_SPEC is None or _AUTOFRAME_SPEC.loader is None:
    raise ImportError("Unable to load autoframe module for tests")
autoframe_module = importlib.util.module_from_spec(_AUTOFRAME_SPEC)
sys.modules.setdefault("autoframe", autoframe_module)
_AUTOFRAME_SPEC.loader.exec_module(autoframe_module)  # type: ignore[attr-defined]
choose_target_center = getattr(autoframe_module, "choose_target_center")


def _make_track(frame: int, center: Tuple[float, float], conf: float) -> BallTrack:
    cx, cy = center
    return BallTrack(
        frame=frame,
        cx=float(cx),
        cy=float(cy),
        width=14.0,
        height=14.0,
        conf=float(conf),
        raw_cx=float(cx),
        raw_cy=float(cy),
        raw_width=14.0,
        raw_height=14.0,
        raw_conf=float(conf),
    )


def test_choose_target_center_prefers_ball_when_confident() -> None:
    motion = (320.0, 180.0)
    confident_ball = _make_track(0, (150.0, 90.0), 0.8)
    fused_x, fused_y, used = choose_target_center(motion, confident_ball, 0.35)
    assert used
    assert fused_x == pytest.approx(confident_ball.cx)
    assert fused_y == pytest.approx(confident_ball.cy)

    faint_ball = _make_track(1, (240.0, 200.0), 0.2)
    fused_x, fused_y, used = choose_target_center(motion, faint_ball, 0.35)
    assert not used
    assert fused_x == pytest.approx(motion[0])
    assert fused_y == pytest.approx(motion[1])

    seq_motion: List[Tuple[float, float]] = [(300.0, 200.0), (302.0, 203.0), (305.0, 207.0)]
    seq_ball: List[Optional[BallTrack]] = [None, _make_track(2, (260.0, 150.0), 0.6), None]
    fused = [choose_target_center(m, b, 0.4) for m, b in zip(seq_motion, seq_ball)]
    assert fused[0][2] is False
    assert fused[1][2] is True and fused[1][0] == pytest.approx(260.0)
    assert fused[2][2] is False


@pytest.mark.slow
def test_fit_expr_and_ffmpeg_pipeline(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for pipeline smoke test")

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        pytest.skip("OpenCV and numpy required for pipeline smoke test")

    width, height, fps = 640, 360, 30.0
    frame_count = 60
    video_path = tmp_path / "sample.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot open MP4 writer")

    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cx = int(width * 0.2 + idx * 2)
        cy = int(height * 0.5 + 20 * np.sin(idx / 5.0))
        cv2.circle(frame, (cx, cy), 18, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()

    csv_path = tmp_path / "autoframe.csv"
    subprocess.run(
        [
            sys.executable,
            "autoframe.py",
            "--in",
            str(video_path),
            "--csv",
            str(csv_path),
            "--profile",
            "portrait",
            "--ball-detector",
            "none",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert csv_path.exists()

    vars_path = tmp_path / "autoframe.ps1vars"
    subprocess.run(
        [
            sys.executable,
            "fit_expr.py",
            "--csv",
            str(csv_path),
            "--out",
            str(vars_path),
            "--profile",
            "portrait",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    contents = vars_path.read_text(encoding="utf-8")
    assert "$cxExpr" in contents
    assert "$cyExpr" in contents
    assert "$zExpr" in contents

    cropped_path = tmp_path / "cropped.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "crop=iw/1.25:ih/1.25:(iw-iw/1.25)/2:(ih-ih/1.25)/2",
            "-frames:v",
            str(frame_count),
            str(cropped_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert cropped_path.exists()
    assert cropped_path.stat().st_size > 0
