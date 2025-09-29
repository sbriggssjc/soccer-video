"""Batch runner for ball tracking and auto-framing renders.

This helper replaces the PowerShell one-liner that previously chained
``track_ball_cli.py`` and ``plan_render_cli.py``.  The PowerShell version relied
on the backtick continuation character and inline comments which do not mix
well, triggering errors such as ``Missing expression after unary operator``
when the script was copied verbatim.  By orchestrating the workflow from
Python we avoid shell quoting pitfalls and the code works uniformly on
Windows, macOS, and Linux.

Example
-------

.. code-block:: console

    $ python -m venv .venv
    $ .venv\\Scripts\\activate
    (.venv) $ python scripts/batch_autoframe.py \
        --atomic-dir out/atomic_clips \
        --work-dir out/autoframe_work \
        --out-dir out/reels/tiktok

The defaults mirror the values from the README snippet so running the command
without any flags replicates the original behaviour.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _run(cmd: List[str]) -> None:
    logging.debug("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _extend_with_pairs(args: List[str], key: str, value: object) -> None:
    args.append(key)
    args.append(f"{value}")


@dataclass
class TrackOptions:
    yolo_conf: float
    roi_pad: int
    roi_pad_max: int
    max_miss: int

    def to_argv(self, clip: Path, csv_path: Path) -> List[str]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "track_ball_cli.py"),
            "--inp",
            str(clip),
            "--out_csv",
            str(csv_path),
        ]
        _extend_with_pairs(cmd, "--yolo_conf", self.yolo_conf)
        _extend_with_pairs(cmd, "--roi_pad", self.roi_pad)
        _extend_with_pairs(cmd, "--roi_pad_max", self.roi_pad_max)
        _extend_with_pairs(cmd, "--max_miss", self.max_miss)
        return cmd


@dataclass
class PlanOptions:
    W_out: int
    H_out: int
    slew: float
    accel: float
    zoom_min: float
    zoom_max: float
    zoom_rate: float
    zoom_accel: float
    left_frac: float
    ball_margin: float
    conf_min: float
    miss_jump: float
    lookahead: int
    widen_step: float
    hyst: float

    def to_argv(self, clip: Path, csv_path: Path, out_mp4: Path) -> List[str]:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "plan_render_cli.py"),
            "--clip",
            str(clip),
            "--track_csv",
            str(csv_path),
            "--out_mp4",
            str(out_mp4),
        ]

        _extend_with_pairs(cmd, "--W_out", self.W_out)
        _extend_with_pairs(cmd, "--H_out", self.H_out)
        _extend_with_pairs(cmd, "--slew", self.slew)
        _extend_with_pairs(cmd, "--accel", self.accel)
        _extend_with_pairs(cmd, "--zoom_min", self.zoom_min)
        _extend_with_pairs(cmd, "--zoom_max", self.zoom_max)
        _extend_with_pairs(cmd, "--zoom_rate", self.zoom_rate)
        _extend_with_pairs(cmd, "--zoom_accel", self.zoom_accel)
        _extend_with_pairs(cmd, "--left_frac", self.left_frac)
        _extend_with_pairs(cmd, "--ball_margin", self.ball_margin)
        _extend_with_pairs(cmd, "--conf_min", self.conf_min)
        _extend_with_pairs(cmd, "--miss_jump", self.miss_jump)
        _extend_with_pairs(cmd, "--lookahead", self.lookahead)
        _extend_with_pairs(cmd, "--widen_step", self.widen_step)
        _extend_with_pairs(cmd, "--hyst", self.hyst)
        return cmd


def _iter_clips(path: Path) -> Iterable[Path]:
    return sorted(p for p in path.glob("*.mp4") if p.is_file())


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch auto-frame a directory of clips.")
    parser.add_argument("--atomic-dir", type=Path, default=Path("out/atomic_clips"))
    parser.add_argument("--work-dir", type=Path, default=Path("out/autoframe_work"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/reels/tiktok"))
    parser.add_argument("--skip-track", action="store_true", help="Reuse existing *_ball_track.csv files")
    parser.add_argument("--skip-plan", action="store_true", help="Only run ball tracking")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # track options
    parser.add_argument("--track-yolo-conf", type=_non_negative_float, default=0.10)
    parser.add_argument("--track-roi-pad", type=_non_negative_int, default=280)
    parser.add_argument("--track-roi-pad-max", type=_non_negative_int, default=760)
    parser.add_argument("--track-max-miss", type=_non_negative_int, default=70)

    # plan options
    parser.add_argument("--plan-width", type=_non_negative_int, default=608)
    parser.add_argument("--plan-height", type=_non_negative_int, default=1080)
    parser.add_argument("--plan-slew", type=_non_negative_float, default=120.0)
    parser.add_argument("--plan-accel", type=_non_negative_float, default=320.0)
    parser.add_argument("--plan-zoom-min", type=_positive_float, default=1.00)
    parser.add_argument("--plan-zoom-max", type=_positive_float, default=1.55)
    parser.add_argument("--plan-zoom-rate", type=_positive_float, default=0.10)
    parser.add_argument("--plan-zoom-accel", type=_positive_float, default=0.30)
    parser.add_argument("--plan-left-frac", type=_non_negative_float, default=0.50)
    parser.add_argument("--plan-ball-margin", type=_non_negative_float, default=0.22)
    parser.add_argument("--plan-conf-min", type=_non_negative_float, default=0.25)
    parser.add_argument("--plan-miss-jump", type=_non_negative_float, default=260.0)
    parser.add_argument("--plan-lookahead", type=_non_negative_int, default=10)
    parser.add_argument("--plan-widen-step", type=_non_negative_float, default=0.04)
    parser.add_argument("--plan-hyst", type=_non_negative_float, default=45.0)

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    atomic_dir = args.atomic_dir if args.atomic_dir.is_absolute() else PROJECT_ROOT / args.atomic_dir
    work_dir = args.work_dir if args.work_dir.is_absolute() else PROJECT_ROOT / args.work_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    track_opts = TrackOptions(
        yolo_conf=args.track_yolo_conf,
        roi_pad=args.track_roi_pad,
        roi_pad_max=args.track_roi_pad_max,
        max_miss=args.track_max_miss,
    )

    plan_opts = PlanOptions(
        W_out=args.plan_width,
        H_out=args.plan_height,
        slew=args.plan_slew,
        accel=args.plan_accel,
        zoom_min=args.plan_zoom_min,
        zoom_max=args.plan_zoom_max,
        zoom_rate=args.plan_zoom_rate,
        zoom_accel=args.plan_zoom_accel,
        left_frac=args.plan_left_frac,
        ball_margin=args.plan_ball_margin,
        conf_min=args.plan_conf_min,
        miss_jump=args.plan_miss_jump,
        lookahead=args.plan_lookahead,
        widen_step=args.plan_widen_step,
        hyst=args.plan_hyst,
    )

    clips = list(_iter_clips(atomic_dir))
    if not clips:
        logging.warning("No clips found in %s", atomic_dir)
        return 0

    for clip in clips:
        base = clip.stem
        logging.info("Processing clip: %s", base)

        track_csv = work_dir / f"{base}_ball_track.csv"
        out_mp4 = out_dir / f"{base}__BALLTRACK.mp4"

        if not args.skip_track and (args.overwrite or not track_csv.exists()):
            logging.info("  tracking ball → %s", track_csv.relative_to(work_dir.parent))
            _run(track_opts.to_argv(clip, track_csv))
        elif track_csv.exists():
            logging.info("  skipping track (exists)")
        else:
            logging.warning("  track CSV missing and --skip-track requested; skipping clip")
            continue

        if args.skip_plan:
            logging.info("  skipping planner per flag")
            continue

        if not args.overwrite and out_mp4.exists():
            logging.info("  output already exists; skipping (use --overwrite to rerun)")
            continue

        logging.info("  rendering plan → %s", out_mp4.relative_to(out_dir.parent))
        _run(plan_opts.to_argv(clip, track_csv, out_mp4))

    logging.info("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
