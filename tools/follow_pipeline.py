"""Unified offline follow + portrait pipeline entrypoint.

"""Unified batch follow pipeline with telemetry-aware planning.

How to test
-----------

Single-clip telemetry test (keeps the ball in frame)::

    cd C:\Users\scott\soccer-video
    $Atomic = "C:\\Users\\scott\\soccer-video\\out\\atomic_clips\\_quarantine\\001__SHOT__t155.50-t166.40.mp4"
    python tools\follow_pipeline.py \
      --clip "$Atomic" \
      --preset wide_follow \
      --portrait 1080x1920 \
      --variant WIDE \
      --brand-script "tools\tsc_brand.ps1"

Expectation: ``out\portrait_reels\clean\001__SHOT__t155.50-t166.40__WIDE_portrait_FINAL.mp4``
is rendered with the ball pinned in the portrait window and a plan such as
``out\plans\001__SHOT__t155.50-t166.40.plan.json`` is created.

Reactive fallback (ignore telemetry)::

    python tools\follow_pipeline.py \
      --clip "$Atomic" \
      --preset wide_follow \
      --portrait 1080x1920 \
      --variant WIDE \
      --brand-script "tools\tsc_brand.ps1" \
      --force-reactive

Telemetry failure debug (fail fast if telemetry missing)::

    python tools\follow_pipeline.py \
      --clip "$Atomic" \
      --preset wide_follow \
      --portrait 1080x1920 \
      --variant WIDE \
      --brand-script "tools\tsc_brand.ps1" \
      --force-telemetry

Key flags:

* ``--preset`` selects the ``render_follow_unified`` preset.
* ``--portrait`` sets the portrait output canvas (e.g. ``1080x1920``).
* ``--brand-script`` points at ``tools/tsc_brand.ps1`` (or equivalent).
* ``--cleanup`` runs ``Cleanup-Intermediates.ps1`` after all clips finish.
* ``--extra`` forwards additional args directly to ``render_follow_unified.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from tools import ball_telemetry
from tools.offline_portrait_planner import plan_camera_from_ball, save_plan

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "out" / "reports" / "pipeline_status.csv"
PORTRAIT_ROOT = REPO_ROOT / "out" / "portrait_reels" / "clean"
PLAN_ROOT = REPO_ROOT / "out" / "plans"


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "selected", "ready"}


def _maybe_path(text: Optional[str]) -> Optional[Path]:
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def parse_portrait(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    txt = str(value).lower().replace("x", "x")
    parts = txt.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Portrait must be formatted as WIDTHxHEIGHT")
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Portrait dimensions must be integers") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Portrait dimensions must be positive")
    return f"{w}x{h}"


def portrait_dims(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    txt = str(value).lower()
    parts = txt.split("x")
    if len(parts) != 2:
        return None
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return w, h


def _parse_rate(text: Optional[str]) -> float:
    if not text:
        return 0.0
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def probe_video(path: Path) -> Tuple[int, int, float]:
    width = 0
    height = 0
    fps = 0.0
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate",
        "-of",
        "json",
        os.fspath(path),
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        data = json.loads(output)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return width, height, fps or 30.0
    streams = data.get("streams", [])
    if streams:
        stream = streams[0] or {}
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = _parse_rate(stream.get("avg_frame_rate"))
    return width, height, fps if fps > 0 else 30.0


def plan_output_path(clip_id: str) -> Path:
    PLAN_ROOT.mkdir(parents=True, exist_ok=True)
    return PLAN_ROOT / f"{clip_id}.plan.json"


@dataclass
class ClipJob:
    clip_id: str
    atomic_path: Path
    match_key: str = ""
    ball_path: Optional[Path] = None


@dataclass
class JobResult:
    job: ClipJob
    success: bool
    output: Optional[Path]
    message: str = ""


def discover_from_selection(selection_csv: Path) -> List[ClipJob]:
    jobs: List[ClipJob] = []
    if not selection_csv.exists():
        logging.info("Selection CSV %s not found; relying on explicit --clip arguments", selection_csv)
        return jobs
    with selection_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            if not any(_truthy(row.get(flag)) for flag in ("Selected", "NeedsPortrait", "DoRender", "Render")):
                continue
            atomic_path = row.get("AtomicPath") or row.get("ClipPath")
            if not atomic_path:
                continue
            clip_id = row.get("ClipID") or Path(atomic_path).stem
            match_key = row.get("MatchKey", "")
            p = _maybe_path(atomic_path)
            if not p or not p.exists():
                logging.warning("Atomic clip missing for %s (%s)", clip_id, atomic_path)
                continue
            jobs.append(ClipJob(clip_id=clip_id, atomic_path=p, match_key=match_key))
    return jobs


def discover_from_args(paths: Iterable[str]) -> List[ClipJob]:
    jobs: List[ClipJob] = []
    for item in paths:
        if not item:
            continue
        p = _maybe_path(item)
        if not p or not p.exists():
            logging.warning("Clip %s not found", item)
            continue
        jobs.append(ClipJob(clip_id=p.stem, atomic_path=p))
    return jobs


def find_ball_path(job: ClipJob) -> Optional[Path]:
    candidates = []
    stem = job.atomic_path.stem
    parent = job.atomic_path.with_suffix("")
    candidates.append(job.atomic_path.with_suffix(".ball_path.jsonl"))
    candidates.append(job.atomic_path.with_suffix(".ball.jsonl"))
    candidates.append(job.atomic_path.with_suffix(".telemetry.jsonl"))
    candidates.append(parent.with_suffix(".jsonl"))
    work_root = REPO_ROOT / "out" / "autoframe_work"
    candidates.append(work_root / stem / f"{stem}.ball_path.jsonl")
    candidates.append(work_root / stem / "ball_path.jsonl")
    candidates.append(work_root / stem / f"{stem}.telemetry.jsonl")
    for cand in candidates:
        if cand and cand.exists():
            return cand
    return None


def default_output_path(clip_id: str, variant: str) -> Path:
    PORTRAIT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = variant.upper() if variant else "CINEMATIC"
    name = f"{clip_id}__{suffix}_portrait_FINAL.mp4"
    return PORTRAIT_ROOT / name


def run_render(
    job: ClipJob,
    args: argparse.Namespace,
    ball_path: Optional[Path],
    plan_path: Optional[Path],
) -> Path:
    output = default_output_path(job.clip_id, args.variant)
    render_script = REPO_ROOT / "tools" / "render_follow_unified.py"
    telemetry_dir = REPO_ROOT / "out" / "autoframe_work" / job.clip_id
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    telemetry_file = telemetry_dir / f"{job.clip_id}.telemetry.jsonl"
    cmd = [
        sys.executable,
        os.fspath(render_script),
        "--in",
        os.fspath(job.atomic_path),
        "--out",
        os.fspath(output),
        "--preset",
        args.preset,
        "--telemetry",
        os.fspath(telemetry_file),
    ]
    if args.portrait:
        cmd.extend(["--portrait", args.portrait])
    if ball_path:
        cmd.extend(["--ball-path", os.fspath(ball_path)])
    if plan_path:
        cmd.extend(["--plan", os.fspath(plan_path)])
    if args.extra:
        cmd.extend(args.extra)
    logging.info("Rendering %s", job.clip_id)
    subprocess.run(cmd, check=True)
    return output


def build_plan_for_job(
    job: ClipJob,
    args: argparse.Namespace,
    portrait: Optional[Tuple[int, int]],
) -> Optional[Path]:
    if not portrait:
        return None
    telemetry_override = getattr(args, "telemetry_path", None)
    samples, telemetry_path = ball_telemetry.load_ball_telemetry_for_clip(
        job.atomic_path,
        match_key=job.match_key,
        telemetry_path=telemetry_override,
    )
    if not ball_telemetry.telemetry_is_usable(samples):
        raise RuntimeError(
            f"Telemetry missing or low confidence ({job.clip_id})"
        )
    width, height, fps = probe_video(job.atomic_path)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not probe video dimensions for {job.clip_id}")
    portrait_w, portrait_h = portrait
    aspect = portrait_w / portrait_h
    pad_frac = max(0.0, float(args.plan_pad_frac))
    inner_band = float(max(0.2, min(0.95, args.plan_inner_band)))
    zoom_max = max(1.0, float(args.plan_zoom_max))
    smooth = float(max(0.0, min(0.99, args.plan_smooth)))

    keyframes = plan_camera_from_ball(
        samples,
        width,
        height,
        out_aspect=aspect,
        pad_frac=pad_frac,
        zoom_max=zoom_max,
        smooth_strength=smooth,
        inner_band_frac=inner_band,
        fps=fps,
    )
    plan_path = plan_output_path(job.clip_id)
    meta = {
        "clip_id": job.clip_id,
        "telemetry_path": os.fspath(telemetry_path) if telemetry_path else None,
        "source": {"width": width, "height": height, "fps": fps},
        "portrait": {"width": portrait_w, "height": portrait_h},
    }
    save_plan(plan_path, keyframes, meta=meta)
    logging.info(
        "Planned portrait path for %s (%s samples) â†' %s",
        job.clip_id,
        ball_telemetry.summarise(samples or []),
        plan_path,
    )
    return plan_path


def run_brand(output: Path, args: argparse.Namespace) -> None:
    """
    Run the external PowerShell branding script, writing to a temporary
    file and then renaming back to *output* so ffmpeg never has to
    read/write the same path.
    """
    if not args.brand_script:
        return

    script = Path(args.brand_script)
    if not script.exists():
        logger.warning("Brand script %s does not exist; skipping brand", script)
        return

    # Choose PowerShell host
    shell = os.environ.get(
        "POWERSHELL",
        "powershell" if os.name == "nt" else "pwsh",
    )

    # Temporary branded output (same folder, different name)
    tmp = output.with_name(output.stem + ".__BRANDTMP" + output.suffix)
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError as e:
            logger.warning("Could not remove stale brand tmp %s: %s", tmp, e)

    cmd = [
        shell,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        os.fspath(script),
        "-In",
        os.fspath(output),
        "-Out",              # <-- tsc_brand.ps1 expects -Out
        os.fspath(tmp),
    ]

    logger.info(
        "Branding %s â†' %s",
        os.fspath(output),
        os.fspath(tmp),
    )
    subprocess.run(cmd, check=True)

    # Replace the original with the branded result
    try:
        tmp.replace(output)
    except OSError:
        import shutil

        shutil.move(os.fspath(tmp), os.fspath(output))


def run_cleanup(args: argparse.Namespace) -> None:
    if not args.cleanup:
        return
    script = REPO_ROOT / "tools" / "Cleanup-Intermediates.ps1"
    if not script.exists():
        logging.warning("Cleanup script %s missing", script)
        return
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        logging.warning("Skipping cleanup; PowerShell not available")
        return
    cmd = [shell, "-NoProfile", "-File", os.fspath(script), "-Root", os.fspath(REPO_ROOT)]
    logging.info("Cleaning intermediates")
    subprocess.run(cmd, check=True)


def rebuild_catalog() -> None:
    catalog_script = REPO_ROOT / "tools" / "build_clips_catalog.py"
    if not catalog_script.exists():
        logging.warning("Catalog script %s missing", catalog_script)
        return
    subprocess.run([sys.executable, os.fspath(catalog_script)], check=True)


def process_jobs(
    jobs: Sequence[ClipJob],
    args: argparse.Namespace,
    portrait: Optional[Tuple[int, int]],
) -> List[JobResult]:
    results: List[JobResult] = []
    for job in jobs:
        ball_path = find_ball_path(job)
        plan_path: Optional[Path] = None
        if not args.force_reactive:
            try:
                plan_path = build_plan_for_job(job, args, portrait)
            except RuntimeError as exc:
                if args.force_telemetry:
                    logging.error("Telemetry required but unavailable for %s: %s", job.clip_id, exc)
                    results.append(
                        JobResult(job=job, success=False, output=None, message=str(exc))
                    )
                    continue
                logging.warning(
                    "No telemetry plan for %s (%s); falling back to reactive follow",
                    job.clip_id,
                    exc,
                )
        try:
            output = run_render(job, args, ball_path, plan_path)
        except subprocess.CalledProcessError as exc:
            logging.error("Render failed for %s", job.clip_id)
            results.append(JobResult(job=job, success=False, output=None, message=str(exc)))
            continue
        try:
            run_brand(output, args)
        except subprocess.CalledProcessError as exc:
            logging.error("Branding failed for %s", job.clip_id)
            results.append(JobResult(job=job, success=False, output=output, message=str(exc)))
            continue
        results.append(JobResult(job=job, success=True, output=output, message="rendered"))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified offline follow pipeline")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION, help="CSV with ClipID + AtomicPath columns")
    parser.add_argument("--clip", action="append", help="Additional atomic clip to process")
    parser.add_argument("--preset", default="cinematic", help="Render preset")
    parser.add_argument("--portrait", type=parse_portrait, help="Portrait canvas (e.g. 1080x1920)")
    parser.add_argument("--variant", default="CINEMATIC", help="Suffix tag for final reels")
    parser.add_argument("--brand-script", type=Path, help="Optional PowerShell branding script to run per clip")
    parser.add_argument("--cleanup", action="store_true", help="Run Cleanup-Intermediates.ps1 after processing")
    parser.add_argument("--skip-catalog", action="store_true", help="Skip catalog rebuild")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed to render_follow_unified.py")
    parser.add_argument("--force-reactive", action="store_true", help="Skip telemetry planning even if files exist")
    parser.add_argument(
        "--force-telemetry",
        action="store_true",
        help="Fail the clip when telemetry/plan generation fails",
    )
    parser.add_argument("--telemetry", type=Path, help="Override telemetry file for testing")
    parser.add_argument("--plan-pad-frac", type=float, default=0.2, help="Planner padding fraction (0-1)")
    parser.add_argument("--plan-inner-band", type=float, default=0.6, help="Planner inner band fraction")
    parser.add_argument("--plan-zoom-max", type=float, default=2.4, help="Planner zoom max")
    parser.add_argument("--plan-smooth", type=float, default=0.2, help="Planner smoothing strength (0-1)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    args.telemetry_path = None
    if args.telemetry:
        maybe = _maybe_path(os.fspath(args.telemetry))
        if maybe and maybe.exists():
            args.telemetry_path = maybe
        else:
            logging.warning("Telemetry override %s not found", args.telemetry)

    jobs = discover_from_selection(args.selection)
    if args.clip:
        jobs.extend(discover_from_args(args.clip))
    if not jobs:
        parser.error("No clips to process; provide --selection or --clip")

    portrait = portrait_dims(args.portrait)

    results = process_jobs(jobs, args, portrait)
    success = sum(1 for r in results if r.success)
    logging.info("%s/%s clips rendered", success, len(results))

    if success and not args.skip_catalog:
        rebuild_catalog()
    run_cleanup(args)


if __name__ == "__main__":
    main()

