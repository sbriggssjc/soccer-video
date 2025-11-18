"""Unified offline follow + portrait pipeline entrypoint."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = REPO_ROOT / "out" / "reports" / "pipeline_status.csv"
PORTRAIT_ROOT = REPO_ROOT / "out" / "portrait_reels" / "clean"


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


def run_render(job: ClipJob, args: argparse.Namespace, ball_path: Optional[Path]) -> Path:
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
    if args.extra:
        cmd.extend(args.extra)
    logging.info("Rendering %s", job.clip_id)
    subprocess.run(cmd, check=True)
    return output


def run_brand(output: Path, args: argparse.Namespace) -> None:
    if not args.brand_script:
        return
    script = args.brand_script
    if not script.exists():
        logging.warning("Branding script %s not found", script)
        return
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        logging.warning("Skipping branding for %s; PowerShell not available", output)
        return
    cmd = [shell, "-NoProfile", "-File", os.fspath(script), "-In", os.fspath(output), "-OutPath", os.fspath(output)]
    logging.info("Branding %s", output)
    subprocess.run(cmd, check=True)


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


def process_jobs(jobs: Sequence[ClipJob], args: argparse.Namespace) -> List[JobResult]:
    results: List[JobResult] = []
    for job in jobs:
        ball_path = find_ball_path(job)
        if not ball_path:
            logging.warning("No telemetry found for %s; planner will fall back to reactive follow", job.clip_id)
        try:
            output = run_render(job, args, ball_path)
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    jobs = discover_from_selection(args.selection)
    if args.clip:
        jobs.extend(discover_from_args(args.clip))
    if not jobs:
        parser.error("No clips to process; provide --selection or --clip")

    results = process_jobs(jobs, args)
    success = sum(1 for r in results if r.success)
    logging.info("%s/%s clips rendered", success, len(results))

    if success and not args.skip_catalog:
        rebuild_catalog()
    run_cleanup(args)


if __name__ == "__main__":
    main()
