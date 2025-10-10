"""Catalog and pipeline tracking utilities for the soccer-video portrait reel pipeline.

This module scans curated atomic clips, records metadata in concise CSV catalogs,
and maintains per-clip JSON sidecars that store provenance for each processing
step. It is designed for Windows/PowerShell workflows but works cross-platform.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"
ATOMIC_DIR = OUT_DIR / "atomic_clips"
CATALOG_DIR = OUT_DIR / "catalog"
SIDE_CAR_DIR = CATALOG_DIR / "sidecar"
ATOMIC_INDEX_PATH = CATALOG_DIR / "atomic_index.csv"
PIPELINE_STATUS_PATH = CATALOG_DIR / "pipeline_status.csv"

ATOMIC_HEADERS = [
    "clip_path",
    "clip_name",
    "created_at",
    "duration_s",
    "width",
    "height",
    "fps",
    "sha1_64",
    "tags",
]

PIPELINE_HEADERS = [
    "clip_path",
    "upscale_done_at",
    "upscaled_path",
    "follow_brand_done_at",
    "branded_path",
    "last_error",
    "last_run_at",
]


class CatalogError(RuntimeError):
    """Raised when catalog operations fail."""


def ensure_catalog_dirs() -> None:
    """Ensure that catalog directories exist."""

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    SIDE_CAR_DIR.mkdir(parents=True, exist_ok=True)


def sha1_64(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a 64-bit (16 hex characters) SHA1 digest for *path*.

    The function reads the file in chunks for efficiency. While the digest is
    truncated to the first 16 hex characters, it remains stable for auditing.
    """

    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _run_ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, check=True, text=True, encoding="utf-8"
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise CatalogError(f"ffprobe failed for {path!s}: {exc}") from exc
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise CatalogError(f"Unable to parse ffprobe output for {path!s}") from exc


def probe_clip(path: Path) -> dict:
    """Return basic metadata for *path* using ffprobe."""

    info = _run_ffprobe(path)
    streams = info.get("streams") or []
    if not streams:
        raise CatalogError(f"No video stream metadata returned for {path!s}")
    stream = streams[0]

    duration = stream.get("duration") or info.get("format", {}).get("duration")
    try:
        duration_value = round(float(duration), 3) if duration is not None else ""
    except (TypeError, ValueError):
        duration_value = ""

    return {
        "duration_s": duration_value,
        "width": stream.get("width", ""),
        "height": stream.get("height", ""),
        "fps": stream.get("r_frame_rate", ""),
    }


def iso_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def read_catalog(path: Path, key: str) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row[key]: row for row in reader}


def write_catalog(path: Path, headers: Iterable[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def update_atomic_index(clip_path: Path, metadata: dict) -> bool:
    ensure_catalog_dirs()
    rows = read_catalog(ATOMIC_INDEX_PATH, "clip_path")

    key = str(clip_path)
    existing = rows.get(key, {})
    row = {header: existing.get(header, "") for header in ATOMIC_HEADERS}

    row.update(
        {
            "clip_path": key,
            "clip_name": clip_path.name,
            "created_at": metadata.get("created_at", ""),
            "duration_s": metadata.get("duration_s", ""),
            "width": metadata.get("width", ""),
            "height": metadata.get("height", ""),
            "fps": metadata.get("fps", ""),
            "sha1_64": metadata.get("sha1_64", ""),
        }
    )
    # Preserve any tag annotations from the existing row.
    if "tags" not in row:
        row["tags"] = existing.get("tags", "")

    changed = row != existing
    rows[key] = row

    sorted_rows = [rows[k] for k in sorted(rows.keys())]
    write_catalog(ATOMIC_INDEX_PATH, ATOMIC_HEADERS, sorted_rows)
    return changed


def update_pipeline_status(clip_path: Path, **updates: Optional[str]) -> dict:
    ensure_catalog_dirs()
    rows = read_catalog(PIPELINE_STATUS_PATH, "clip_path")
    key = str(clip_path)
    if key not in rows:
        rows[key] = {header: "" for header in PIPELINE_HEADERS}
        rows[key]["clip_path"] = key

    row = rows[key]
    for column, value in updates.items():
        if value is None:
            continue
        if column not in PIPELINE_HEADERS:
            raise CatalogError(f"Unknown pipeline status column: {column}")
        row[column] = value

    rows[key] = row
    sorted_rows = [rows[k] for k in sorted(rows.keys())]
    write_catalog(PIPELINE_STATUS_PATH, PIPELINE_HEADERS, sorted_rows)
    return row


def sidecar_path_for(clip_path: Path) -> Path:
    ensure_catalog_dirs()
    stem = clip_path.stem
    return SIDE_CAR_DIR / f"{stem}.json"


def load_sidecar(clip_path: Path) -> dict:
    path = sidecar_path_for(clip_path)
    if not path.exists():
        return {
            "clip_path": str(clip_path),
            "source_sha1_64": "",
            "meta": {},
            "steps": {},
            "errors": [],
        }
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_sidecar(clip_path: Path, data: dict) -> None:
    path = sidecar_path_for(clip_path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _update_sidecar_base(clip_path: Path, metadata: dict) -> None:
    data = load_sidecar(clip_path)
    data["clip_path"] = str(clip_path)
    if metadata.get("sha1_64"):
        data["source_sha1_64"] = metadata["sha1_64"]
    meta_block = data.setdefault("meta", {})
    for key in ("duration_s", "fps", "size"):
        if key in metadata:
            meta_block[key] = metadata[key]
    save_sidecar(clip_path, data)


def _append_error(data: dict, step: str, message: str, at: str) -> None:
    errors = data.setdefault("errors", [])
    errors.append({"step": step, "message": message, "at": at})


def scan_atomic_clips() -> tuple[int, int]:
    ensure_catalog_dirs()
    clips = sorted(ATOMIC_DIR.glob("*.mp4"))
    updated = 0
    for clip in clips:
        stat = clip.stat()
        created_at = _dt.datetime.fromtimestamp(
            getattr(stat, "st_ctime", stat.st_mtime), tz=_dt.timezone.utc
        ).replace(microsecond=0)
        meta = probe_clip(clip)
        meta["created_at"] = created_at.isoformat()
        meta["sha1_64"] = sha1_64(clip)
        try:
            width = int(meta.get("width") or 0)
        except (TypeError, ValueError):
            width = 0
        try:
            height = int(meta.get("height") or 0)
        except (TypeError, ValueError):
            height = 0
        meta["size"] = [width, height]
        changed = update_atomic_index(clip, meta)
        _update_sidecar_base(clip, meta)
        if changed:
            updated += 1
    return len(clips), updated


def mark_upscaled(
    clip_path: Path,
    out_path: Optional[Path],
    *,
    at: Optional[str],
    scale: Optional[int],
    model: Optional[str],
    error: Optional[str],
) -> None:
    timestamp = at or iso_now()
    clip_path = clip_path.resolve()
    out_str = str(out_path.resolve()) if out_path else ""

    if error:
        last_error = error
        updates = {
            "last_error": last_error,
            "last_run_at": timestamp,
        }
    else:
        updates = {
            "upscale_done_at": timestamp,
            "upscaled_path": out_str,
            "last_error": "",
            "last_run_at": timestamp,
        }

    update_pipeline_status(clip_path, **updates)

    data = load_sidecar(clip_path)
    steps = data.setdefault("steps", {})
    step_info = {
        "done": error is None,
        "out": out_str,
    }
    if not error:
        step_info["at"] = timestamp
    if scale is not None:
        step_info["scale"] = scale
    if model:
        step_info["model"] = model
    if error:
        step_info["error"] = error
        _append_error(data, "upscale", error, timestamp)
    steps["upscale"] = {k: v for k, v in step_info.items() if v not in (None, "")}
    save_sidecar(clip_path, data)


def mark_branded(
    clip_path: Path,
    out_path: Optional[Path],
    *,
    at: Optional[str],
    brand: Optional[str],
    args: Optional[list[str]],
    error: Optional[str],
) -> None:
    timestamp = at or iso_now()
    clip_path = clip_path.resolve()
    out_str = str(out_path.resolve()) if out_path else ""

    if error:
        updates = {
            "last_error": error,
            "last_run_at": timestamp,
        }
    else:
        updates = {
            "follow_brand_done_at": timestamp,
            "branded_path": out_str,
            "last_error": "",
            "last_run_at": timestamp,
        }

    update_pipeline_status(clip_path, **updates)

    data = load_sidecar(clip_path)
    steps = data.setdefault("steps", {})
    step_info = {
        "done": error is None,
        "out": out_str,
    }
    if not error:
        step_info["at"] = timestamp
    if brand:
        step_info["brand"] = brand
    if args:
        step_info["args"] = args
    if error:
        step_info["error"] = error
        _append_error(data, "follow_crop_brand", error, timestamp)
    steps["follow_crop_brand"] = {
        k: v for k, v in step_info.items() if v not in (None, "", [])
    }
    save_sidecar(clip_path, data)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan-atomic", action="store_true", help="Scan atomic clips")
    group.add_argument("--mark-upscaled", action="store_true", help="Mark upscale step")
    group.add_argument("--mark-branded", action="store_true", help="Mark branding step")

    parser.add_argument("--clip", type=Path, help="Clip path for mark commands")
    parser.add_argument("--out", type=Path, help="Output path for mark commands")
    parser.add_argument("--scale", type=int, help="Upscale scale factor")
    parser.add_argument("--model", help="Upscale model name")
    parser.add_argument("--brand", help="Brand identifier")
    parser.add_argument("--args", nargs="*", help="Invocation arguments for branding step")
    parser.add_argument("--at", help="Timestamp override (ISO8601)")
    parser.add_argument("--error", help="Error message to record")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.scan_atomic:
        total, updated = scan_atomic_clips()
        print(f"Scanned {total} clip(s); {updated} new/updated row(s).")
        return 0

    if args.mark_upscaled:
        if not args.clip:
            raise SystemExit("--clip is required for --mark-upscaled")
        clip_path = args.clip
        out_path = args.out
        mark_upscaled(
            clip_path,
            out_path if out_path else None,
            at=args.at,
            scale=args.scale,
            model=args.model,
            error=args.error,
        )
        print(f"Updated upscale status for {clip_path}.")
        return 0

    if args.mark_branded:
        if not args.clip:
            raise SystemExit("--clip is required for --mark-branded")
        clip_path = args.clip
        out_path = args.out
        mark_branded(
            clip_path,
            out_path if out_path else None,
            at=args.at,
            brand=args.brand,
            args=args.args,
            error=args.error,
        )
        print(f"Updated branding status for {clip_path}.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

