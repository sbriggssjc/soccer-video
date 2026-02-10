"""Catalog and pipeline tracking utilities for the soccer-video pipeline.

This module scans curated atomic clips, records metadata in concise CSV
catalogs, maintains per-clip JSON sidecars that store provenance for each
processing step, detects duplicates, and provides cleanup helpers. It is
designed for Windows/PowerShell workflows but works cross-platform.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"
ATOMIC_DIR = OUT_DIR / "atomic_clips"
GAMES_DIR = OUT_DIR / "games"
CATALOG_DIR = OUT_DIR / "catalog"
SIDE_CAR_DIR = CATALOG_DIR / "sidecar"
TRASH_ROOT = OUT_DIR / "_trash" / "atomic_dupes"

ATOMIC_INDEX_PATH = CATALOG_DIR / "atomic_index.csv"
PIPELINE_STATUS_PATH = CATALOG_DIR / "pipeline_status.csv"
DUPLICATES_PATH = CATALOG_DIR / "duplicates.csv"
MASTERS_INDEX_PATH = CATALOG_DIR / "masters_index.csv"
CLEANUP_LOG_PATH = CATALOG_DIR / "cleanup_log.txt"


VIDEO_EXTENSIONS = {".mp4", ".mov"}

ATOMIC_HEADERS = [
    "clip_id",
    "clip_name",
    "clip_path",
    "clip_rel",
    "clip_stem",
    "created_at_utc",
    "duration_s",
    "width",
    "height",
    "fps",
    "sha1_64",
    "tags",
    "t_start_s",
    "t_end_s",
    "master_path",
    "master_rel",
]

PIPELINE_HEADERS = [
    "clip_path",
    "upscaled_path",
    "upscale_done_at",
    "branded_path",
    "follow_brand_done_at",
    "last_error",
    "last_run_at",
]

DUPLICATES_HEADERS = [
    "dup_group_id",
    "canonical_clip_rel",
    "duplicate_clip_rel",
    "reason",
    "overlap_ratio",
]

MASTERS_HEADERS = [
    "master_path",
    "master_rel",
    "duration_s",
    "width",
    "height",
    "fps",
    "n_clips_linked",
]


TIMESTAMP_RE = re.compile(
    r"__t(-?\d+(?:\.\d+)?)-t?(-?\d+(?:\.\d+)?)(?:\.[_A-Za-z0-9]+)*$",
    re.IGNORECASE,
)


class CatalogError(RuntimeError):
    """Raised when catalog operations fail."""


@dataclass
class ClipRecord:
    clip_path: Path
    clip_rel: str
    clip_name: str
    clip_stem: str
    created_at_utc: str
    created_ts: float
    duration_s: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: str
    sha1_64: str
    tags: str = ""
    t_start_s: Optional[float] = None
    t_end_s: Optional[float] = None
    master_path: str = ""
    master_rel: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def clip_id(self) -> str:
        sha = self.sha1_64 or ""
        return f"{sha}:{self.clip_stem}"

    def to_row(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "clip_name": self.clip_name,
            "clip_path": str(self.clip_path),
            "clip_rel": self.clip_rel,
            "clip_stem": self.clip_stem,
            "created_at_utc": self.created_at_utc,
            "duration_s": format_float(self.duration_s),
            "width": "" if self.width is None else str(self.width),
            "height": "" if self.height is None else str(self.height),
            "fps": self.fps,
            "sha1_64": self.sha1_64,
            "tags": self.tags,
            "t_start_s": format_float(self.t_start_s),
            "t_end_s": format_float(self.t_end_s),
            "master_path": self.master_path,
            "master_rel": self.master_rel,
        }


@dataclass
class MasterRecord:
    path: Path
    rel: str
    duration_s: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: str
    n_clips_linked: int = 0

    def to_row(self) -> dict:
        return {
            "master_path": self.path.resolve().as_posix(),
            "master_rel": self.rel,
            "duration_s": format_float(self.duration_s),
            "width": "" if self.width is None else str(self.width),
            "height": "" if self.height is None else str(self.height),
            "fps": self.fps,
            "n_clips_linked": str(self.n_clips_linked),
        }


@dataclass
class DuplicateRecord:
    group_id: str
    canonical_rel: str
    duplicate_rel: str
    reason: str
    overlap_ratio: Optional[float]

    def to_row(self) -> dict:
        return {
            "dup_group_id": self.group_id,
            "canonical_clip_rel": self.canonical_rel,
            "duplicate_clip_rel": self.duplicate_rel,
            "reason": self.reason,
            "overlap_ratio": format_float(self.overlap_ratio),
        }


@dataclass
class BuildResult:
    scanned: int
    indexed: int
    changed: int
    masters_found: int
    hard_dupes: int
    soft_dupes: int
    probe_failures: int


def ensure_catalog_dirs() -> None:
    """Ensure catalog directories exist."""

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    SIDE_CAR_DIR.mkdir(parents=True, exist_ok=True)


def to_repo_relative(path: Path) -> str:
    # Try without resolving first — preserves paths through symlinks/junctions
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def from_repo_relative(rel: str) -> Path:
    normalized = rel.replace("\\", "/")
    return (ROOT / Path(normalized)).resolve()


def sha1_64(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute a 64-bit (16 hex characters) SHA1 digest for *path*."""

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
        "-show_entries",
        "format=duration",
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


def _probe_video_cv2(path: Path) -> dict:
    """Fallback video probe using OpenCV when ffprobe is unavailable."""
    try:
        import cv2
    except ImportError:
        raise CatalogError("Neither ffprobe nor OpenCV available for probing")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"duration_s": None, "width": None, "height": None, "fps": "", "stream_exists": False}
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = f"{fps_val:.0f}/1" if fps_val and fps_val > 0 else ""
        duration = round(frame_count / fps_val, 3) if fps_val and fps_val > 0 and frame_count > 0 else None
        return {
            "duration_s": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "stream_exists": width is not None and height is not None,
        }
    finally:
        cap.release()


def probe_video(path: Path) -> dict:
    """Return basic metadata for *path* using ffprobe, falling back to OpenCV."""

    try:
        info = _run_ffprobe(path)
    except CatalogError:
        return _probe_video_cv2(path)
    streams = info.get("streams") or []
    stream = streams[0] if streams else {}

    duration = stream.get("duration")
    if duration is None:
        duration = info.get("format", {}).get("duration")

    try:
        duration_value = round(float(duration), 3) if duration is not None else None
    except (TypeError, ValueError):
        duration_value = None

    width = stream.get("width")
    height = stream.get("height")
    fps = stream.get("r_frame_rate", "")

    try:
        width_value = int(width) if width is not None else None
    except (TypeError, ValueError):
        width_value = None
    try:
        height_value = int(height) if height is not None else None
    except (TypeError, ValueError):
        height_value = None

    return {
        "duration_s": duration_value,
        "width": width_value,
        "height": height_value,
        "fps": fps or "",
        "stream_exists": bool(streams),
    }


def iso_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def read_catalog(path: Path, key: str) -> Dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: Dict[str, dict] = {}
        for row in reader:
            if key not in row:
                continue
            rows[row[key]] = row
        return rows


def write_catalog(path: Path, headers: Sequence[str], rows: Iterable[dict]) -> None:
    ensure_catalog_dirs()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in headers})


def ensure_pipeline_status_columns() -> None:
    rows = read_catalog(PIPELINE_STATUS_PATH, "clip_path")
    if not rows:
        return
    updated = False
    ordered_rows = []
    for key in sorted(rows.keys()):
        row = rows[key]
        for header in PIPELINE_HEADERS:
            if header not in row:
                row[header] = ""
                updated = True
        ordered_rows.append(row)
    if updated:
        write_catalog(PIPELINE_STATUS_PATH, PIPELINE_HEADERS, ordered_rows)


def load_sidecar(clip_path: Path) -> dict:
    ensure_catalog_dirs()
    path = SIDE_CAR_DIR / f"{clip_path.stem}.json"
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
    ensure_catalog_dirs()
    path = SIDE_CAR_DIR / f"{clip_path.stem}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _append_error(data: dict, step: str, message: str, at: Optional[str] = None) -> None:
    timestamp = at or iso_now()
    errors = data.setdefault("errors", [])
    errors.append({"step": step, "message": message, "at": timestamp})


def format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.3f}" if not math.isnan(value) else ""


def parse_timestamps(stem: str) -> tuple[Optional[float], Optional[float]]:
    """Parse start/end timestamps from a clip filename stem.

    Detects and corrects non-standard encodings where timestamps are stored
    in units other than seconds (60x from older pipeline, 1440x from legacy).
    """
    match = TIMESTAMP_RE.search(stem)
    if not match:
        return None, None
    try:
        start = float(match.group(1))
        end = float(match.group(2))
    except ValueError:
        return None, None
    return _normalize_timestamps(start, end)


# Maximum plausible timestamp in seconds for a game (~3 hours).
_MAX_PLAUSIBLE_TS = 10800.0


def _normalize_timestamps(
    start: float, end: float,
) -> tuple[Optional[float], Optional[float]]:
    """Correct timestamps encoded in non-standard units.

    Some clips have filename timestamps stored as:
      - 60x seconds (from older pipeline)  e.g. t18900 = 315s
      - 1440x seconds (legacy encoding)    e.g. t2937600 = 2040s
    Detects these by checking if values exceed a plausible game duration
    and divides by the appropriate factor.
    """
    if start < 0 and end < 0:
        return start, end  # negative offsets are intentional

    if max(abs(start), abs(end)) <= _MAX_PLAUSIBLE_TS:
        return start, end  # already in seconds

    # Try /60 first (most common pipeline encoding)
    s60, e60 = start / 60, end / 60
    if max(abs(s60), abs(e60)) <= _MAX_PLAUSIBLE_TS and e60 > s60:
        return s60, e60

    # Try /1440 for very large values (legacy encoding)
    s1440, e1440 = start / 1440, end / 1440
    if max(abs(s1440), abs(e1440)) <= _MAX_PLAUSIBLE_TS and e1440 > s1440:
        return s1440, e1440

    # Cannot normalize — return raw values
    return start, end


def _created_at(path: Path) -> tuple[str, float]:
    stat = path.stat()
    created_ts = getattr(stat, "st_ctime", stat.st_mtime)
    created_dt = _dt.datetime.fromtimestamp(created_ts, tz=_dt.timezone.utc).replace(
        microsecond=0
    )
    return created_dt.isoformat(), created_dt.timestamp()


def _normalize_tokenize(name: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return [token for token in cleaned.split() if token]


def probe_master_cached(path: Path, cache: Dict[Path, MasterRecord]) -> MasterRecord:
    if path in cache:
        return cache[path]
    meta = probe_video(path)
    rel = to_repo_relative(path) if path.is_absolute() else path.as_posix()
    record = MasterRecord(
        path=path,
        rel=rel,
        duration_s=meta.get("duration_s"),
        width=meta.get("width"),
        height=meta.get("height"),
        fps=meta.get("fps", ""),
    )
    cache[path] = record
    return record


def find_master_for_clip(
    clip_path: Path, master_cache: Dict[Path, MasterRecord]
) -> Optional[MasterRecord]:
    """Resolve the most likely master video for *clip_path*."""

    clip_path = clip_path.resolve()
    for parent in [clip_path.parent, *clip_path.parents]:
        if parent == clip_path.drive or parent == parent.parent:
            break
        try:
            parent.relative_to(ROOT)
        except ValueError:
            continue
        candidates: list[Path] = []
        for entry in parent.iterdir():
            if not entry.is_file() or entry.suffix.lower() != ".mp4":
                continue
            name_lower = entry.name.lower()
            if name_lower == "master.mp4" or name_lower.startswith("full_game"):
                candidates.append(entry)
        if candidates:
            if len(candidates) > 1:
                ranked = sorted(
                    candidates,
                    key=lambda p: (
                        0 if p.name.lower() == "master.mp4" else 1,
                        -probe_master_cached(p, master_cache).duration_s
                        if probe_master_cached(p, master_cache).duration_s is not None
                        else float("-inf"),
                    ),
                )
                chosen = ranked[0]
            else:
                chosen = candidates[0]
            return probe_master_cached(chosen, master_cache)
        if parent == ROOT:
            break

    try:
        clip_rel = clip_path.relative_to(ROOT)
        parts = clip_rel.parts
        clip_group = parts[2] if len(parts) >= 3 else clip_path.parent.name
    except ValueError:
        # Clip is outside the repo (e.g. OneDrive symlink) — use parent name
        clip_group = clip_path.parent.name
    target_tokens = set(_normalize_tokenize(clip_group))

    if not GAMES_DIR.exists():
        return None

    best_score = -1
    best_candidate: Optional[Path] = None
    for candidate in GAMES_DIR.rglob("*.mp4"):
        # Only consider actual master files, not raw footage or rendered reels
        cname = candidate.name.lower()
        if cname != "master.mp4" and not cname.startswith("full_game"):
            continue
        try:
            candidate_rel = candidate.relative_to(ROOT)
        except ValueError:
            continue
        cand_tokens = set(_normalize_tokenize(candidate.parent.name))
        score = len(target_tokens & cand_tokens)
        if candidate.parent.name.lower() == clip_path.parent.name.lower():
            score += 2
        if not cand_tokens and target_tokens:
            score -= 1
        duration = probe_master_cached(candidate, master_cache).duration_s or 0.0
        if score > best_score or (score == best_score and duration > (probe_master_cached(best_candidate, master_cache).duration_s if best_candidate else 0.0)):
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        return None
    return probe_master_cached(best_candidate, master_cache)


def is_canonical_rel(rel: str) -> bool:
    path = Path(rel)
    if len(path.parts) < 3:
        return False
    if path.parts[0:2] != ("out", "atomic_clips"):
        return False
    folder = path.parts[2]
    return bool(re.match(r"\d{4}-\d{2}-\d{2}__", folder))


def choose_canonical(records: Sequence[ClipRecord]) -> ClipRecord:
    def rank(rec: ClipRecord) -> tuple:
        return (
            0 if is_canonical_rel(rec.clip_rel) else 1,
            len(Path(rec.clip_rel).parts),
            rec.clip_rel.lower(),
            -(rec.created_ts or 0.0),
        )

    return sorted(records, key=rank)[0]


def compute_overlap_ratio(
    start_a: Optional[float],
    end_a: Optional[float],
    start_b: Optional[float],
    end_b: Optional[float],
) -> Optional[float]:
    if None in (start_a, end_a, start_b, end_b):
        return None
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 1.0 if overlap > 0 else None
    return overlap / union if union else None


def load_existing_atomic_rows() -> Dict[str, dict]:
    if not ATOMIC_INDEX_PATH.exists():
        return {}
    with ATOMIC_INDEX_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: Dict[str, dict] = {}
        for row in reader:
            clip_rel = row.get("clip_rel")
            if not clip_rel and row.get("clip_path"):
                try:
                    clip_rel = str(Path(row["clip_path"]).resolve().relative_to(ROOT))
                except Exception:
                    clip_rel = row["clip_path"]
            if clip_rel:
                rows[clip_rel.replace("\\", "/")] = row
        return rows


def gather_clip_record(
    clip_path: Path,
    existing_row: Optional[dict],
    master_cache: Dict[Path, MasterRecord],
) -> ClipRecord:
    clip_path = clip_path.resolve()
    clip_rel = to_repo_relative(clip_path)
    clip_name = clip_path.name
    clip_stem = clip_path.stem
    created_at_iso, created_ts = _created_at(clip_path)
    tags = (existing_row or {}).get("tags", "")
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps = ""
    errors: list[str] = []

    try:
        probe = probe_video(clip_path)
        duration_s = probe.get("duration_s")
        width = probe.get("width")
        height = probe.get("height")
        fps = probe.get("fps", "")
        if not probe.get("stream_exists"):
            errors.append("No video stream detected")
    except CatalogError as exc:
        errors.append(str(exc))
        duration_s = None
        width = None
        height = None
        fps = ""

    try:
        digest = sha1_64(clip_path)
    except OSError as exc:
        errors.append(f"hash failure: {exc}")
        digest = ""

    t_start, t_end = parse_timestamps(clip_stem)

    # Prefer existing (possibly manually corrected) timestamps when they
    # are valid and the file hash hasn't changed.
    if existing_row:
        ex_start = existing_row.get("t_start_s", "").strip()
        ex_end = existing_row.get("t_end_s", "").strip()
        ex_sha = existing_row.get("sha1_64", "").strip()
        if ex_start and ex_end:
            try:
                es, ee = float(ex_start), float(ex_end)
                if ee > es > 0:
                    # Existing timestamps are valid; keep them if the clip
                    # is the same file (matching hash) or if filename-parsed
                    # timestamps are unreasonable.
                    same_file = ex_sha and ex_sha == digest
                    parsed_bad = (t_start is None or t_end is None
                                  or t_end <= t_start
                                  or max(abs(t_start), abs(t_end)) > _MAX_PLAUSIBLE_TS)
                    if same_file or parsed_bad:
                        t_start, t_end = es, ee
            except ValueError:
                pass

    master_record = None
    try:
        master_record = find_master_for_clip(clip_path, master_cache)
    except CatalogError as exc:
        errors.append(str(exc))

    master_path = master_record.path.resolve().as_posix() if master_record else ""
    master_rel = master_record.rel if master_record else ""
    if master_record:
        master_record.n_clips_linked += 1

    record = ClipRecord(
        clip_path=clip_path,
        clip_rel=clip_rel,
        clip_name=clip_name,
        clip_stem=clip_stem,
        created_at_utc=created_at_iso,
        created_ts=created_ts,
        duration_s=duration_s,
        width=width,
        height=height,
        fps=fps,
        sha1_64=digest,
        tags=tags,
        t_start_s=t_start,
        t_end_s=t_end,
        master_path=master_path,
        master_rel=master_rel,
        errors=errors,
    )
    update_sidecar_from_record(record)
    return record


def update_sidecar_from_record(record: ClipRecord) -> None:
    data = load_sidecar(record.clip_path)
    data["clip_path"] = str(record.clip_path)
    if record.sha1_64:
        data["source_sha1_64"] = record.sha1_64
    data["master_path"] = record.master_path
    data["master_rel"] = record.master_rel
    if record.t_start_s is not None:
        data["t_start_s"] = record.t_start_s
    if record.t_end_s is not None:
        data["t_end_s"] = record.t_end_s
    meta = data.setdefault("meta", {})
    if record.duration_s is not None:
        meta["duration_s"] = record.duration_s
    if record.width is not None:
        meta["width"] = record.width
    if record.height is not None:
        meta["height"] = record.height
    if record.fps:
        meta["fps"] = record.fps
    if record.errors:
        existing = {
            (item.get("step"), item.get("message"))
            for item in data.get("errors", [])
        }
        for err in record.errors:
            signature = ("scan", err)
            if signature not in existing:
                _append_error(data, "scan", err)
                existing.add(signature)
    save_sidecar(record.clip_path, data)


def scan_atomic_clips() -> tuple[list[ClipRecord], Dict[Path, MasterRecord], int]:
    ensure_catalog_dirs()
    master_cache: Dict[Path, MasterRecord] = {}
    existing_rows = load_existing_atomic_rows()
    records: list[ClipRecord] = []
    probe_failures = 0

    clip_paths = sorted(ATOMIC_DIR.rglob("*.mp4")) if ATOMIC_DIR.exists() else []
    for clip_path in clip_paths:
        clip_rel = to_repo_relative(clip_path)
        existing = existing_rows.get(clip_rel)
        record = gather_clip_record(clip_path, existing, master_cache)
        probe_failures += len(record.errors)
        records.append(record)

    return records, master_cache, probe_failures


def compute_duplicate_groups(records: Sequence[ClipRecord]) -> tuple[list[DuplicateRecord], int, int]:
    duplicates: list[DuplicateRecord] = []
    hard_count = 0
    soft_count = 0
    group_index = 1
    recorded_pairs: set[tuple[str, str]] = set()

    hash_map: Dict[str, list[ClipRecord]] = {}
    for record in records:
        if not record.sha1_64:
            continue
        hash_map.setdefault(record.sha1_64, []).append(record)

    for digest, group in hash_map.items():
        if len(group) < 2:
            continue
        canonical = choose_canonical(group)
        group_id = f"D{group_index:04d}"
        group_index += 1
        hard_count += 1
        for dup in group:
            if dup is canonical:
                continue
            pair_key = (canonical.clip_rel, dup.clip_rel)
            recorded_pairs.add(pair_key)
            duplicates.append(
                DuplicateRecord(
                    group_id=group_id,
                    canonical_rel=canonical.clip_rel,
                    duplicate_rel=dup.clip_rel,
                    reason="hard",
                    overlap_ratio=1.0,
                )
            )

    masters_map: Dict[str, list[ClipRecord]] = {}
    for record in records:
        if record.master_rel:
            masters_map.setdefault(record.master_rel, []).append(record)

    for master_rel, group in masters_map.items():
        if len(group) < 2:
            continue
        indices = list(range(len(group)))
        adjacency: Dict[int, set[int]] = {i: set() for i in indices}
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]
                ratio = compute_overlap_ratio(
                    a.t_start_s, a.t_end_s, b.t_start_s, b.t_end_s
                )
                start_close = (
                    a.t_start_s is not None
                    and b.t_start_s is not None
                    and abs(a.t_start_s - b.t_start_s) <= 0.25
                )
                end_close = (
                    a.t_end_s is not None
                    and b.t_end_s is not None
                    and abs(a.t_end_s - b.t_end_s) <= 0.25
                )
                if (ratio is not None and ratio >= 0.9) or (start_close and end_close):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        visited: set[int] = set()
        for idx in indices:
            if idx in visited:
                continue
            stack = [idx]
            component: list[int] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                stack.extend(adjacency[current])
            if len(component) <= 1:
                continue
            members = [group[i] for i in component]
            canonical = choose_canonical(members)
            if len({m.sha1_64 for m in members if m.sha1_64}) <= 1:
                all_reported = all(
                    (canonical.clip_rel, member.clip_rel) in recorded_pairs
                    for member in members
                    if member is not canonical
                )
                if all_reported:
                    continue
            group_id = f"D{group_index:04d}"
            group_index += 1
            soft_count += 1
            for member in members:
                if member is canonical:
                    continue
                pair_key = (canonical.clip_rel, member.clip_rel)
                if pair_key in recorded_pairs:
                    continue
                recorded_pairs.add(pair_key)
                ratio = compute_overlap_ratio(
                    canonical.t_start_s,
                    canonical.t_end_s,
                    member.t_start_s,
                    member.t_end_s,
                )
                duplicates.append(
                    DuplicateRecord(
                        group_id=group_id,
                        canonical_rel=canonical.clip_rel,
                        duplicate_rel=member.clip_rel,
                        reason="soft",
                        overlap_ratio=ratio,
                    )
                )

    return duplicates, hard_count, soft_count


def write_atomic_index(records: Sequence[ClipRecord]) -> int:
    existing_rows = load_existing_atomic_rows()
    changed = 0
    new_rows = []
    for record in records:
        row = record.to_row()
        existing = existing_rows.get(record.clip_rel)
        if existing:
            if not row.get("tags") and existing.get("tags"):
                row["tags"] = existing["tags"]
            if any(row.get(key, "") != existing.get(key, "") for key in ATOMIC_HEADERS):
                changed += 1
        else:
            changed += 1
        new_rows.append(row)
    new_rows.sort(key=lambda r: r["clip_rel"])
    write_catalog(ATOMIC_INDEX_PATH, ATOMIC_HEADERS, new_rows)
    return changed


def write_masters_index(masters: Dict[Path, MasterRecord]) -> None:
    rows = [record.to_row() for record in masters.values()]
    rows.sort(key=lambda r: r["master_rel"])
    write_catalog(MASTERS_INDEX_PATH, MASTERS_HEADERS, rows)


def write_duplicates_csv(duplicates: Sequence[DuplicateRecord]) -> None:
    rows = [record.to_row() for record in duplicates]
    rows.sort(key=lambda r: (r["dup_group_id"], r["duplicate_clip_rel"]))
    write_catalog(DUPLICATES_PATH, DUPLICATES_HEADERS, rows)


def rebuild_atomic_index() -> BuildResult:
    ensure_catalog_dirs()
    ensure_pipeline_status_columns()

    records, masters_cache, probe_failures = scan_atomic_clips()
    changed = write_atomic_index(records)
    write_masters_index(masters_cache)
    duplicates, hard_count, soft_count = compute_duplicate_groups(records)
    write_duplicates_csv(duplicates)

    return BuildResult(
        scanned=len(records),
        indexed=len(records),
        changed=changed,
        masters_found=len(masters_cache),
        hard_dupes=hard_count,
        soft_dupes=soft_count,
        probe_failures=probe_failures,
    )


def load_pipeline_status_table() -> Dict[str, dict]:
    rows = read_catalog(PIPELINE_STATUS_PATH, "clip_path")
    table: Dict[str, dict] = {}
    for key, row in rows.items():
        for header in PIPELINE_HEADERS:
            row.setdefault(header, "")
        table[key] = row
    return table


def save_pipeline_status_table(table: Dict[str, dict]) -> None:
    rows = [table[key] for key in sorted(table.keys())]
    write_catalog(PIPELINE_STATUS_PATH, PIPELINE_HEADERS, rows)


def update_pipeline_status(clip_path: Path, **updates: Optional[str]) -> dict:
    ensure_catalog_dirs()
    table = load_pipeline_status_table()
    key = str(clip_path.resolve())
    row = table.get(key)
    if row is None:
        row = {header: "" for header in PIPELINE_HEADERS}
        row["clip_path"] = key
    for column, value in updates.items():
        if value is None:
            continue
        if column not in PIPELINE_HEADERS:
            raise CatalogError(f"Unknown pipeline status column: {column}")
        row[column] = value
    table[key] = row
    save_pipeline_status_table(table)
    return row


def update_sidecar_step(
    clip_path: Path,
    step_key: str,
    step_info: dict,
    error: Optional[str] = None,
) -> None:
    data = load_sidecar(clip_path)
    steps = data.setdefault("steps", {})
    info = {k: v for k, v in step_info.items() if v not in (None, "", [], {})}
    steps[step_key] = info
    if error:
        _append_error(data, step_key, error)
    save_sidecar(clip_path, data)


def mark_upscaled(
    clip_path: Path,
    out_path: Optional[Path],
    *,
    at: Optional[str] = None,
    scale: Optional[int] = None,
    model: Optional[str] = None,
    error: Optional[str] = None,
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
            "upscaled_path": out_str,
            "upscale_done_at": timestamp,
            "last_error": "",
            "last_run_at": timestamp,
        }

    update_pipeline_status(clip_path, **updates)

    step_info = {
        "done": error is None,
        "out": out_str,
        "at": timestamp if not error else None,
        "scale": scale,
        "model": model,
        "error": error,
    }
    update_sidecar_step(clip_path, "upscale", step_info, error=error)


def mark_branded(
    clip_path: Path,
    out_path: Optional[Path],
    *,
    at: Optional[str] = None,
    brand: Optional[str] = None,
    args: Optional[Sequence[str]] = None,
    error: Optional[str] = None,
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
            "branded_path": out_str,
            "follow_brand_done_at": timestamp,
            "last_error": "",
            "last_run_at": timestamp,
        }

    update_pipeline_status(clip_path, **updates)

    step_info = {
        "done": error is None,
        "out": out_str,
        "at": timestamp if not error else None,
        "brand": brand,
        "args": list(args) if args else None,
        "error": error,
    }
    update_sidecar_step(clip_path, "follow_crop_brand", step_info, error=error)


def load_duplicates() -> list[DuplicateRecord]:
    if not DUPLICATES_PATH.exists():
        return []
    with DUPLICATES_PATH.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        records: list[DuplicateRecord] = []
        for row in reader:
            records.append(
                DuplicateRecord(
                    group_id=row.get("dup_group_id", ""),
                    canonical_rel=row.get("canonical_clip_rel", ""),
                    duplicate_rel=row.get("duplicate_clip_rel", ""),
                    reason=row.get("reason", ""),
                    overlap_ratio=float(row["overlap_ratio"]) if row.get("overlap_ratio") else None,
                )
            )
        return records


def normalize_tree(*, dry_run: bool, force: bool, purge: bool) -> dict:
    ensure_catalog_dirs()
    if not DUPLICATES_PATH.exists():
        raise CatalogError("duplicates.csv not found; run --write-duplicates or rebuild first")
    if not dry_run and not force:
        raise CatalogError("Use --force to perform filesystem changes or --dry-run to preview")

    duplicates = load_duplicates()
    if not duplicates:
        return {
            "moved": 0,
            "upscaled_redirected": 0,
            "branded_redirected": 0,
            "removed_dirs": 0,
            "trash_purged": False,
        }

    table = load_pipeline_status_table()
    moved = 0
    upscaled_redirected = 0
    branded_redirected = 0
    removed_dirs = 0
    log_entries: list[str] = []
    today = _dt.datetime.now().strftime("%Y-%m-%d")
    trash_day_root = TRASH_ROOT / today

    def plan_trash_path(rel: str) -> Path:
        p = Path(rel.replace("\\", "/"))
        # Use only the last 2 components (game_folder/clip.mp4) to keep
        # trash paths short and avoid Windows MAX_PATH (260 char) limit.
        parts = p.parts
        if len(parts) >= 2:
            p = Path(parts[-2]) / parts[-1]
        else:
            p = Path(p.name)
        return trash_day_root / p

    for record in duplicates:
        if not record.canonical_rel or not record.duplicate_rel:
            continue
        canonical = from_repo_relative(record.canonical_rel)
        duplicate = from_repo_relative(record.duplicate_rel)
        if canonical == duplicate or not duplicate.exists():
            continue
        duplicate_key = str(duplicate.resolve())
        canonical_key = str(canonical.resolve())
        if duplicate_key == canonical_key:
            continue
        dup_stem = Path(record.duplicate_rel.replace("\\", "/")).stem
        canon_stem = Path(record.canonical_rel.replace("\\", "/")).stem
        trash_target = plan_trash_path(record.duplicate_rel)
        if trash_target.resolve() == duplicate.resolve():
            continue
        dup_row = table.get(duplicate_key)
        canon_row = table.get(canonical_key)

        move_msg = f"{'[DRY-RUN] ' if dry_run else ''}Move {duplicate} -> {trash_target} ({record.reason})"
        log_entries.append(move_msg)
        print(move_msg)

        if dry_run:
            if dup_row and dup_row.get("upscaled_path"):
                preview = Path(dup_row["upscaled_path"])
                if dup_stem and canon_stem and dup_stem in preview.name:
                    planned = preview.with_name(preview.name.replace(dup_stem, canon_stem, 1))
                    msg = f"[DRY-RUN] Redirect upscale {preview} -> {planned}"
                    log_entries.append(msg)
                    print(msg)
            if dup_row and dup_row.get("branded_path"):
                preview = Path(dup_row["branded_path"])
                if dup_stem and canon_stem and dup_stem in preview.name:
                    planned = preview.with_name(preview.name.replace(dup_stem, canon_stem, 1))
                    msg = f"[DRY-RUN] Redirect branded {preview} -> {planned}"
                    log_entries.append(msg)
                    print(msg)
            continue

        try:
            trash_target.parent.mkdir(parents=True, exist_ok=True)
            if trash_target.exists() and not force:
                raise CatalogError(f"Refusing to overwrite {trash_target}")
            if trash_target.exists() and force:
                trash_target.unlink()
            shutil.move(str(duplicate), str(trash_target))
        except (OSError, CatalogError) as exc:
            err_msg = f"FAILED to move {duplicate}: {exc}"
            log_entries.append(err_msg)
            print(err_msg)
            continue
        moved += 1
        move_done_msg = f"Moved {duplicate} -> {trash_target} [{record.reason}]"
        log_entries.append(move_done_msg)
        print(move_done_msg)
        if record.reason == "hard" and canonical.exists():
            canonical_hash = sha1_64(canonical)
            duplicate_hash = sha1_64(trash_target)
            if canonical_hash != duplicate_hash:
                raise CatalogError(f"Hash mismatch after moving duplicate {duplicate}")
        parent = duplicate.parent
        while parent != ATOMIC_DIR and parent.exists():
            try:
                next(parent.iterdir())
            except StopIteration:
                try:
                    parent.rmdir()
                except OSError:
                    break  # OneDrive / locked folders — skip cleanup
                removed_dirs += 1
                parent = parent.parent
                continue
            break

        if dup_row:
            pipeline_changed = False
            if dup_row.get("upscaled_path"):
                dup_upscaled = Path(dup_row["upscaled_path"])
                if dup_upscaled.exists() and dup_stem and canon_stem:
                    if dup_stem in dup_upscaled.name:
                        new_upscaled = dup_upscaled.with_name(
                            dup_upscaled.name.replace(dup_stem, canon_stem, 1)
                        )
                    else:
                        new_upscaled = dup_upscaled
                    if new_upscaled != dup_upscaled:
                        new_upscaled.parent.mkdir(parents=True, exist_ok=True)
                        if new_upscaled.exists() and not force:
                            raise CatalogError(
                                f"Refusing to overwrite existing {new_upscaled}"
                            )
                        if new_upscaled.exists() and force:
                            new_upscaled.unlink()
                        shutil.move(str(dup_upscaled), str(new_upscaled))
                        upscaled_redirected += 1
                        msg = f"Redirected upscale {dup_upscaled} -> {new_upscaled}"
                        log_entries.append(msg)
                        print(msg)
                    if canon_row is None:
                        canon_row = {header: "" for header in PIPELINE_HEADERS}
                        canon_row["clip_path"] = canonical_key
                    if not canon_row.get("upscaled_path"):
                        canon_row["upscaled_path"] = str(new_upscaled)
                        canon_row["upscale_done_at"] = dup_row.get("upscale_done_at", "")
                    dup_row["upscaled_path"] = ""
                    dup_row["upscale_done_at"] = ""
                    pipeline_changed = True
                    update_sidecar_redirect(
                        duplicate,
                        canonical,
                        step="upscale",
                        out_path=str(new_upscaled),
                    )
            if dup_row.get("branded_path"):
                dup_branded = Path(dup_row["branded_path"])
                if dup_branded.exists() and dup_stem and canon_stem:
                    if dup_stem in dup_branded.name:
                        new_branded = dup_branded.with_name(
                            dup_branded.name.replace(dup_stem, canon_stem, 1)
                        )
                    else:
                        new_branded = dup_branded
                    if new_branded != dup_branded:
                        new_branded.parent.mkdir(parents=True, exist_ok=True)
                        if new_branded.exists() and not force:
                            raise CatalogError(
                                f"Refusing to overwrite existing {new_branded}"
                            )
                        if new_branded.exists() and force:
                            new_branded.unlink()
                        shutil.move(str(dup_branded), str(new_branded))
                        branded_redirected += 1
                        msg = f"Redirected branded {dup_branded} -> {new_branded}"
                        log_entries.append(msg)
                        print(msg)
                    if canon_row is None:
                        canon_row = {header: "" for header in PIPELINE_HEADERS}
                        canon_row["clip_path"] = canonical_key
                    if not canon_row.get("branded_path"):
                        canon_row["branded_path"] = str(new_branded)
                        canon_row["follow_brand_done_at"] = dup_row.get(
                            "follow_brand_done_at", ""
                        )
                    dup_row["branded_path"] = ""
                    dup_row["follow_brand_done_at"] = ""
                    pipeline_changed = True
                    update_sidecar_redirect(
                        duplicate,
                        canonical,
                        step="follow_crop_brand",
                        out_path=str(new_branded),
                    )
            if pipeline_changed and canon_row:
                table[canonical_key] = canon_row
            if pipeline_changed:
                table[duplicate_key] = dup_row

    if not dry_run:
        save_pipeline_status_table(table)

    if log_entries:
        with CLEANUP_LOG_PATH.open("a", encoding="utf-8") as fh:
            for entry in log_entries:
                fh.write(f"{iso_now()} {entry}\n")

    trash_purged = False
    if purge and not dry_run and TRASH_ROOT.exists():
        def _onerror(func, path, exc_info):
            """Best-effort removal — skip OneDrive / locked items."""
            pass
        shutil.rmtree(TRASH_ROOT, onerror=_onerror)
        trash_purged = True

    return {
        "moved": moved,
        "upscaled_redirected": upscaled_redirected,
        "branded_redirected": branded_redirected,
        "removed_dirs": removed_dirs,
        "trash_purged": trash_purged,
    }


def update_sidecar_redirect(
    duplicate_path: Path, canonical_path: Path, *, step: str, out_path: str
) -> None:
    dup_data = load_sidecar(duplicate_path)
    steps = dup_data.setdefault("steps", {})
    step_info = steps.setdefault(step, {})
    step_info["redirected_to"] = str(canonical_path)
    step_info["redirected_out"] = out_path
    save_sidecar(duplicate_path, dup_data)

    canon_data = load_sidecar(canonical_path)
    steps = canon_data.setdefault("steps", {})
    step_block = steps.setdefault(step, {})
    step_block.setdefault("out", out_path)
    step_block.setdefault("done", True)
    step_block.setdefault("at", iso_now())
    save_sidecar(canonical_path, canon_data)


def generate_report() -> dict:
    atomic_rows = load_existing_atomic_rows()
    table = load_pipeline_status_table()
    duplicates = load_duplicates()

    total = len(atomic_rows)
    upscaled = 0
    branded = 0
    orphans = 0
    for row in atomic_rows.values():
        clip_path = row.get("clip_path", "")
        status = table.get(clip_path, {})
        if status.get("upscaled_path"):
            upscaled += 1
        if status.get("branded_path"):
            branded += 1
        if not row.get("master_rel"):
            orphans += 1

    error_sidecars = 0
    if SIDE_CAR_DIR.exists():
        for sidecar_path in SIDE_CAR_DIR.glob("*.json"):
            try:
                with sidecar_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data.get("errors"):
                    error_sidecars += 1
            except Exception:
                error_sidecars += 1

    dup_groups = {rec.group_id for rec in duplicates}
    return {
        "total_clips": total,
        "upscaled": upscaled,
        "branded": branded,
        "orphans": orphans,
        "dup_groups": len(dup_groups),
        "sidecars_with_errors": error_sidecars,
    }


def write_duplicates_from_index() -> tuple[int, int]:
    atomic_rows = load_existing_atomic_rows()
    records: list[ClipRecord] = []
    for row in atomic_rows.values():
        rel = row.get("clip_rel")
        if rel:
            clip_path = from_repo_relative(rel)
        elif row.get("clip_path"):
            clip_path = Path(row["clip_path"])
        else:
            continue
        created_at = row.get("created_at_utc") or iso_now()
        try:
            created_dt = _dt.datetime.fromisoformat(created_at)
        except ValueError:
            created_dt = _dt.datetime.now(tz=_dt.timezone.utc)
        records.append(
            ClipRecord(
                clip_path=clip_path,
                clip_rel=row.get("clip_rel", ""),
                clip_name=row.get("clip_name", clip_path.name),
                clip_stem=row.get("clip_stem", clip_path.stem),
                created_at_utc=created_at,
                created_ts=created_dt.timestamp(),
                duration_s=float(row["duration_s"]) if row.get("duration_s") else None,
                width=int(row["width"]) if row.get("width") else None,
                height=int(row["height"]) if row.get("height") else None,
                fps=row.get("fps", ""),
                sha1_64=row.get("sha1_64", ""),
                tags=row.get("tags", ""),
                t_start_s=float(row["t_start_s"]) if row.get("t_start_s") else None,
                t_end_s=float(row["t_end_s"]) if row.get("t_end_s") else None,
                master_path=row.get("master_path", ""),
                master_rel=row.get("master_rel", ""),
            )
        )

    duplicates, hard_count, soft_count = compute_duplicate_groups(records)
    write_duplicates_csv(duplicates)
    return hard_count, soft_count


def audit_clips() -> dict:
    """Audit atomic clips: per-master breakdown, overlaps, and masterless clips."""
    rows = read_catalog(ATOMIC_INDEX_PATH, "clip_rel")
    masters = read_catalog(MASTERS_INDEX_PATH, "master_rel")

    # Group clips by master_rel
    by_master: Dict[str, list] = {}
    no_master: list = []
    for rel, row in rows.items():
        mrel = row.get("master_rel", "")
        if not mrel:
            no_master.append(rel)
        else:
            by_master.setdefault(mrel, []).append(row)

    # Masters with no clips
    empty_masters = [m for m in masters if m not in by_master]

    # Per-master analysis
    master_reports = []
    total_overlaps = 0
    for mrel in sorted(by_master):
        clips = by_master[mrel]
        # Sort by start time
        timed = []
        for c in clips:
            try:
                t0 = float(c.get("t_start_s") or "nan")
                t1 = float(c.get("t_end_s") or "nan")
            except (ValueError, TypeError):
                t0 = t1 = float("nan")
            timed.append((t0, t1, c["clip_rel"]))
        timed.sort()

        # Check for time overlaps (>50% overlap = suspicious)
        overlaps = []
        for i in range(len(timed) - 1):
            s1, e1, r1 = timed[i]
            for j in range(i + 1, len(timed)):
                s2, e2, r2 = timed[j]
                if s2 >= e1:
                    break  # sorted, no more overlaps
                # Compute overlap
                ov_start = max(s1, s2)
                ov_end = min(e1, e2)
                if ov_end > ov_start:
                    shorter = min(e1 - s1, e2 - s2)
                    if shorter > 0 and (ov_end - ov_start) / shorter > 0.5:
                        overlaps.append((r1, r2, round(ov_end - ov_start, 1)))

        total_overlaps += len(overlaps)
        minfo = masters.get(mrel, {})
        dur = minfo.get("duration_s", "")
        master_reports.append({
            "master_rel": mrel,
            "duration_s": dur,
            "clip_count": len(clips),
            "overlaps": overlaps,
        })

    # Print report
    print("=" * 72)
    print("CLIP AUDIT REPORT")
    print("=" * 72)
    print(f"Total clips: {len(rows)} | Masters with clips: {len(by_master)} | "
          f"Masters without clips: {len(empty_masters)}")
    print()

    for mr in master_reports:
        dur_str = ""
        if mr["duration_s"]:
            try:
                secs = float(mr["duration_s"])
                dur_str = f" ({int(secs // 60)}m{int(secs % 60):02d}s)"
            except (ValueError, TypeError):
                pass
        print(f"  {mr['master_rel']}{dur_str}")
        print(f"    Clips: {mr['clip_count']}")
        if mr["overlaps"]:
            print(f"    WARNING: {len(mr['overlaps'])} overlapping clip pair(s):")
            for r1, r2, ov in mr["overlaps"]:
                print(f"      {Path(r1).name}  <->  {Path(r2).name}  ({ov}s overlap)")
        print()

    if empty_masters:
        print("MASTERS WITH NO CLIPS:")
        for m in sorted(empty_masters):
            print(f"  {m}")
        print()

    if no_master:
        print(f"CLIPS WITH NO MASTER ({len(no_master)}):")
        for rel in sorted(no_master):
            print(f"  {rel}")
        print()

    if total_overlaps == 0 and not empty_masters and not no_master:
        print("All clips are unique moments with no overlaps. All masters have clips.")

    return {
        "total_clips": len(rows),
        "masters_with_clips": len(by_master),
        "empty_masters": empty_masters,
        "no_master_clips": no_master,
        "total_overlaps": total_overlaps,
        "master_reports": master_reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rebuild-atomic-index", action="store_true")
    group.add_argument("--write-duplicates", action="store_true")
    group.add_argument("--normalize-tree", action="store_true")
    group.add_argument("--report", action="store_true")
    group.add_argument("--audit-clips", action="store_true")
    group.add_argument("--mark-upscaled", action="store_true")
    group.add_argument("--mark-branded", action="store_true")
    group.add_argument("--scan-atomic", action="store_true")
    group.add_argument("--scan-list", type=Path)

    parser.add_argument("--clip", type=Path, help="Clip path for mark commands")
    parser.add_argument("--out", type=Path, help="Output path for mark commands")
    parser.add_argument("--scale", type=int, help="Upscale scale factor")
    parser.add_argument("--model", help="Upscale model name")
    parser.add_argument("--brand", help="Brand identifier")
    parser.add_argument("--args", nargs="*", help="Invocation arguments for branding step")
    parser.add_argument("--at", help="Timestamp override (ISO8601)")
    parser.add_argument("--error", help="Error message to record")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for normalization")
    parser.add_argument("--force", action="store_true", help="Allow modifications when normalizing")
    parser.add_argument("--purge", action="store_true", help="Delete trash after normalization")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.scan_atomic:
        args.rebuild_atomic_index = True

    if args.rebuild_atomic_index:
        result = rebuild_atomic_index()
        print(
            "Scanned: {scanned} | Indexed: {indexed} | Changed: {changed} | "
            "Masters: {masters} | Hard dupes: {hard} | Soft dupes: {soft}".format(
                scanned=result.scanned,
                indexed=result.indexed,
                changed=result.changed,
                masters=result.masters_found,
                hard=result.hard_dupes,
                soft=result.soft_dupes,
            )
        )
        return 1 if result.probe_failures else 0

    if args.write_duplicates:
        hard_count, soft_count = write_duplicates_from_index()
        print(
            f"Duplicate catalog refreshed. Hard groups: {hard_count} | Soft groups: {soft_count}"
        )
        return 0

    if args.normalize_tree:
        report = normalize_tree(
            dry_run=args.dry_run,
            force=args.force,
            purge=args.purge,
        )
        summary = (
            "Normalization complete. Moved: {moved}, Upscale redirects: {upscaled}, "
            "Brand redirects: {branded}, Directories removed: {dirs}, Trash purged: {purged}".format(
                moved=report["moved"],
                upscaled=report["upscaled_redirected"],
                branded=report["branded_redirected"],
                dirs=report["removed_dirs"],
                purged="yes" if report["trash_purged"] else "no",
            )
        )
        print(summary)
        return 0

    if args.report:
        report = generate_report()
        print(
            "Atomic clips: {total} | Upscaled: {upscaled} | Branded: {branded} | "
            "Dup groups: {dups} | Orphans: {orphans} | Sidecars w/ errors: {errors}".format(
                total=report["total_clips"],
                upscaled=report["upscaled"],
                branded=report["branded"],
                dups=report["dup_groups"],
                orphans=report["orphans"],
                errors=report["sidecars_with_errors"],
            )
        )
        return 0

    if args.audit_clips:
        audit_clips()
        return 0

    if args.mark_upscaled:
        if not args.clip:
            raise SystemExit("--clip is required for --mark-upscaled")
        mark_upscaled(
            args.clip,
            args.out,
            at=args.at,
            scale=args.scale,
            model=args.model,
            error=args.error,
        )
        print(f"Updated upscale status for {args.clip}.")
        return 0

    if args.mark_branded:
        if not args.clip:
            raise SystemExit("--clip is required for --mark-branded")
        mark_branded(
            args.clip,
            args.out,
            at=args.at,
            brand=args.brand,
            args=args.args,
            error=args.error,
        )
        print(f"Updated branding status for {args.clip}.")
        return 0

    if args.scan_list:
        print("--scan-list is deprecated; use --rebuild-atomic-index instead.")
        result = rebuild_atomic_index()
        print(
            "Scanned: {scanned} | Indexed: {indexed} | Changed: {changed}".format(
                scanned=result.scanned,
                indexed=result.indexed,
                changed=result.changed,
            )
        )
        return 1 if result.probe_failures else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

