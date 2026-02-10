"""Utilities for building Top-10 reels anchored by crowd cheers."""
from __future__ import annotations

import csv
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

SHOT_KEYWORDS = {
    "shot",
    "goal",
    "attempt",
    "finish",
    "chance",
    "penalty",
    "header",
    "volley",
}
TACTICAL_KEYWORDS = {
    "tackle",
    "press",
    "pressure",
    "switch",
    "steal",
    "block",
    "intercept",
    "turnover",
    "duel",
}

_NUMERIC_RE = re.compile(r"[^0-9.\-]+")


@dataclass
class ClipCandidate:
    """A highlight window candidate before padding/overlap resolution."""

    start: float
    end: float
    score: float
    priority: int
    source: str
    event: str = "unknown"
    is_opponent: bool = False


@dataclass
class SelectedClip:
    """A highlight window selected for export with padding applied."""

    start: float
    end: float
    score: float
    source: str
    raw_start: float
    raw_end: float
    event: str = "unknown"
    bucket: str | None = None
    is_goal_like: bool = False
    is_opponent: bool = False


def _parse_float(value: object) -> float | None:
    """Parse a float from loosely formatted CSV values.

    Values may contain stray characters such as "s" or "sec". The PowerShell
    prototype that inspired this module stripped everything except digits,
    decimal points and minus signs, so we do the same for compatibility.
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = _NUMERIC_RE.sub("", text)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_event(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text or "unknown"


def _infer_bucket(event: str) -> str:
    name = (event or "").lower()
    if any(keyword in name for keyword in SHOT_KEYWORDS):
        return "shots"
    if any(keyword in name for keyword in TACTICAL_KEYWORDS):
        return "tackles"
    return "build"


def _is_goal_like(event: str, source: str) -> bool:
    if source == "cheer":
        return True
    name = (event or "").lower()
    return any(keyword in name for keyword in ("goal", "shot", "chance", "attempt", "penalty"))


def _value_is_true(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _row_is_opponent(row: Dict[str, object]) -> bool:
    """Best-effort inference of whether a clip favours the opponent."""

    candidate_keys = (
        "team",
        "side",
        "favors",
        "favours",
        "benefits",
        "for",
        "club",
        "label",
        "who",
        "possession",
        "opponent",
        "is_opponent",
        "favour",
    )
    positive_tokens = {"us", "ours", "our", "home", "team", "self", "favourable", "favorable"}
    negative_tokens = {"opp", "opponent", "opposition", "against", "their", "them", "away", "rival"}

    for key in candidate_keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        flag = _value_is_true(value)
        if flag is not None:
            return bool(flag)
        num = _parse_float(value)
        if num is not None:
            if num < 0:
                return True
            if num > 0:
                return False
        text = str(value).strip().lower()
        if not text:
            continue
        if any(tok in text for tok in negative_tokens):
            return True
        if any(tok in text for tok in positive_tokens):
            return False

    return False


def _allocate_bucket_targets(weights: Dict[str, float], remaining: int) -> Dict[str, int]:
    if remaining <= 0:
        return {name: 0 for name in weights}
    base: Dict[str, int] = {}
    used = 0
    for name, weight in weights.items():
        count = int(math.floor(max(0.0, remaining * max(0.0, weight))))
        base[name] = count
        used += count
    leftover = max(0, remaining - used)
    if leftover:
        for name, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True):
            if leftover <= 0:
                break
            base[name] += 1
            leftover -= 1
    return base


def load_cheer_candidates(
    csv_path: Path,
    duration: float,
    max_count: int = 4,
    min_spacing: float = 45.0,
    pre_window: float = 7.5,
    post_window: float = 2.5,
    min_length: float = 0.8,
) -> List[ClipCandidate]:
    """Load forced highlight windows anchored on cheer timestamps."""

    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        # look for a column named "time" (case insensitive). Fallback to first column.
        columns = [name.lower() for name in reader.fieldnames]
        try:
            time_idx = columns.index("time")
            time_key = reader.fieldnames[time_idx]
        except ValueError:
            time_key = reader.fieldnames[0]

        cheer_times: List[float] = []
        for row in reader:
            t = _parse_float(row.get(time_key))
            if t is None:
                continue
            cheer_times.append(t)

    cheer_times.sort()
    picked: List[float] = []
    for ts in cheer_times:
        if len(picked) >= max_count:
            break
        if not picked or abs(ts - picked[-1]) >= min_spacing:
            picked.append(ts)

    forced: List[ClipCandidate] = []
    for ts in picked:
        start = max(0.0, ts - pre_window)
        end = min(duration, ts + post_window)
        if end - start < min_length:
            continue
        forced.append(
            ClipCandidate(
                start=start,
                end=end,
                score=999.0,
                priority=0,
                source="cheer",
                event="cheer",
            )
        )
    return forced


def load_filtered_candidates(csv_path: Path) -> List[ClipCandidate]:
    """Load ranked highlight windows from the motion/action filter."""

    if not csv_path.exists():
        return []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []

        candidates: List[ClipCandidate] = []
        for row in reader:
            start = _parse_float(row.get("start"))
            end = _parse_float(row.get("end"))
            if start is None or end is None:
                continue
            if end <= start:
                continue
            score_value = None
            for key in ("action_score", "score"):
                score_value = _parse_float(row.get(key))
                if score_value is not None:
                    break
            if score_value is None:
                score_value = 0.0
            event = _normalize_event(row.get("event"))
            is_opponent = _row_is_opponent(row)
            candidates.append(
                ClipCandidate(
                    start=start,
                    end=end,
                    score=score_value,
                    priority=1,
                    source="filtered",
                    event=event,
                    is_opponent=is_opponent,
                )
            )
    return candidates


def rank_candidates(candidates: Iterable[ClipCandidate]) -> List[ClipCandidate]:
    """Sort candidates by priority and score (descending)."""

    return sorted(candidates, key=lambda c: (c.priority, -c.score))


def select_top_candidates(
    candidates: Sequence[ClipCandidate],
    duration: float,
    max_count: int,
    pad_pre: float,
    pad_post: float,
    min_length: float,
    overlap_threshold: float,
    *,
    bucket_weights: Dict[str, float] | None = None,
    min_spacing: float = 20.0,
    opponent_cap: int | None = 2,
) -> List[SelectedClip]:
    """Apply padding and overlap suppression to choose the final highlight list."""

    selected: List[SelectedClip] = []
    taken: List[tuple[float, float]] = []
    bucket_weights = bucket_weights or {"shots": 0.4, "build": 0.4, "tackles": 0.2}
    bucket_weights = {k: max(0.0, v) for k, v in bucket_weights.items()}
    forced = [c for c in candidates if c.priority == 0]
    others = [c for c in candidates if c.priority != 0]
    opponent_selected = 0
    bucket_counts: Dict[str, int] = defaultdict(int)
    bucket_targets: Dict[str, int] = {}

    def attempt(candidate: ClipCandidate, bucket: str | None, enforce_bucket: bool) -> bool:
        nonlocal opponent_selected
        if len(selected) >= max_count:
            return False
        padded_start = max(0.0, candidate.start - pad_pre)
        padded_end = min(duration, candidate.end + pad_post)
        if padded_end - padded_start < min_length:
            return False
        for other_start, other_end in taken:
            intersect = max(0.0, min(padded_end, other_end) - max(padded_start, other_start))
            denom = max(padded_end - padded_start, other_end - other_start)
            if denom > 0 and intersect / denom > overlap_threshold:
                return False
        goal_like = _is_goal_like(candidate.event, candidate.source)
        if min_spacing and min_spacing > 0 and not goal_like:
            for clip in selected:
                if clip.is_goal_like:
                    continue
                if candidate.start < clip.raw_end + min_spacing and candidate.end > clip.raw_start - min_spacing:
                    return False
        if opponent_cap is not None and candidate.is_opponent and opponent_selected >= opponent_cap:
            return False
        if enforce_bucket and bucket:
            target = bucket_targets.get(bucket, 0)
            if bucket_counts[bucket] >= target:
                return False

        selected.append(
            SelectedClip(
                start=padded_start,
                end=padded_end,
                score=candidate.score,
                source=candidate.source,
                raw_start=candidate.start,
                raw_end=candidate.end,
                event=candidate.event,
                bucket=bucket,
                is_goal_like=goal_like,
                is_opponent=candidate.is_opponent,
            )
        )
        taken.append((padded_start, padded_end))
        if bucket:
            bucket_counts[bucket] += 1
        if candidate.is_opponent:
            opponent_selected += 1
        return True

    for candidate in forced:
        attempt(candidate, bucket=None, enforce_bucket=False)

    remaining_slots = max(0, max_count - len(selected))
    bucket_targets = _allocate_bucket_targets(bucket_weights, remaining_slots)

    deferred: List[tuple[ClipCandidate, str | None]] = []
    for candidate in others:
        if len(selected) >= max_count:
            break
        bucket = _infer_bucket(candidate.event)
        if bucket not in bucket_targets:
            bucket = max(bucket_targets, key=bucket_targets.get, default=None)
        if bucket and bucket_targets.get(bucket, 0) <= 0:
            deferred.append((candidate, bucket))
            continue
        if bucket and bucket_counts[bucket] >= bucket_targets.get(bucket, 0):
            deferred.append((candidate, bucket))
            continue
        added = attempt(candidate, bucket=bucket, enforce_bucket=True)
        if not added:
            continue

    if len(selected) < max_count:
        for candidate, bucket in deferred:
            if len(selected) >= max_count:
                break
            attempt(candidate, bucket=bucket if bucket in bucket_targets else None, enforce_bucket=False)

    return selected


def write_selection_csv(path: Path, clips: Sequence[SelectedClip]) -> None:
    """Write the final highlight windows to a CSV for inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "score", "source", "raw_start", "raw_end"])
        for clip in clips:
            writer.writerow(
                [
                    f"{clip.start:.3f}",
                    f"{clip.end:.3f}",
                    f"{clip.score:.3f}",
                    clip.source,
                    f"{clip.raw_start:.3f}",
                    f"{clip.raw_end:.3f}",
                ]
            )


def _run_command(cmd: Sequence[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with code %s: %s\nSTDOUT: %s\nSTDERR: %s"
            % (result.returncode, " ".join(cmd), result.stdout.strip(), result.stderr.strip())
        )


def render_clips(
    video_path: Path,
    clips: Sequence[SelectedClip],
    output_dir: Path,
    ffmpeg: str = "ffmpeg",
    video_codec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 20,
    audio_bitrate: str = "160k",
) -> List[Path]:
    """Render individual highlight clips using ffmpeg."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove old clip files but keep directory.
    for old_file in output_dir.glob("*.mp4"):
        if old_file.is_file():
            old_file.unlink()

    exported: List[Path] = []
    for idx, clip in enumerate(clips, start=1):
        start_ts = f"{clip.start:.2f}"
        end_ts = f"{clip.end:.2f}"
        out_path = output_dir / f"clip_{idx:04d}.mp4"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            start_ts,
            "-to",
            end_ts,
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            str(out_path),
        ]
        _run_command(cmd)
        if out_path.exists() and out_path.stat().st_size > 0:
            exported.append(out_path)
    return exported


def write_concat_file(clips: Sequence[Path], concat_path: Path) -> None:
    """Write a concat list file for ffmpeg."""

    concat_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for clip in sorted(clips):
        full = clip.resolve()
        # Use forward slashes for ffmpeg compatibility and escape single quotes.
        normalized = full.as_posix().replace("'", "'\\''")

        lines.append(f"file '{normalized}'")
    concat_path.write_text("\n".join(lines), encoding="utf-8")


def concat_clips(concat_path: Path, output_path: Path, ffmpeg: str = "ffmpeg") -> None:
    """Concatenate clips using ffmpeg's concat demuxer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c",
        "copy",
        str(output_path),
    ]
    _run_command(cmd)


def probe_duration(video_path: Path, *, ffprobe: str = "ffprobe", fallback: float | None = None) -> float:
    """Return media duration in seconds using ffprobe when available."""

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        if fallback is not None:
            return fallback
        raise RuntimeError("ffprobe not found; install FFmpeg or supply fallback duration") from exc

    if result.returncode == 0:
        output = result.stdout.strip()
        try:
            value = float(output)
            if math.isfinite(value):
                return value
        except ValueError:
            pass

    if fallback is not None:
        return fallback

    raise RuntimeError(
        f"Unable to determine duration for {video_path}. ffprobe output: {result.stderr.strip() or result.stdout.strip()}"
    )


def build_cheers_top10(
    video_path: Path,
    filtered_csv: Path,
    cheers_csv: Path | None,
    clips_dir: Path,
    concat_path: Path,
    output_video: Path,
    *,
    max_count: int = 10,
    pad_pre: float = 2.0,
    pad_post: float = 3.0,
    min_length: float = 0.8,
    overlap_threshold: float = 0.5,
    cheer_max: int = 4,
    cheer_spacing: float = 45.0,
    cheer_pre: float = 7.5,
    cheer_post: float = 2.5,
    cheer_min_length: float = 0.8,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    fallback_duration: float | None = None,
    csv_out: Path | None = None,
    skip_render: bool = False,
) -> List[SelectedClip]:
    """Orchestrate the full Top-10 build with optional cheer anchors."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = probe_duration(video_path, ffprobe=ffprobe, fallback=fallback_duration)

    candidates: List[ClipCandidate] = []
    if cheers_csv:
        candidates.extend(
            load_cheer_candidates(
                cheers_csv,
                duration,
                max_count=cheer_max,
                min_spacing=cheer_spacing,
                pre_window=cheer_pre,
                post_window=cheer_post,
                min_length=cheer_min_length,
            )
        )
    candidates.extend(load_filtered_candidates(filtered_csv))

    ranked = rank_candidates(candidates)
    if not ranked:
        raise RuntimeError("No highlight candidates found; check CSV inputs")

    selected = select_top_candidates(
        ranked,
        duration=duration,
        max_count=max_count,
        pad_pre=pad_pre,
        pad_post=pad_post,
        min_length=min_length,
        overlap_threshold=overlap_threshold,
    )

    if csv_out is not None:
        write_selection_csv(csv_out, selected)

    if skip_render or not selected:
        return selected

    clip_paths = render_clips(video_path, selected, clips_dir, ffmpeg=ffmpeg)
    # Filter out zero-sized clips as an extra guard before writing concat.
    clip_paths = [path for path in clip_paths if path.exists() and path.stat().st_size > 0]
    if not clip_paths:
        raise RuntimeError("No clips were rendered; ensure ffmpeg is available")

    write_concat_file(clip_paths, concat_path)
    concat_clips(concat_path, output_video, ffmpeg=ffmpeg)

    return selected


__all__ = [
    "ClipCandidate",
    "SelectedClip",
    "load_cheer_candidates",
    "load_filtered_candidates",
    "rank_candidates",
    "select_top_candidates",
    "write_selection_csv",
    "render_clips",
    "write_concat_file",
    "concat_clips",
    "probe_duration",
    "build_cheers_top10",
]
