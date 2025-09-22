"""CSV-driven clip and social reel exporter."""
from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:  # pragma: no cover - reuse helpers when available
    from tools.export_clips import (
        _anchor_time,
        _build_concat_file,
        _ffprobe_fps,
        _format_time,
        _run_ffmpeg,
        _sanitize_label,
        _write_ffconcat,
        _write_srt,
    )
except Exception:  # pragma: no cover
    def _ffprobe_fps(video: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if "/" in out:
            num, denom = out.split("/")
            fps = float(num) / float(denom) if float(denom) else float(num)
        else:
            fps = float(out)
        return fps

    def _build_concat_file(video: Path, start: float, end: float) -> Path:
        import tempfile

        tmp = tempfile.NamedTemporaryFile("w", suffix=".ffconcat", delete=False)
        try:
            tmp.write("ffconcat version 1.0\n")
            escaped = str(video).replace("'", "'\\''")
            tmp.write(f"file '{escaped}'\n")
            tmp.write(f"inpoint {start:.3f}\n")
            tmp.write(f"outpoint {end:.3f}\n")
            tmp.flush()
        finally:
            tmp.close()
        return Path(tmp.name)

    def _run_ffmpeg(cmd: List[str]) -> None:
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{process.stderr}")

    def _sanitize_label(label: str) -> str:
        import re

        cleaned = re.sub(r"[^0-9A-Z]+", "_", str(label).upper())
        return cleaned.strip("_") or "EVENT"

    def _format_time(value: float) -> str:
        return f"{value:.2f}"

    def _anchor_time(label: str, start: float, end: float) -> float:
        upper = str(label).upper()
        if any(token in upper for token in ["GOAL", "SHOT", "CROSS", "SAVE", "GK"]):
            return end
        return (start + end) / 2.0

    def _write_ffconcat(path: Path, files: Iterable[Path]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write("ffconcat version 1.0\n")
            for file_path in files:
                escaped = str(file_path).replace("'", "'\\''")
                fh.write(f"file '{escaped}'\n")

    def _write_srt(path: Path, events: pd.DataFrame) -> None:
        def _format_srt_time(seconds: float) -> str:
            millis = int(round(seconds * 1000))
            hrs, rem = divmod(millis, 3600_000)
            mins, rem = divmod(rem, 60_000)
            secs, ms = divmod(rem, 1000)
            return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for idx, row in enumerate(events.itertuples(index=False), start=1):
                fh.write(f"{idx}\n")
                fh.write(
                    f"{_format_srt_time(getattr(row, 't0'))} --> {_format_srt_time(getattr(row, 't1'))}\n"
                )
                fh.write(f"{getattr(row, 'label', 'EVENT')}\n\n")

from tools.reels_profiles import PROFILES, Profile, profile_filters, resolve_crop_box


@dataclass
class ClipSpec:
    index: int
    event_start: float
    event_end: float
    start: float
    end: float
    label: str
    speed: float
    gain_db: Optional[float]
    stem: str
    out_name: Optional[str]
    caption: Optional[str]
    t0_offset: float
    t1_offset: float


@dataclass
class VariantSpec:
    index: int
    label: str
    safe_label: str
    variant_name: str
    safe_variant: str
    profile: Profile
    start: float
    end: float
    speed: float
    gain_db: Optional[float]
    crop_box: Optional[Tuple[int, int, int, int]]
    out_name: Optional[str]
    caption: Optional[str]


TEMPLATE_COLUMNS = [
    "index",
    "label",
    "t0_adj",
    "t1_adj",
    "speed",
    "gain_db",
    "variant",
    "profile",
    "crop_w",
    "crop_h",
    "crop_x",
    "crop_y",
    "focus_x",
    "focus_y",
    "zoom",
    "out_name",
    "caption",
]


def _ffprobe_dimensions(video: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if not out:
        raise RuntimeError(f"Unable to determine dimensions for {video}")
    parts = out.split(",")
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected ffprobe output for {video}: {out}")
    return int(float(parts[0])), int(float(parts[1]))


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _to_float(value: object) -> Optional[float]:
    if _is_blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: object) -> Optional[str]:
    if _is_blank(value):
        return None
    return str(value).strip()


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"t0", "t1", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Events CSV is missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["t0"] = df["t0"].astype(float)
    df["t1"] = df["t1"].astype(float)
    df["label"] = df["label"].astype(str)
    return df


def load_adjustments(path: Optional[Path], n_events: int) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=TEMPLATE_COLUMNS)
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    if "index" not in df.columns:
        df["index"] = range(1, len(df) + 1)
    else:
        numeric_index = pd.to_numeric(df["index"], errors="coerce")
        fallback = pd.Series(range(1, len(df) + 1), index=df.index)
        df["index"] = numeric_index.fillna(fallback).astype(int)
    return df


def group_adjustments(df: pd.DataFrame) -> Dict[int, List[Dict[str, object]]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    if df.empty:
        return grouped
    for idx, group in df.groupby("index"):
        grouped[int(idx)] = group.to_dict(orient="records")
    return grouped


def apply_tweaks(events: pd.DataFrame, adjustments: Dict[int, List[Dict[str, object]]]) -> List[ClipSpec]:
    specs: List[ClipSpec] = []
    for idx, row in enumerate(events.itertuples(index=False), start=1):
        event_start = float(getattr(row, "t0"))
        event_end = float(getattr(row, "t1"))
        label = str(getattr(row, "label", "EVENT"))
        rows = adjustments.get(idx, [])
        base_row = next((item for item in rows if _is_blank(item.get("variant"))), None)

        base_t0 = _to_float(base_row.get("t0_adj")) if base_row else None
        base_t1 = _to_float(base_row.get("t1_adj")) if base_row else None
        base_t0 = base_t0 or 0.0
        base_t1 = base_t1 or 0.0

        start = max(0.0, event_start + base_t0)
        end = event_end + base_t1
        if base_row and not _is_blank(base_row.get("label")):
            label = str(base_row.get("label"))

        if end <= start:
            print(
                f"Skipping event {idx}: invalid timing after base adjustments ({start:.2f} >= {end:.2f})"
            )
            continue

        speed = _to_float(base_row.get("speed")) if base_row else None
        if speed is None or speed <= 0:
            speed = 1.0
        gain_db = _to_float(base_row.get("gain_db")) if base_row else None
        out_name = _to_str(base_row.get("out_name")) if base_row else None
        caption = _to_str(base_row.get("caption")) if base_row else None

        safe_label = _sanitize_label(label)
        if out_name:
            stem = f"{idx:03d}__{_sanitize_label(out_name)}"
        else:
            stem = f"{idx:03d}__{safe_label}__t{_format_time(start)}-t{_format_time(end)}"

        specs.append(
            ClipSpec(
                index=idx,
                event_start=event_start,
                event_end=event_end,
                start=start,
                end=end,
                label=label,
                speed=speed,
                gain_db=gain_db,
                stem=stem,
                out_name=_sanitize_label(out_name) if out_name else None,
                caption=caption,
                t0_offset=base_t0,
                t1_offset=base_t1,
            )
        )
    return specs


def variants_for(
    clip: ClipSpec,
    adjustment_rows: List[Dict[str, object]],
    profile_order: List[str],
    profiles: Dict[str, Profile],
    src_w: int,
    src_h: int,
) -> List[VariantSpec]:
    variants: List[VariantSpec] = []
    variant_rows = [row for row in adjustment_rows if not _is_blank(row.get("variant"))]

    if not variant_rows:
        for profile_name in profile_order:
            profile = profiles[profile_name]
            variants.append(
                VariantSpec(
                    index=clip.index,
                    label=clip.label,
                    safe_label=_sanitize_label(clip.label),
                    variant_name="base",
                    safe_variant=_sanitize_label("base"),
                    profile=profile,
                    start=clip.start,
                    end=clip.end,
                    speed=clip.speed,
                    gain_db=clip.gain_db,
                    crop_box=None,
                    out_name=clip.out_name,
                    caption=clip.caption,
                )
            )
        return variants

    for row in variant_rows:
        variant_name = _to_str(row.get("variant"))
        if not variant_name:
            continue

        profile_name = (_to_str(row.get("profile")) or profile_order[0]).lower()
        if profile_name not in profiles:
            raise ValueError(
                f"Adjustment for clip {clip.index} references unknown profile '{profile_name}'"
            )
        profile = profiles[profile_name]

        label = _to_str(row.get("label")) or clip.label
        speed = _to_float(row.get("speed")) or clip.speed
        if speed <= 0:
            speed = clip.speed
        gain = _to_float(row.get("gain_db"))
        if gain is None:
            gain = clip.gain_db

        t0_adj = _to_float(row.get("t0_adj")) or 0.0
        t1_adj = _to_float(row.get("t1_adj")) or 0.0

        start = max(0.0, clip.start + t0_adj)
        end = clip.end + t1_adj
        if end <= start:
            print(
                f"Skipping variant '{variant_name}' for clip {clip.index}: invalid timing after adjustments"
            )
            continue

        out_name = _to_str(row.get("out_name")) or clip.out_name
        caption = _to_str(row.get("caption")) or clip.caption

        explicit_crop = None
        if not _is_blank(row.get("crop_w")) and not _is_blank(row.get("crop_h")):
            crop_w = _to_float(row.get("crop_w"))
            crop_h = _to_float(row.get("crop_h"))
            crop_x = _to_float(row.get("crop_x")) or 0.0
            crop_y = _to_float(row.get("crop_y")) or 0.0
            if crop_w is not None and crop_h is not None:
                explicit_crop = (crop_w, crop_h, crop_x, crop_y)

        focus_x = _to_float(row.get("focus_x"))
        focus_y = _to_float(row.get("focus_y"))
        zoom = _to_float(row.get("zoom"))

        crop_box = resolve_crop_box(src_w, src_h, profile, explicit_crop, focus_x, focus_y, zoom)

        variants.append(
            VariantSpec(
                index=clip.index,
                label=label,
                safe_label=_sanitize_label(label),
                variant_name=variant_name,
                safe_variant=_sanitize_label(variant_name),
                profile=profile,
                start=start,
                end=end,
                speed=speed,
                gain_db=gain,
                crop_box=crop_box,
                out_name=_sanitize_label(out_name) if out_name else None,
                caption=caption,
            )
        )

    return variants


def _audio_filters(speed: float, gain_db: Optional[float]) -> str:
    filters: List[str] = ["aresample=async=1:first_pts=0"]
    tempo = speed if speed > 0 else 1.0
    if not math.isclose(tempo, 1.0, rel_tol=1e-6):
        filters.extend(_atempo_chain(tempo))
    if gain_db is not None:
        filters.append(f"volume={gain_db}dB")
    return ",".join(filters)


def _atempo_chain(speed: float) -> List[str]:
    filters: List[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    return filters


def build_clip(
    video: Path,
    spec: ClipSpec,
    out_path: Path,
    fps: float,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat = _build_concat_file(video, spec.start, spec.end)
    try:
        cmd: List[str] = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-safe",
            "0",
            "-fflags",
            "+genpts",
            "-f",
            "concat",
            "-i",
            str(concat),
            "-vsync",
            "cfr",
            "-fps_mode",
            "cfr",
            "-r",
            f"{fps:.6f}",
        ]

        video_filters: List[str] = []
        if not math.isclose(spec.speed, 1.0, rel_tol=1e-6):
            video_filters.append(f"setpts=PTS/{spec.speed:.6f}")
        if video_filters:
            cmd.extend(["-filter:v", ",".join(video_filters)])

        cmd.extend(["-af", _audio_filters(spec.speed, spec.gain_db)])
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        _run_ffmpeg(cmd)
    finally:
        concat.unlink(missing_ok=True)


def build_variant(
    video: Path,
    clip: ClipSpec,
    spec: VariantSpec,
    out_path: Path,
    fps: float,
    crf: int,
    preset: str,
    audio_bitrate: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat = _build_concat_file(video, spec.start, spec.end)
    try:
        cmd: List[str] = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-safe",
            "0",
            "-fflags",
            "+genpts",
            "-f",
            "concat",
            "-i",
            str(concat),
            "-vsync",
            "cfr",
            "-fps_mode",
            "cfr",
            "-r",
            f"{fps:.6f}",
        ]

        video_filters: List[str] = []
        if not math.isclose(spec.speed, 1.0, rel_tol=1e-6):
            video_filters.append(f"setpts=PTS/{spec.speed:.6f}")
        video_filters.extend(profile_filters(spec.profile, spec.crop_box))
        cmd.extend(["-filter:v", ",".join(video_filters)])

        cmd.extend(["-af", _audio_filters(spec.speed, spec.gain_db)])
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-movflags",
                "+faststart",
                str(out_path),
            ]
        )
        _run_ffmpeg(cmd)
    finally:
        concat.unlink(missing_ok=True)


def _bucket_targets(label: str) -> List[str]:
    upper = label.upper()
    buckets: List[str] = []
    if "GOAL" in upper:
        buckets.append("goals")
    if any(token in upper for token in ["SHOT", "CROSS"]):
        buckets.append("shots")
    if any(token in upper for token in ["SAVE", "GK"]):
        buckets.append("saves")
    if any(token in upper for token in ["DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]):
        buckets.append("defense")
    if any(token in upper for token in ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE"]):
        buckets.append("offense")
    return buckets


def write_playlists(
    variants_dir: Path,
    bucket_lists: Dict[str, Dict[str, List[Path]]],
) -> None:
    for profile_name, buckets in bucket_lists.items():
        profile_dir = variants_dir / profile_name
        profile_dir.mkdir(parents=True, exist_ok=True)
        for bucket, files in buckets.items():
            if bucket == "all" or files:
                _write_ffconcat(profile_dir / f"{bucket}.ffconcat", files)


def maybe_emit_template(path: Path, events: pd.DataFrame) -> bool:
    if path.exists():
        return False
    rows = []
    for idx, row in enumerate(events.itertuples(index=False), start=1):
        rows.append({"index": idx, "label": getattr(row, "label", "")})
    template = pd.DataFrame(rows, columns=[col for col in TEMPLATE_COLUMNS if col in {"index", "label"}])
    for col in TEMPLATE_COLUMNS:
        if col not in template.columns:
            template[col] = pd.NA
    template = template[TEMPLATE_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(path, index=False)
    return True


def thumbnail_path(clips_dir: Path, clip: ClipSpec) -> Path:
    return clips_dir / f"{clip.index:03d}__thumb.jpg"


def extract_thumbnail(video: Path, clip: ClipSpec, clips_dir: Path) -> None:
    anchor = _anchor_time(clip.label, clip.start, clip.end)
    anchor = max(clip.start, min(anchor, clip.end))
    thumb = thumbnail_path(clips_dir, clip)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-ss",
        f"{anchor:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(thumb),
    ]
    _run_ffmpeg(cmd)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export event clips and social variants")
    parser.add_argument("--video", required=True, type=Path, help="Source game video")
    parser.add_argument("--events", required=True, type=Path, help="Events CSV")
    parser.add_argument("--clips-dir", required=True, type=Path, help="Directory for base clips")
    parser.add_argument("--variants-dir", required=True, type=Path, help="Directory for social variants")
    parser.add_argument(
        "--profiles",
        default="tiktok,instagram,landscape",
        help="Comma-separated list of output profiles",
    )
    parser.add_argument("--adjustments", type=Path, help="Optional adjustments CSV")
    parser.add_argument(
        "--emit-template",
        action="store_true",
        help="Write template adjustments CSV (no rendering)",
    )
    parser.add_argument("--overlay", type=Path, help="Optional SRT output path")
    parser.add_argument("--crf", type=int, default=20, help="CRF for libx264")
    parser.add_argument("--preset", default="veryfast", help="Encoder preset")
    parser.add_argument("--audio-bitrate", default="160k", help="Audio bitrate (e.g. 160k)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    profiles_input = [name.strip().lower() for name in args.profiles.split(",") if name.strip()]
    if not profiles_input:
        raise ValueError("At least one profile must be specified")

    for name in profiles_input:
        if name not in PROFILES:
            raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(PROFILES)}")

    profiles = {name: PROFILES[name] for name in profiles_input}

    events = load_events(args.events)

    if args.emit_template:
        if args.adjustments is None:
            raise ValueError("--emit-template requires --adjustments path")
        created = maybe_emit_template(args.adjustments, events)
        if created:
            print(f"Wrote template adjustments to {args.adjustments}")
        else:
            print(f"Adjustments file already exists at {args.adjustments}, nothing written")
        return

    adjustments_df = load_adjustments(args.adjustments, len(events))
    adjustment_groups = group_adjustments(adjustments_df)

    clips = apply_tweaks(events, adjustment_groups)
    if not clips:
        print("No clips to export after applying adjustments")
        return

    fps = _ffprobe_fps(args.video)
    src_w, src_h = _ffprobe_dimensions(args.video)

    args.clips_dir.mkdir(parents=True, exist_ok=True)
    args.variants_dir.mkdir(parents=True, exist_ok=True)

    bucket_lists: Dict[str, Dict[str, List[Path]]] = {}
    for name in profiles_input:
        bucket_lists[PROFILES[name].name] = {
            "all": [],
            "goals": [],
            "shots": [],
            "saves": [],
            "defense": [],
            "offense": [],
        }

    for clip in clips:
        clip_path = args.clips_dir / f"{clip.stem}.mp4"
        print(f"Exporting clip {clip.index:03d} -> {clip_path}")
        build_clip(args.video, clip, clip_path, fps, args.crf, args.preset, args.audio_bitrate)
        extract_thumbnail(args.video, clip, args.clips_dir)

        rows = adjustment_groups.get(clip.index, [])
        variant_specs = variants_for(clip, rows, profiles_input, profiles, src_w, src_h)

        for spec in variant_specs:
            base = (
                f"{clip.index:03d}__{spec.out_name}"
                if spec.out_name
                else f"{clip.index:03d}__{spec.safe_label}__t{_format_time(spec.start)}-t{_format_time(spec.end)}"
            )
            filename = f"{base}__{spec.safe_variant}__{spec.profile.name}.mp4"
            variant_path = args.variants_dir / spec.profile.name / filename
            print(f"  Variant {spec.variant_name} [{spec.profile.name}] -> {variant_path}")
            build_variant(
                args.video,
                clip,
                spec,
                variant_path,
                fps,
                args.crf,
                args.preset,
                args.audio_bitrate,
            )
            if spec.caption:
                caption_path = variant_path.with_suffix(".txt")
                caption_path.write_text(spec.caption, encoding="utf-8")

            resolved = variant_path.resolve()
            profile_key = spec.profile.name
            bucket_lists.setdefault(
                profile_key,
                {
                    "all": [],
                    "goals": [],
                    "shots": [],
                    "saves": [],
                    "defense": [],
                    "offense": [],
                },
            )
            bucket_lists[profile_key]["all"].append(resolved)
            for bucket in _bucket_targets(spec.label):
                bucket_lists[profile_key][bucket].append(resolved)

    write_playlists(args.variants_dir, bucket_lists)

    if args.overlay is not None:
        overlay_rows = [
            {"t0": clip.start, "t1": clip.end, "label": clip.label}
            for clip in clips
        ]
        overlay_df = pd.DataFrame(overlay_rows)
        _write_srt(args.overlay, overlay_df)
        print(f"Wrote overlay SRT to {args.overlay}")


if __name__ == "__main__":
    main()
