"""Assemble a branded highlight reel from pre-rendered portrait clips.

Scans a clips directory for a game, selects the best moments by priority,
orders them for social-media pacing, and renders a single polished reel
with crossfade transitions, title card intro, end card outro, and watermark.

Usage
-----
    python tools/assemble_game_reel.py \
        --clips-dir out/portrait_reels/clean \
        --game 2026-02-21__TSC_vs_Greenwood \
        --brand tsc \
        --target-duration 90 \
        --out out/reels/2026-02-21__TSC_vs_Greenwood__highlight_reel.mp4

If ``--clips-dir`` points directly at a folder of MP4s (e.g. the atomic_clips
subfolder for a game), you can omit ``--game``.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Priority scoring (mirrors make_reels.py)
# ---------------------------------------------------------------------------

PRIORITY_RULES: List[Tuple[int, List[str]]] = [
    (5, ["GOAL"]),
    (4, ["SHOT", "CROSS"]),
    (3, ["SAVE", "GK", "FREE KICK", "CORNER"]),
    (2, ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE", "DRIBBL", "SKILL", "PRESSURE"]),
    (1, ["DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]),
]


def _compute_priority(label: str) -> int:
    upper = label.upper()
    best = 0
    for score, tokens in PRIORITY_RULES:
        if any(tok in upper for tok in tokens):
            best = max(best, score)
    return best


# ---------------------------------------------------------------------------
# Clip discovery and metadata
# ---------------------------------------------------------------------------

@dataclass
class ClipInfo:
    path: Path
    index: int  # sequence number from filename (001, 002, ...)
    label: str  # event label extracted from filename
    priority: int
    duration: float = 0.0
    fps: float = 30.0
    width: int = 0
    height: int = 0


# Pattern: 001__Free Kick & Goal__551.43-567.97.mp4
# or with portrait suffix: 001__Free Kick & Goal__551.43-567.97_portrait_FINAL.mp4
_FILENAME_RE = re.compile(
    r"^(?P<idx>\d+)__(?P<label>.+?)__[\d.]+-[\d.]+(?:_portrait_FINAL|_WIDE_portrait_FINAL)?\.mp4$",
    re.IGNORECASE,
)


def _probe_clip(path: Path) -> Tuple[float, float, int, int]:
    """Return (duration_s, fps, width, height) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        info = json.loads(out)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return 0.0, 30.0, 0, 0

    duration = 0.0
    fmt = info.get("format", {})
    if "duration" in fmt:
        try:
            duration = float(fmt["duration"])
        except (ValueError, TypeError):
            pass

    fps = 30.0
    w, h = 0, 0
    streams = info.get("streams", [])
    if streams:
        s = streams[0]
        w = int(s.get("width", 0))
        h = int(s.get("height", 0))
        fr = s.get("avg_frame_rate", "30/1")
        if "/" in str(fr):
            num, denom = str(fr).split("/")
            denom_f = float(denom) if denom else 1.0
            fps = float(num) / denom_f if denom_f else float(num)
        else:
            try:
                fps = float(fr)
            except (ValueError, TypeError):
                pass

    return duration, fps, w, h


def discover_clips(clips_dir: Path) -> List[ClipInfo]:
    """Find and parse all MP4 clips in *clips_dir*."""
    clips: List[ClipInfo] = []
    for mp4 in sorted(clips_dir.glob("*.mp4")):
        m = _FILENAME_RE.match(mp4.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        label = m.group("label").strip()
        pri = _compute_priority(label)
        clips.append(ClipInfo(path=mp4, index=idx, label=label, priority=pri))
    return clips


def probe_clips(clips: List[ClipInfo]) -> None:
    """Populate duration/fps/resolution via ffprobe (in-place)."""
    for clip in clips:
        clip.duration, clip.fps, clip.width, clip.height = _probe_clip(clip.path)


# ---------------------------------------------------------------------------
# Selection and ordering
# ---------------------------------------------------------------------------

def select_clips(
    clips: List[ClipInfo],
    target_seconds: float,
) -> List[ClipInfo]:
    """Pick the best clips within *target_seconds* total duration.

    Strategy: greedy selection by priority (descending), breaking ties
    by chronological order.
    """
    if not clips:
        return []

    ranked = sorted(clips, key=lambda c: (-c.priority, c.index))
    selected: List[ClipInfo] = []
    accumulated = 0.0

    for clip in ranked:
        if clip.duration <= 0:
            continue
        if accumulated + clip.duration <= target_seconds or not selected:
            selected.append(clip)
            accumulated += clip.duration
        if accumulated >= target_seconds:
            break

    return selected


def order_for_pacing(clips: List[ClipInfo]) -> List[ClipInfo]:
    """Reorder clips for social-media pacing.

    Strategy ("impact sandwich"):
      - Open with the highest-priority clip (hook)
      - Close with the second-highest-priority clip (payoff)
      - Fill the middle in chronological order
    """
    if len(clips) <= 2:
        return sorted(clips, key=lambda c: (-c.priority, c.index))

    by_priority = sorted(clips, key=lambda c: (-c.priority, c.index))
    opener = by_priority[0]
    closer = by_priority[1]

    middle = [c for c in clips if c is not opener and c is not closer]
    middle.sort(key=lambda c: c.index)

    return [opener] + middle + [closer]


# ---------------------------------------------------------------------------
# Brand asset resolution
# ---------------------------------------------------------------------------

@dataclass
class BrandAssets:
    title_ribbon: Optional[Path] = None
    end_card: Optional[Path] = None
    watermark: Optional[Path] = None
    brand_dir: Optional[Path] = None


def resolve_brand(brand_name: str, repo_root: Path, aspect: str = "9x16") -> BrandAssets:
    """Locate brand assets for the given brand and aspect ratio."""
    brand_dir = repo_root / "brand" / brand_name
    if not brand_dir.is_dir():
        print(f"[warn] Brand directory not found: {brand_dir}", file=sys.stderr)
        return BrandAssets()

    kit_path = brand_dir / "brand_kit.json"
    overlays: Dict[str, str] = {}
    if kit_path.exists():
        try:
            kit = json.loads(kit_path.read_text(encoding="utf-8"))
            overlays = kit.get("overlays", {})
        except (json.JSONDecodeError, KeyError):
            pass

    aspect_key = aspect.replace(":", "x")  # "9:16" -> "9x16"
    ribbon_key = f"title_ribbon_{aspect_key}"
    endcard_key = f"end_card_{aspect_key}"

    def _resolve(key: str) -> Optional[Path]:
        name = overlays.get(key)
        if name:
            p = brand_dir / name
            if p.exists():
                return p
        return None

    wm_name = (kit.get("watermarks", {}) if kit_path.exists() else {}).get("corner_watermark")
    wm_path = None
    if wm_name:
        wm_path = brand_dir / wm_name
        if not wm_path.exists():
            wm_path = None
    # Fallback to transparent variant
    if wm_path is None:
        fallback = brand_dir / "watermark_corner_256_transparent.png"
        if fallback.exists():
            wm_path = fallback

    return BrandAssets(
        title_ribbon=_resolve(ribbon_key),
        end_card=_resolve(endcard_key),
        watermark=wm_path,
        brand_dir=brand_dir,
    )


# ---------------------------------------------------------------------------
# FFmpeg reel assembly
# ---------------------------------------------------------------------------

def _escape_ffmpeg_path(path: Path) -> str:
    """Return a forward-slash path safe for FFmpeg on Windows and Linux."""
    return str(path.resolve()).replace("\\", "/")


def build_reel(
    clips: List[ClipInfo],
    brand: BrandAssets,
    output: Path,
    crossfade: float = 0.5,
    title_duration: float = 2.0,
    endcard_duration: float = 3.0,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
    title_text: str = "",
) -> None:
    """Render the final highlight reel with transitions and branding."""
    if not clips:
        raise ValueError("No clips to assemble")

    # Use fps from first clip
    fps = clips[0].fps or 30.0
    # Determine output dimensions from first clip (or default portrait)
    out_w = clips[0].width or 1080
    out_h = clips[0].height or 1920

    cmd: List[str] = ["ffmpeg", "-hide_banner", "-y"]
    filters: List[str] = []

    # ---------------------------------------------------------------
    # Inputs
    # ---------------------------------------------------------------
    input_idx = 0

    # Title card (from PNG or black + text)
    has_title = brand.title_ribbon is not None
    if has_title:
        cmd += ["-loop", "1", "-t", f"{title_duration:.3f}",
                "-i", str(brand.title_ribbon)]
        cmd += ["-f", "lavfi", "-t", f"{title_duration:.3f}",
                "-i", f"anullsrc=r=48000:cl=stereo"]
        title_v_idx = input_idx
        title_a_idx = input_idx + 1
        input_idx += 2
    else:
        # Generate a black title slate with text
        cmd += ["-f", "lavfi", "-t", f"{title_duration:.3f}",
                "-i", f"color=c=black:size={out_w}x{out_h}:r={fps:.6f}"]
        cmd += ["-f", "lavfi", "-t", f"{title_duration:.3f}",
                "-i", f"anullsrc=r=48000:cl=stereo"]
        title_v_idx = input_idx
        title_a_idx = input_idx + 1
        input_idx += 2

    # Clip inputs
    clip_input_indices: List[int] = []
    for clip in clips:
        cmd += ["-i", str(clip.path)]
        clip_input_indices.append(input_idx)
        input_idx += 1

    # End card
    has_endcard = brand.end_card is not None
    if has_endcard:
        cmd += ["-loop", "1", "-t", f"{endcard_duration:.3f}",
                "-i", str(brand.end_card)]
        cmd += ["-f", "lavfi", "-t", f"{endcard_duration:.3f}",
                "-i", f"anullsrc=r=48000:cl=stereo"]
        end_v_idx = input_idx
        end_a_idx = input_idx + 1
        input_idx += 2

    # Watermark
    has_watermark = brand.watermark is not None
    if has_watermark:
        cmd += ["-i", str(brand.watermark)]
        wm_idx = input_idx
        input_idx += 1

    # ---------------------------------------------------------------
    # Filter graph: prep each segment
    # ---------------------------------------------------------------

    # Title card
    filters.append(
        f"[{title_v_idx}:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1,fps={fps:.6f},format=yuv420p[vtitle]"
    )
    filters.append(f"[{title_a_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo[atitle]")

    video_labels = ["vtitle"]
    audio_labels = ["atitle"]
    durations = [title_duration]

    # Clip segments
    for i, (clip, in_idx) in enumerate(zip(clips, clip_input_indices)):
        vlabel = f"v{i}"
        alabel = f"a{i}"
        filters.append(
            f"[{in_idx}:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={fps:.6f},format=yuv420p[{vlabel}]"
        )
        filters.append(
            f"[{in_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"alimiter=limit=0.9[{alabel}]"
        )
        video_labels.append(vlabel)
        audio_labels.append(alabel)
        durations.append(clip.duration)

    # End card
    if has_endcard:
        filters.append(
            f"[{end_v_idx}:v]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={fps:.6f},format=yuv420p[vend]"
        )
        filters.append(f"[{end_a_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo[aend]")
        video_labels.append("vend")
        audio_labels.append("aend")
        durations.append(endcard_duration)

    # ---------------------------------------------------------------
    # Filter graph: xfade chain
    # ---------------------------------------------------------------

    current_v = video_labels[0]
    current_a = audio_labels[0]
    timeline = durations[0]

    for i in range(1, len(video_labels)):
        src_v = video_labels[i]
        src_a = audio_labels[i]
        out_v = f"xv{i}"
        out_a = f"xa{i}"

        prev_dur = durations[i - 1]
        curr_dur = durations[i]
        cf = min(crossfade, prev_dur / 2.0, curr_dur / 2.0)
        if cf < 0.001:
            cf = 0.001

        offset = max(timeline - cf - 0.001, 0.0)

        filters.append(
            f"[{current_v}][{src_v}]xfade=transition=fade:duration={cf:.3f}:offset={offset:.3f}[{out_v}]"
        )
        filters.append(
            f"[{current_a}][{src_a}]acrossfade=d={cf:.3f}:curve1=tri:curve2=tri[{out_a}]"
        )
        current_v = out_v
        current_a = out_a
        timeline += curr_dur - cf

    # ---------------------------------------------------------------
    # Filter graph: watermark overlay
    # ---------------------------------------------------------------

    if has_watermark:
        wm_label = f"wm_scaled"
        # Scale watermark, position bottom-right with 32px padding, 65% opacity
        filters.append(
            f"[{wm_idx}:v]format=rgba,scale=192:-1[{wm_label}]"
        )
        final_v = "vfinal"
        filters.append(
            f"[{current_v}][{wm_label}]overlay=x=main_w-overlay_w-32:y=main_h-overlay_h-32:"
            f"format=auto[{final_v}]"
        )
        current_v = final_v

    # Final format
    filters.append(f"[{current_v}]format=yuv420p[vout]")
    filters.append(f"[{current_a}]volume=1.0[aout]")

    # ---------------------------------------------------------------
    # Output encoding
    # ---------------------------------------------------------------

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-map", "[aout]",
        "-r", f"{fps:.6f}",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-g", "48",
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", "48000",
        "-movflags", "+faststart",
        str(output),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[reel] Assembling {len(clips)} clips → {output}")
    print(f"[reel] Estimated duration: {timeline:.1f}s")

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"FFmpeg failed (exit {result.returncode})")

    print(f"[reel] Done → {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _game_label_to_title(game_label: str) -> str:
    """Convert '2026-02-21__TSC_vs_Greenwood' → 'TSC vs Greenwood'."""
    # Strip date prefix
    parts = game_label.split("__", 1)
    name = parts[-1] if len(parts) > 1 else parts[0]
    return name.replace("_", " ")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble a branded highlight reel from portrait clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--clips-dir", type=Path, required=True,
        help="Directory containing rendered portrait clips (or atomic clips).",
    )
    p.add_argument(
        "--game", type=str, default=None,
        help="Game subfolder name within clips-dir (e.g. 2026-02-21__TSC_vs_Greenwood). "
             "Omit if clips-dir already points at the game folder.",
    )
    p.add_argument(
        "--brand", type=str, default="tsc",
        help="Brand name for title/endcard/watermark assets (default: tsc).",
    )
    p.add_argument(
        "--target-duration", type=float, default=90.0,
        help="Target reel duration in seconds (default: 90). Use 0 for no cap.",
    )
    p.add_argument(
        "--crossfade", type=float, default=0.5,
        help="Crossfade duration between clips in seconds (default: 0.5).",
    )
    p.add_argument(
        "--title-duration", type=float, default=2.0,
        help="Title card duration in seconds (default: 2.0).",
    )
    p.add_argument(
        "--endcard-duration", type=float, default=3.0,
        help="End card duration in seconds (default: 3.0).",
    )
    p.add_argument(
        "--title", type=str, default=None,
        help="Title text (auto-generated from game name if omitted).",
    )
    p.add_argument(
        "--crf", type=int, default=18,
        help="x264 CRF quality (default: 18, lower = better).",
    )
    p.add_argument(
        "--preset", type=str, default="medium",
        help="x264 encoding preset (default: medium).",
    )
    p.add_argument(
        "--aspect", type=str, default="9x16", choices=["9x16", "16x9"],
        help="Output aspect ratio for brand asset selection (default: 9x16).",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output path. Auto-generated if omitted.",
    )
    p.add_argument(
        "--no-title-card", action="store_true",
        help="Skip the title card intro.",
    )
    p.add_argument(
        "--no-end-card", action="store_true",
        help="Skip the end card outro.",
    )
    p.add_argument(
        "--no-watermark", action="store_true",
        help="Skip the watermark overlay.",
    )
    p.add_argument(
        "--chronological", action="store_true",
        help="Keep clips in chronological order instead of impact-sandwich pacing.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be assembled without rendering.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Resolve clips directory
    clips_dir = args.clips_dir
    if args.game:
        clips_dir = clips_dir / args.game
    if not clips_dir.is_dir():
        print(f"[error] Clips directory not found: {clips_dir}", file=sys.stderr)
        return 1

    # Discover clips
    print(f"[reel] Scanning {clips_dir} ...")
    clips = discover_clips(clips_dir)
    if not clips:
        print(f"[error] No clips found in {clips_dir}", file=sys.stderr)
        return 1
    print(f"[reel] Found {len(clips)} clips")

    # Probe durations
    print("[reel] Probing clip metadata ...")
    probe_clips(clips)
    total = sum(c.duration for c in clips)
    print(f"[reel] Total available: {total:.1f}s across {len(clips)} clips")

    # Select
    target = args.target_duration if args.target_duration > 0 else total
    selected = select_clips(clips, target)
    sel_total = sum(c.duration for c in selected)
    print(f"[reel] Selected {len(selected)} clips ({sel_total:.1f}s) for ~{target:.0f}s target")

    # Order
    if args.chronological:
        ordered = sorted(selected, key=lambda c: c.index)
    else:
        ordered = order_for_pacing(selected)

    # Display selection
    print()
    print("  #  | Pri | Dur    | Label")
    print("  ---|-----|--------|-------------------------------")
    for i, c in enumerate(ordered, 1):
        marker = ""
        if i == 1:
            marker = " ← OPENER"
        elif i == len(ordered):
            marker = " ← CLOSER"
        print(f"  {i:2d} |  {c.priority}  | {c.duration:5.1f}s | {c.label}{marker}")
    print()

    if args.dry_run:
        print("[dry-run] Would assemble the above clips. Exiting.")
        return 0

    # Resolve brand assets
    repo_root = Path(__file__).resolve().parent.parent
    brand = resolve_brand(args.brand, repo_root, args.aspect)

    if args.no_title_card:
        brand.title_ribbon = None
    if args.no_end_card:
        brand.end_card = None
    if args.no_watermark:
        brand.watermark = None

    assets_found = []
    if brand.title_ribbon:
        assets_found.append(f"title ribbon: {brand.title_ribbon.name}")
    if brand.end_card:
        assets_found.append(f"end card: {brand.end_card.name}")
    if brand.watermark:
        assets_found.append(f"watermark: {brand.watermark.name}")
    if assets_found:
        print(f"[reel] Brand assets: {', '.join(assets_found)}")
    else:
        print("[reel] No brand assets (plain assembly)")

    # Determine output path
    game_label = args.game or clips_dir.name
    output = args.out
    if output is None:
        output = repo_root / "out" / "reels" / f"{game_label}__highlight_reel.mp4"

    # Title text
    title = args.title or _game_label_to_title(game_label)

    # Build
    build_reel(
        clips=ordered,
        brand=brand,
        output=output,
        crossfade=args.crossfade,
        title_duration=args.title_duration if not args.no_title_card else 0.0,
        endcard_duration=args.endcard_duration if not args.no_end_card else 0.0,
        crf=args.crf,
        preset=args.preset,
        title_text=title,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
