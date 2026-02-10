"""Render highlight reels with transitions and slates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from ._loguru import logger

from .config import AppConfig
from .io import run_command


@dataclass
class ReelEntry:
    path: Path
    inpoint: float
    outpoint: float

    @property
    def duration(self) -> float:
        return max(0.0, self.outpoint - self.inpoint)


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\\\'")


def _parse_concat(list_path: Path) -> List[ReelEntry]:
    entries: List[ReelEntry] = []
    current = {}
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("file"):
            path = line.split(" ", 1)[1].strip().strip("'")
            current["path"] = Path(path)
        elif line.startswith("inpoint"):
            current["inpoint"] = float(line.split()[1])
        elif line.startswith("outpoint"):
            current["outpoint"] = float(line.split()[1])
            entries.append(ReelEntry(path=current["path"], inpoint=current.get("inpoint", 0.0), outpoint=current.get("outpoint", 0.0)))
            current = {}
    return entries


def render_reel(config: AppConfig, list_path: Path, output_path: Path, title: str, profile_name: str | None = None) -> float:
    """Render a highlight reel with crossfade transitions and title slate."""
    entries = _parse_concat(list_path)
    if not entries:
        logger.warning("No entries in concat list %s", list_path)
        return 0.0
    profile = config.resolve_profile(profile_name)
    crossfade = profile.crossfade_frames / profile.fps
    title_duration = profile.title_duration
    drawtext_title = _escape_drawtext(title)
    cmd: List[str] = ["ffmpeg", "-hide_banner", "-y"]
    cmd += [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:size={profile.width}x{profile.height}:duration={title_duration:.3f}",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=48000:cl=stereo:d={title_duration:.3f}",
    ]
    for entry in entries:
        cmd += ["-i", str(entry.path)]

    filters: List[str] = []
    filters.append(
        f"[0:v]drawtext=text='{drawtext_title}':x=(w-text_w)/2:y=(h-text_h)/2:fontcolor=white:fontsize=72:borderw=2,format=yuv420p[vtitle]"
    )
    filters.append(f"[1:a]anull[audtitle]")

    video_labels = ["vtitle"]
    audio_labels = ["audtitle"]
    durations = [title_duration]

    label_expr = profile.label_position.replace("W", "w").replace("H", "h")

    for idx, entry in enumerate(entries, start=1):
        input_idx = idx + 1  # accounts for title video/audio inputs
        number_text = _escape_drawtext(f"#{idx}")
        filters.append(
            f"[{input_idx}:v]trim=start={entry.inpoint:.3f}:end={entry.outpoint:.3f},setpts=PTS-STARTPTS,"
            f"scale=w={profile.width}:h={profile.height}:force_original_aspect_ratio=decrease,"
            f"pad={profile.width}:{profile.height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p,"
            f"drawtext=text='{number_text}':x={label_expr.split(':')[0]}:y={label_expr.split(':')[1]}:fontcolor=white:fontsize=64:borderw=2[v{idx}]"
        )
        filters.append(
            f"[{input_idx}:a]atrim=start={entry.inpoint:.3f}:end={entry.outpoint:.3f},asetpts=PTS-STARTPTS,alimiter=limit=0.9[a{idx}]"
        )
        video_labels.append(f"v{idx}")
        audio_labels.append(f"a{idx}")
        durations.append(entry.duration)

    current_v = video_labels[0]
    current_a = audio_labels[0]
    timeline = durations[0]
    for idx in range(1, len(video_labels)):
        src_v = video_labels[idx]
        src_a = audio_labels[idx]
        out_v = f"vmix{idx}"
        out_a = f"amix{idx}"
        prev_dur = durations[idx - 1]
        curr_dur = durations[idx]
        cf = min(crossfade, prev_dur / 2.0, curr_dur / 2.0) if crossfade > 0 else 0.0
        if cf < 1e-3:
            cf = 1e-3
        offset = max(timeline - cf, 0.0)
        # Subtract a tiny epsilon so rounding the formatted value never nudges the
        # offset past the trimmed duration, which would trigger frame drops in
        # ffmpeg's xfade/acrossfade filters.
        offset = max(offset - 1e-3, 0.0)
        filters.append(
            f"[{current_v}][{src_v}]xfade=transition=fade:duration={cf:.3f}:offset={offset:.3f}[{out_v}]"
        )
        filters.append(
            f"[{current_a}][{src_a}]acrossfade=d={cf:.3f}:curve1=tri:curve2=tri[{out_a}]"
        )
        current_v = out_v
        current_a = out_a
        timeline += curr_dur - cf

    filters.append(f"[{current_a}]volume=1.0[aout]")
    filters.append(f"[{current_v}]format=yuv420p[vout]")

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-r",
        f"{profile.fps}",
        "-c:v",
        "libx264",
        "-preset",
        config.clips.preset,
        "-crf",
        str(config.clips.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        config.clips.audio_bitrate,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(cmd)
    logger.info("Rendered reel to %s", output_path)
    return timeline
