#!/usr/bin/env python3
"""Season compilation: pick the best action from each game and assemble with
branded transition slates into a single highlight reel.

Data Sources (loaded in priority order)
---------------------------------------
1. out/catalog/atomic_index.csv — canonical clip catalog (274+ clips, 14+ games)
2. out/catalog/{game}/events_selected.csv — annotated plays for games not yet
   in the atomic index (auto gap-fill for newly annotated games)
3. AtomicClips.All.csv — legacy fallback (4 early games, used only if
   nothing else is found)

Workflow
-------
1. Scan the unified clip catalog for all games.
2. For each game, rank clips by social-media impact and pick the top one.
3. Generate a branded transition slate between each game segment.
4. Assemble everything with crossfade transitions, title card, and end card.

There are two rendering modes:
  - **ffmpeg** (default) — generate slates via FFmpeg drawtext using TSC brand
    colors and Montserrat. Runs entirely on the local machine.
  - **canva** — generate premium slates via Canva design generation API and
    export as PNGs. Requires Claude + Canva MCP integration.

Usage
-----
    # Preview what would be assembled (uses atomic_index + events_selected):
    python tools/season_compilation.py plan

    # Build with FFmpeg slates (local):
    python tools/season_compilation.py build \
        --portrait-root D:\\Projects\\soccer-video\\out\\portrait_reels

    # Generate Canva slate specs (for Claude to execute):
    python tools/season_compilation.py canva-slates

    # Use legacy CSV only (override):
    python tools/season_compilation.py plan \
        --legacy-csv AtomicClips.All.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
BRAND_DIR = ROOT / "brand" / "tsc"
BRAND_KIT_PATH = BRAND_DIR / "brand_kit.json"
CONFIG_PATH = ROOT / "social" / "social_config.json"

# ---------------------------------------------------------------------------
# Brand constants
# ---------------------------------------------------------------------------

BRAND_COLORS = {
    "navy": "#1F2B3D",
    "red": "#9B1B33",
    "gold": "#B7A37C",
    "white": "#FFFFFF",
    "black": "#000000",
}

# FFmpeg hex format (0xRRGGBB)
FF_NAVY = "0x1F2B3D"
FF_RED  = "0x9B1B33"
FF_GOLD = "0xB7A37C"
FF_WHITE = "0xFFFFFF"

# ---------------------------------------------------------------------------
# Clip scoring (shared logic with social_kit_pipeline.py)
# ---------------------------------------------------------------------------

PRIORITY_RULES: List[Tuple[int, List[str]]] = [
    (5, ["GOAL"]),
    (4, ["SHOT", "CROSS"]),
    (3, ["SAVE", "GK", "FREE KICK", "CORNER"]),
    (2, ["BUILD", "OFFENSE", "ATTACK", "PASS", "COMBINE", "DRIBBL",
          "SKILL", "PRESSURE"]),
    (1, ["DEFENSE", "TACKLE", "INTERCEPT", "BLOCK", "CLEAR"]),
]

ENGAGEMENT_BOOST = {
    "GOAL": 1.5, "SHOT": 1.2, "SAVE": 1.3,
    "DRIBBL": 1.1, "FREE_KICK": 1.1,
}


def compute_priority(label: str) -> int:
    upper = label.upper()
    best = 0
    for score, tokens in PRIORITY_RULES:
        if any(tok in upper for tok in tokens):
            best = max(best, score)
    return best


def compute_social_score(label: str, duration: float) -> float:
    pri = compute_priority(label)
    upper = label.upper()
    boost = 1.0
    for token, mult in ENGAGEMENT_BOOST.items():
        if token in upper:
            boost = max(boost, mult)
    if 8.0 <= duration <= 20.0:
        dur_bonus = 1.2
    elif 5.0 <= duration <= 30.0:
        dur_bonus = 1.0
    else:
        dur_bonus = 0.7
    return pri * boost * dur_bonus


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClipRecord:
    """A single clip from the catalog."""
    key: str
    idx: str
    date: str
    home: str
    away: str
    label: str
    t_start: float
    t_end: float
    duration: float
    priority: int
    social_score: float
    clip_folder: str = ""
    clip_folder_path: str = ""
    branded_paths: str = ""
    stabilized_path: str = ""
    proxy_path: str = ""
    fps: float = 30.0


@dataclass
class GameSummary:
    """Summary of one game with its best clip."""
    date: str
    home: str
    away: str
    game_label: str          # e.g. "2025-09-13__TSC_vs_NEOFC"
    total_clips: int
    best_clip: Optional[ClipRecord]
    all_clips: List[ClipRecord] = field(default_factory=list)


@dataclass
class SlateSpec:
    """Specification for a transition slate between games."""
    game: GameSummary
    slate_text_line1: str    # e.g. "TSC vs NEO FC"
    slate_text_line2: str    # e.g. "September 13, 2025"
    slate_text_line3: str    # e.g. "Best: GOAL (clip 002)"
    duration: float = 2.5


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> float:
    """Parse 't155.50', '155.50', or '0:06:37' to float seconds."""
    t = t.strip().lstrip("t")
    if not t:
        return 0.0
    # H:M:S or M:S format
    if ":" in t:
        parts = t.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return 0.0
    try:
        return float(t)
    except ValueError:
        return 0.0


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def opponent_display_name(away_code: str, config: dict) -> str:
    """Convert opponent code to display name."""
    opponents = config.get("opponents", {})
    entry = opponents.get(away_code, {})
    return entry.get("name", away_code.replace("_", " "))


def format_game_date(date_str: str) -> str:
    """Convert '2025-09-13' to 'September 13, 2025'."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except ValueError:
        return date_str


GAME_FOLDER_ALIASES: Dict[str, str] = {
    "2025-10-04__TSC_vs_FC": "2025-10-04__TSC_vs_FC_Tulsa_Black",
    "2025-10-12__TSC_SLSG_FallFestival": "2025-10-12__TSC_vs_SLSG",
    "2025-11-04__TSC_-_Navy_Oct_2025": "2025-11-05__TSC_vs_TSC_Navy",
}


def _normalize_game_folder(folder: str) -> str:
    """Map variant folder names to canonical game keys."""
    return GAME_FOLDER_ALIASES.get(folder, folder)


def _parse_game_folder(folder: str) -> Tuple[str, str, str]:
    """Extract (date, home, away) from a canonical game folder name.

    Handles:  2025-09-13__TSC_vs_NEOFC  →  ('2025-09-13', 'TSC', 'NEOFC')
    """
    m = re.match(r"(\d{4}-\d{2}-\d{2})__(\w+)_vs_(.+)$", folder)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", "", ""


def _extract_label_from_clip_stem(stem: str, game_folder: str) -> str:
    """Extract the action label from a clip filename.

    Two naming conventions:
      Old: 001__2025-09-13__TSC_vs_NEOFC__SHOT__t155.50-t166.40
      New: 001__Free Kick & Goal__551.43-567.97
    """
    # Old convention: strip idx, game label, and timestamp → action tokens remain
    old_m = re.match(
        r"\d+__\d{4}-\d{2}-\d{2}__\w+_vs_\w+__(.+?)__t[\d.]+-t[\d.]+$",
        stem,
    )
    if old_m:
        return old_m.group(1).replace("_", " ")

    # New convention: strip idx and trailing timestamp
    new_m = re.match(r"\d+__(.+?)__[\d.]+-[\d.]+$", stem)
    if new_m:
        return new_m.group(1)

    # Fallback: everything after the leading index
    fb = re.match(r"\d+__(.+)$", stem)
    return fb.group(1) if fb else stem


def _extract_idx_from_stem(stem: str) -> str:
    """Pull the leading clip index (e.g. '001') from the stem."""
    m = re.match(r"(\d+)", stem)
    return m.group(1) if m else "000"


def load_clips_from_atomic_index(csv_path: Path) -> List[ClipRecord]:
    """Load clips from out/catalog/atomic_index.csv (canonical 16-game catalog).

    Columns: clip_id, clip_name, clip_path, clip_rel, clip_stem,
             created_at_utc, duration_s, width, height, fps, sha1_64,
             tags, t_start_s, t_end_s, master_path, master_rel
    """
    clips: List[ClipRecord] = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_rel = row.get("clip_rel", "") or row.get("clip_path", "")
            stem = row.get("clip_stem", "")

            # Identify the game from the folder path
            m = re.search(r"atomic_clips[/\\]([^/\\]+)", clip_rel)
            if not m:
                continue
            raw_folder = m.group(1)
            game_folder = _normalize_game_folder(raw_folder)
            date, home, away = _parse_game_folder(game_folder)
            if not date:
                continue

            # Duration — prefer the column, fall back to t_end - t_start
            dur_s = row.get("duration_s", "")
            t_start_s = row.get("t_start_s", "")
            t_end_s = row.get("t_end_s", "")
            t_start = float(t_start_s) if t_start_s else 0.0
            t_end = float(t_end_s) if t_end_s else 0.0
            duration = float(dur_s) if dur_s else (t_end - t_start if t_end > t_start else 0.0)

            label = _extract_label_from_clip_stem(stem, game_folder)
            idx = _extract_idx_from_stem(stem)

            # FPS — handle rational format like "30/1" or "24000/1001"
            fps_raw = row.get("fps", "")
            fps_val = 30.0
            if fps_raw:
                if "/" in fps_raw:
                    parts = fps_raw.split("/")
                    try:
                        fps_val = float(parts[0]) / float(parts[1])
                    except (ValueError, ZeroDivisionError):
                        fps_val = 30.0
                else:
                    try:
                        fps_val = float(fps_raw)
                    except ValueError:
                        fps_val = 30.0

            pri = compute_priority(label)
            score = compute_social_score(label, duration)

            clips.append(ClipRecord(
                key=row.get("clip_id", ""),
                idx=idx,
                date=date,
                home=home,
                away=away,
                label=label,
                t_start=t_start,
                t_end=t_end,
                duration=duration,
                priority=pri,
                social_score=score,
                clip_folder=raw_folder,
                clip_folder_path=row.get("clip_path", ""),
                fps=fps_val,
            ))
    return clips


def load_clips_from_events_selected(catalog_dir: Path) -> List[ClipRecord]:
    """Load annotated-but-not-yet-clipped plays from events_selected.csv files.

    These live in out/catalog/{game_label}/events_selected.csv and represent
    games that have been annotated but haven't run through the atomic clipping
    pipeline yet.

    Columns: id, game_label, brand, master_path, master_start, master_end,
             playtag, phase, side, formation, notes, ...
    """
    clips: List[ClipRecord] = []
    for es_path in sorted(catalog_dir.glob("*/events_selected.csv")):
        raw_folder = es_path.parent.name
        game_folder = _normalize_game_folder(raw_folder)
        date, home, away = _parse_game_folder(game_folder)
        if not date:
            continue

        with open(es_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ms = row.get("master_start", "0")
                me = row.get("master_end", "0")
                t_start = _parse_time(ms)
                t_end = _parse_time(me)
                duration = t_end - t_start if t_end > t_start else 0.0

                label = row.get("playtag", "").strip() or "HIGHLIGHT"
                idx = row.get("id", "000").strip()

                pri = compute_priority(label)
                score = compute_social_score(label, duration)

                clips.append(ClipRecord(
                    key=f"{game_folder}__{idx}",
                    idx=idx,
                    date=date,
                    home=home,
                    away=away,
                    label=label,
                    t_start=t_start,
                    t_end=t_end,
                    duration=duration,
                    priority=pri,
                    social_score=score,
                    clip_folder=raw_folder,
                    clip_folder_path=row.get("master_path", ""),
                ))
    return clips


def load_all_clips(
    atomic_index_path: Optional[Path] = None,
    catalog_dir: Optional[Path] = None,
    legacy_csv_path: Optional[Path] = None,
) -> List[ClipRecord]:
    """Unified loader: merge atomic_index + events_selected + legacy CSV.

    Priority:
      1. atomic_index.csv — clips already cut and on disk (canonical source)
      2. events_selected.csv — annotated plays ready to clip (gap-fill for
         games not yet in the atomic index)
      3. AtomicClips.All.csv — legacy fallback (only if nothing else found)
    """
    clips: List[ClipRecord] = []
    seen_games: set = set()

    # 1. atomic_index.csv
    if atomic_index_path and atomic_index_path.exists():
        ai_clips = load_clips_from_atomic_index(atomic_index_path)
        clips.extend(ai_clips)
        for c in ai_clips:
            seen_games.add(f"{c.date}|{c.home}|{c.away}")

    # 2. events_selected.csv — only for games NOT already covered
    if catalog_dir and catalog_dir.exists():
        es_clips = load_clips_from_events_selected(catalog_dir)
        for c in es_clips:
            game_key = f"{c.date}|{c.home}|{c.away}"
            if game_key not in seen_games:
                clips.append(c)
        # track newly added games
        for c in es_clips:
            seen_games.add(f"{c.date}|{c.home}|{c.away}")

    # 3. Legacy fallback
    if legacy_csv_path and legacy_csv_path.exists() and not clips:
        clips.extend(load_clips_from_legacy_csv(legacy_csv_path))

    return clips


def load_clips_from_legacy_csv(csv_path: Path) -> List[ClipRecord]:
    """Load clips from AtomicClips.All.csv (legacy 4-game format)."""
    clips = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("date", "").strip()
            home = row.get("home", "").strip()
            away = row.get("away", "").strip()
            label = row.get("label", "").strip()

            # Skip rows with no game info
            if not date or not home or date == "1970-01-01":
                continue

            t_start = _parse_time(row.get("t_start", "0"))
            t_end = _parse_time(row.get("t_end", "0"))
            duration = t_end - t_start if t_end > t_start else 0.0

            pri = compute_priority(label)
            score = compute_social_score(label, duration)

            clips.append(ClipRecord(
                key=row.get("key", ""),
                idx=row.get("idx", ""),
                date=date,
                home=home,
                away=away,
                label=label,
                t_start=t_start,
                t_end=t_end,
                duration=duration,
                priority=pri,
                social_score=score,
                clip_folder=row.get("clip_folder", ""),
                clip_folder_path=row.get("clip_folder_path", ""),
                branded_paths=row.get("branded_paths", ""),
                stabilized_path=row.get("stabilized_path", ""),
                proxy_path=row.get("proxy_path", ""),
            ))
    return clips


def group_by_game(clips: List[ClipRecord], config: dict) -> List[GameSummary]:
    """Group clips into games and pick the best from each."""
    games: Dict[str, List[ClipRecord]] = {}
    for clip in clips:
        key = f"{clip.date}|{clip.home}|{clip.away}"
        if key not in games:
            games[key] = []
        games[key].append(clip)

    summaries = []
    for key, game_clips in sorted(games.items()):
        date, home, away = key.split("|")
        game_label = f"{date}__{home}_vs_{away}"

        # Rank by social score
        ranked = sorted(game_clips, key=lambda c: (-c.social_score, -c.priority))
        best = ranked[0] if ranked else None

        summaries.append(GameSummary(
            date=date,
            home=home,
            away=away,
            game_label=game_label,
            total_clips=len(game_clips),
            best_clip=best,
            all_clips=ranked,
        ))

    return summaries


# ---------------------------------------------------------------------------
# Slate generation
# ---------------------------------------------------------------------------

def build_slate_specs(games: List[GameSummary], config: dict) -> List[SlateSpec]:
    """Create transition slate specifications for each game."""
    slates = []
    for game in games:
        away_name = opponent_display_name(game.away, config)
        line1 = f"TSC vs {away_name}"
        line2 = format_game_date(game.date)
        line3 = ""
        if game.best_clip:
            line3 = f"{game.best_clip.label} (clip {game.best_clip.idx})"

        slates.append(SlateSpec(
            game=game,
            slate_text_line1=line1,
            slate_text_line2=line2,
            slate_text_line3=line3,
        ))
    return slates


def build_canva_slate_prompts(
    slates: List[SlateSpec],
) -> List[dict]:
    """Build Canva design prompts for each game transition slate."""
    prompts = []
    for i, slate in enumerate(slates, 1):
        prompt = (
            f"Create a bold sports transition slate (1080x1920, portrait). "
            f"Brand colors: navy ({BRAND_COLORS['navy']}), red ({BRAND_COLORS['red']}), "
            f"gold ({BRAND_COLORS['gold']}). Montserrat ExtraBold typography. "
            f"Large centered headline: \"{slate.slate_text_line1}\". "
            f"Subheading below: \"{slate.slate_text_line2}\". "
            f"Small accent text: \"{slate.slate_text_line3}\". "
            f"Include dynamic diagonal speed lines, geometric accents, "
            f"and soccer/football graphic elements. "
            f"Add \"Tulsa Soccer Club\" small branding at top. "
            f"This is a transition card between game highlights in a season compilation. "
            f"Make it feel premium, bold, and match-day energetic."
        )
        prompts.append({
            "index": i,
            "game_label": slate.game.game_label,
            "design_type": "your_story",
            "prompt": prompt,
            "slate_data": {
                "line1": slate.slate_text_line1,
                "line2": slate.slate_text_line2,
                "line3": slate.slate_text_line3,
            },
        })
    return prompts


def generate_ffmpeg_slate(
    slate: SlateSpec,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: float = 30.0,
    font_path: str = "",
) -> Path:
    """Generate a transition slate PNG using FFmpeg with TSC brand styling.

    Creates a navy background with gold/white text overlay.
    """
    # Use Montserrat if available, otherwise fallback
    if not font_path:
        candidates = [
            str(ROOT / "fonts" / "Montserrat-ExtraBold.ttf"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "Sans",
        ]
        for c in candidates:
            if c == "Sans" or Path(c).exists():
                font_path = c
                break

    # Escape for FFmpeg drawtext
    def esc(s: str) -> str:
        return s.replace(":", "\\:").replace("'", "\\'").replace("%", "%%")

    line1 = esc(slate.slate_text_line1)
    line2 = esc(slate.slate_text_line2)
    line3 = esc(slate.slate_text_line3)

    # Build filter for a branded slate
    filters = [
        f"color=c={FF_NAVY}:s={width}x{height}:r={fps}:d={slate.duration}",
    ]

    # Add red accent bar at top
    draw_filters = [
        f"drawbox=x=0:y=0:w={width}:h=6:c={FF_RED}:t=fill",
        # Gold line accent
        f"drawbox=x=80:y={height//2 - 180}:w={width-160}:h=3:c={FF_GOLD}@0.6:t=fill",
        f"drawbox=x=80:y={height//2 + 120}:w={width-160}:h=3:c={FF_GOLD}@0.6:t=fill",
    ]

    # Club name at top
    draw_filters.append(
        f"drawtext=text='TULSA SOCCER CLUB':fontsize=32:fontcolor={FF_GOLD}@0.85"
        f":x=(w-text_w)/2:y=180"
    )

    # Main matchup text
    draw_filters.append(
        f"drawtext=text='{line1}':fontsize=72:fontcolor={FF_WHITE}"
        f":x=(w-text_w)/2:y=(h-text_h)/2-60"
    )

    # Date
    draw_filters.append(
        f"drawtext=text='{line2}':fontsize=40:fontcolor={FF_GOLD}@0.9"
        f":x=(w-text_w)/2:y=(h)/2+20"
    )

    # Best clip label
    if line3:
        draw_filters.append(
            f"drawtext=text='{line3}':fontsize=28:fontcolor={FF_WHITE}@0.6"
            f":x=(w-text_w)/2:y=(h)/2+80"
        )

    # Red accent bar at bottom
    draw_filters.append(
        f"drawbox=x=0:y={height-6}:w={width}:h=6:c={FF_RED}:t=fill"
    )

    # Add font file to drawtext if available
    if font_path != "Sans" and Path(font_path).exists():
        ff_font = font_path.replace("\\", "/")
        draw_filters = [
            dt.replace("drawtext=", f"drawtext=fontfile='{ff_font}':")
            if "drawtext=" in dt else dt
            for dt in draw_filters
        ]

    full_filter = ",".join(filters) + "," + ",".join(draw_filters)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-f", "lavfi", "-i", full_filter,
        "-frames:v", "1",
        "-update", "1",
        str(output_path),
    ]

    # For video slates (mp4), use different output
    if output_path.suffix == ".mp4":
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-f", "lavfi", "-i", full_filter,
            "-f", "lavfi", "-t", str(slate.duration),
            "-i", f"anullsrc=r=48000:cl=stereo",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]

    return output_path


# ---------------------------------------------------------------------------
# Clip resolution (find actual video files)
# ---------------------------------------------------------------------------

def resolve_clip_video(
    clip: ClipRecord,
    portrait_root: Optional[Path] = None,
) -> Optional[Path]:
    """Find the actual video file for a clip.

    Search order:
    1. branded_paths from catalog
    2. stabilized_path from catalog
    3. portrait_reels directory tree
    4. proxy_path from catalog
    """
    # Try branded paths
    if clip.branded_paths:
        for bp in clip.branded_paths.split("|"):
            p = Path(bp.strip())
            if p.exists():
                return p

    # Try stabilized
    if clip.stabilized_path:
        p = Path(clip.stabilized_path)
        if p.exists():
            return p

    # Try portrait_reels directory
    if portrait_root:
        game_dir = portrait_root / f"{clip.date}__{clip.home}_vs_{clip.away}"
        if game_dir.is_dir():
            # Look for matching clip index
            pattern = f"{clip.idx}__*"
            matches = list(game_dir.glob(f"{pattern}.mp4"))
            if matches:
                return matches[0]

        # Also try without game subfolder
        matches = list(portrait_root.glob(f"**/{clip.idx}__*.mp4"))
        if matches:
            return matches[0]

    # Try proxy
    if clip.proxy_path:
        p = Path(clip.proxy_path)
        if p.exists():
            return p

    return None


# ---------------------------------------------------------------------------
# Assembly manifest
# ---------------------------------------------------------------------------

@dataclass
class CompilationManifest:
    """Full compilation manifest for the season reel."""
    title: str
    generated_at: str
    total_games: int
    total_clips_available: int
    segments: List[dict]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def build_manifest(
    games: List[GameSummary],
    slates: List[SlateSpec],
    config: dict,
    portrait_root: Optional[Path] = None,
) -> CompilationManifest:
    """Build the full compilation manifest."""
    segments = []
    total_available = 0

    for game, slate in zip(games, slates):
        total_available += game.total_clips

        clip_path = None
        if game.best_clip and portrait_root:
            clip_path = resolve_clip_video(game.best_clip, portrait_root)

        segment = {
            "game_label": game.game_label,
            "date": game.date,
            "opponent": opponent_display_name(game.away, config),
            "clips_in_game": game.total_clips,
            "best_clip": {
                "idx": game.best_clip.idx if game.best_clip else None,
                "label": game.best_clip.label if game.best_clip else None,
                "priority": game.best_clip.priority if game.best_clip else None,
                "social_score": game.best_clip.social_score if game.best_clip else None,
                "duration": game.best_clip.duration if game.best_clip else None,
                "video_resolved": str(clip_path) if clip_path else None,
            },
            "slate": {
                "line1": slate.slate_text_line1,
                "line2": slate.slate_text_line2,
                "line3": slate.slate_text_line3,
            },
            "runner_up_clips": [
                {
                    "idx": c.idx,
                    "label": c.label,
                    "score": c.social_score,
                    "duration": c.duration,
                }
                for c in game.all_clips[1:4]  # top 3 runners-up
            ],
        }
        segments.append(segment)

    return CompilationManifest(
        title="TSC 2025-26 Season Highlights",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        total_games=len(games),
        total_clips_available=total_available,
        segments=segments,
    )


# ---------------------------------------------------------------------------
# FFmpeg assembly (builds the actual video)
# ---------------------------------------------------------------------------

def build_ffmpeg_compilation(
    manifest: CompilationManifest,
    output_path: Path,
    slate_dir: Path,
    title_duration: float = 3.0,
    slate_duration: float = 2.5,
    endcard_duration: float = 3.0,
    crossfade: float = 0.5,
    crf: int = 18,
    fps: float = 30.0,
) -> None:
    """Build the full compilation video using ffconcat or xfade chain.

    This generates a script that can be run on the local machine where
    the video files actually exist.
    """
    out_w, out_h = 1080, 1920

    # Generate the PowerShell build script (to run on the PC)
    lines = [
        "# Season Compilation Build Script",
        f"# Generated: {manifest.generated_at}",
        f"# Games: {manifest.total_games}",
        "",
        '$ErrorActionPreference = "Stop"',
        f'$outW = {out_w}',
        f'$outH = {out_h}',
        f'$fps = {fps}',
        f'$crf = {crf}',
        f'$xfade = {crossfade}',
        "",
        '# Segment list: [slate_path, clip_path] pairs',
        '$segments = @(',
    ]

    for seg in manifest.segments:
        clip_path = seg["best_clip"].get("video_resolved", "")
        slate_path = str(slate_dir / f"{seg['game_label']}__slate.mp4")
        if clip_path:
            lines.append(f'  @("{slate_path}", "{clip_path}"),')

    lines.append(")")
    lines.append("")
    lines.append("# Brand assets")
    lines.append(f'$titleCard = "{BRAND_DIR / "title_ribbon_1080x1920.png"}"')
    lines.append(f'$endCard = "{BRAND_DIR / "end_card_1080x1920.png"}"')
    lines.append(f'$watermark = "{BRAND_DIR / "watermark_corner_256_transparent.png"}"')
    lines.append("")
    lines.append("# Build concat list")
    lines.append('$concatFile = New-TemporaryFile')
    lines.append('$entries = @()')
    lines.append("")
    lines.append("foreach ($seg in $segments) {")
    lines.append("  $slate = $seg[0]")
    lines.append("  $clip = $seg[1]")
    lines.append("  if (Test-Path $slate) { $entries += \"file '$slate'\" }")
    lines.append("  if (Test-Path $clip) { $entries += \"file '$clip'\" }")
    lines.append("}")
    lines.append("")
    lines.append("$entries | Out-File -FilePath $concatFile.FullName -Encoding ascii")
    lines.append("")
    lines.append(f'$output = Join-Path $PSScriptRoot "{output_path.name}"')
    lines.append('$outputDir = Split-Path -Parent $output')
    lines.append('if (!(Test-Path $outputDir)) { New-Item -ItemType Directory -Force $outputDir }')
    lines.append("")
    lines.append("# Simple concat assembly (upgrade to xfade for crossfades)")
    lines.append('ffmpeg -y -f concat -safe 0 -i $concatFile.FullName `')
    lines.append(f'  -vf "scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p" `')
    lines.append(f'  -c:v libx264 -crf $crf -preset medium -c:a aac -b:a 192k -ar 48000 `')
    lines.append('  -movflags +faststart $output')
    lines.append("")
    lines.append('Remove-Item $concatFile.FullName -ErrorAction SilentlyContinue')
    lines.append('Write-Host "Compilation complete: $output"')

    script_content = "\n".join(lines)
    return script_content


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_plan(args):
    """Show the compilation plan without building anything."""
    config = load_config()
    clips = load_all_clips(
        atomic_index_path=Path(args.atomic_index),
        catalog_dir=Path(args.catalog_dir),
        legacy_csv_path=Path(args.legacy_csv) if args.legacy_csv else None,
    )
    if not clips:
        print("No clips found in any source.", file=sys.stderr)
        return 1
    games = group_by_game(clips, config)

    print(f"\n  {'═' * 60}")
    print(f"  TSC SEASON COMPILATION PLAN")
    print(f"  {'═' * 60}")
    print(f"  Games: {len(games)}  |  Total clips: {sum(g.total_clips for g in games)}")
    print(f"  {'─' * 60}\n")

    for i, game in enumerate(games, 1):
        away_name = opponent_display_name(game.away, config)
        date_fmt = format_game_date(game.date)
        print(f"  Game {i}: TSC vs {away_name}")
        print(f"          {date_fmt}")
        print(f"          Clips available: {game.total_clips}")
        if game.best_clip:
            bc = game.best_clip
            print(f"          ★ Best: [{bc.idx}] {bc.label}  "
                  f"(score={bc.social_score:.1f}, {bc.duration:.1f}s)")
            # Show top 3 runners-up
            for j, rc in enumerate(game.all_clips[1:4], 2):
                print(f"            {j}. [{rc.idx}] {rc.label}  "
                      f"(score={rc.social_score:.1f}, {rc.duration:.1f}s)")
        print()

    est_duration = sum(
        (g.best_clip.duration if g.best_clip else 0) + 2.5  # clip + slate
        for g in games
    ) + 6.0  # title + end card
    print(f"  Estimated compilation: ~{est_duration:.0f}s ({est_duration/60:.1f} min)")
    print(f"  {'═' * 60}\n")

    return 0


def cmd_build(args):
    """Build the compilation manifest and generate FFmpeg slates + build script."""
    portrait_root = Path(args.portrait_root) if args.portrait_root else None

    config = load_config()
    clips = load_all_clips(
        atomic_index_path=Path(args.atomic_index),
        catalog_dir=Path(args.catalog_dir),
        legacy_csv_path=Path(args.legacy_csv) if args.legacy_csv else None,
    )
    if not clips:
        print("No clips found in any source.", file=sys.stderr)
        return 1
    games = group_by_game(clips, config)
    slates = build_slate_specs(games, config)

    manifest = build_manifest(games, slates, config, portrait_root)

    # Save manifest
    out_dir = ROOT / "out" / "compilation"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "season_compilation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    print(f"  Manifest: {manifest_path}")

    # Generate FFmpeg slates
    slate_dir = out_dir / "slates"
    slate_dir.mkdir(parents=True, exist_ok=True)
    for slate in slates:
        slate_path = slate_dir / f"{slate.game.game_label}__slate.mp4"
        try:
            generate_ffmpeg_slate(slate, slate_path)
            print(f"  Slate: {slate_path.name}")
        except Exception as e:
            print(f"  [warn] Slate generation failed for {slate.game.game_label}: {e}")

    # Generate build script
    output_video = out_dir / "TSC_Season_Highlights_2025-26.mp4"
    script = build_ffmpeg_compilation(manifest, output_video, slate_dir)
    script_path = out_dir / "Build-SeasonCompilation.ps1"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Build script: {script_path}")
    print(f"\n  Run the build script on your PC to assemble the final video.")

    return 0


def cmd_canva_slates(args):
    """Generate Canva design prompts for premium transition slates."""
    config = load_config()
    clips = load_all_clips(
        atomic_index_path=Path(args.atomic_index),
        catalog_dir=Path(args.catalog_dir),
        legacy_csv_path=Path(args.legacy_csv) if args.legacy_csv else None,
    )
    if not clips:
        print("No clips found in any source.", file=sys.stderr)
        return 1
    games = group_by_game(clips, config)
    slates = build_slate_specs(games, config)

    prompts = build_canva_slate_prompts(slates)

    out_dir = ROOT / "out" / "compilation"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = out_dir / "canva_slate_prompts.json"
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f"\n  CANVA SLATE PROMPTS ({len(prompts)} slates):\n")
    for p in prompts:
        print(f"  {p['index']}. {p['game_label']}")
        print(f"     {p['slate_data']['line1']} — {p['slate_data']['line2']}")
        print()

    print(f"  Saved: {prompts_path}")
    print(f"  Use these prompts with Canva:generate-design (design_type=your_story)")

    return 0


# ---------------------------------------------------------------------------
# Burst extraction: 3–5 s peak-moment windows from atomic clips
# ---------------------------------------------------------------------------

# Heuristic: where does the peak action live within a clip, by label type?
# Values are (peak_position_ratio, burst_duration_seconds).
#   peak_position_ratio: 0.0 = start of clip, 1.0 = end of clip
#   burst_duration: how long the extracted burst should be

BURST_PROFILES: Dict[str, Tuple[float, float]] = {
    # Goals — approach, strike, and net; peak shifted earlier to catch contact
    "GOAL":              (0.72, 5.0),
    # Shots — approach, strike, and reaction; match goal timing
    "SHOT":              (0.72, 5.0),
    # Build-up ending in goal — match GOAL timing
    "BUILD_UP_GOAL":     (0.72, 5.0),
    "BUILD_GOAL":        (0.72, 5.0),
    "PRESSURE_GOAL":     (0.72, 5.0),
    "CROSS_GOAL":        (0.72, 5.0),
    "CORNER_GOAL":       (0.72, 5.0),
    "FREE_KICK_GOAL":    (0.72, 5.0),
    # Build-up ending in shot — match SHOT timing
    "BUILD_UP_SHOT":     (0.72, 5.0),
    "BUILD_SHOT":        (0.72, 5.0),
    "PRESSURE_SHOT":     (0.72, 5.0),
    "CROSS_SHOT":        (0.72, 5.0),
    # Dribbling / skill — the move itself is in the middle
    "DRIBBL":            (0.50, 4.0),
    "SKILL":             (0.50, 4.0),
    "COMBINATION":       (0.55, 4.0),
    # Crosses — the delivery moment
    "CROSS":             (0.68, 3.5),
    # Set pieces — delivery + immediate reaction
    "CORNER":            (0.72, 4.0),
    "FREE_KICK":         (0.72, 3.5),
    "FREE KICK":         (0.72, 3.5),
    # Saves — the save itself
    "SAVE":              (0.75, 4.0),
    # Defense — the challenge / tackle
    "DEFENSE":           (0.55, 3.5),
    "TACKLE":            (0.55, 3.5),
    "BLOCK":             (0.60, 3.5),
    "INTERCEPT":         (0.50, 3.5),
    # Pressure — the squeeze play
    "PRESSURE":          (0.65, 3.5),
    # Build-up only — the key pass
    "BUILD":             (0.65, 4.0),
    # Generic
    "HIGHLIGHT":         (0.60, 4.0),
}

# Fallback for unmatched labels
BURST_DEFAULT = (0.60, 4.0)


def _resolve_burst_profile(label: str) -> Tuple[float, float]:
    """Match a clip label to its burst profile (peak_ratio, burst_duration).

    Tries compound labels first (most specific), then individual tokens.
    E.g. "Build, Cross & Goal" matches CROSS_GOAL before CROSS or BUILD.
    """
    upper = label.upper()
    # Normalize separators for matching
    norm = re.sub(r"[,&\-]+", "_", upper)
    norm = re.sub(r"\s+", "_", norm)
    norm = re.sub(r"_+", "_", norm).strip("_")

    # 1. Try exact normalized match
    if norm in BURST_PROFILES:
        return BURST_PROFILES[norm]

    # 2. Try compound: check if label contains GOAL, SHOT, etc. (priority order)
    has_goal = "GOAL" in upper
    has_shot = "SHOT" in upper
    has_save = "SAVE" in upper

    if has_goal:
        for prefix in ("BUILD", "PRESSURE", "CROSS", "CORNER", "FREE_KICK", "FREE KICK"):
            if prefix.replace("_", " ") in upper or prefix in upper:
                key = prefix.replace(" ", "_") + "_GOAL"
                if key in BURST_PROFILES:
                    return BURST_PROFILES[key]
        return BURST_PROFILES["GOAL"]

    if has_shot:
        for prefix in ("BUILD", "PRESSURE", "CROSS"):
            if prefix in upper:
                key = prefix + "_SHOT"
                if key in BURST_PROFILES:
                    return BURST_PROFILES[key]
        return BURST_PROFILES["SHOT"]

    if has_save:
        return BURST_PROFILES["SAVE"]

    # 3. Single-token fallback
    for token in ("DRIBBL", "SKILL", "COMBINATION", "CROSS", "CORNER",
                   "FREE_KICK", "FREE KICK", "DEFENSE", "TACKLE", "BLOCK",
                   "INTERCEPT", "PRESSURE", "BUILD", "HIGHLIGHT"):
        if token in upper:
            return BURST_PROFILES[token]

    return BURST_DEFAULT


@dataclass
class BurstSpec:
    """A trimmed burst window from an atomic clip."""
    clip: ClipRecord
    game_label: str
    burst_start: float        # seconds into the atomic clip (relative)
    burst_end: float          # seconds into the atomic clip (relative)
    burst_duration: float
    peak_ratio: float         # where we estimated the peak
    profile_label: str        # which heuristic profile matched
    master_burst_start: float # absolute time in the master video
    master_burst_end: float   # absolute time in the master video

    @property
    def clip_path(self) -> str:
        return self.clip.clip_folder_path or ""

    @property
    def fps(self) -> float:
        return self.clip.fps

    @property
    def clip_stem(self) -> str:
        """Clip stem (filename without extension) for portrait reel matching."""
        p = self.clip.clip_folder_path
        if p:
            from pathlib import PureWindowsPath
            return PureWindowsPath(p).stem
        return ""


def compute_burst_window(clip: ClipRecord) -> BurstSpec:
    """Compute the optimal 3-5 second burst window for a clip."""
    peak_ratio, burst_dur = _resolve_burst_profile(clip.label)

    clip_dur = clip.duration
    if clip_dur <= 0:
        clip_dur = clip.t_end - clip.t_start if clip.t_end > clip.t_start else 5.0

    # Clamp burst duration to clip duration
    burst_dur = min(burst_dur, clip_dur)

    # Minimum burst: 2 seconds
    burst_dur = max(burst_dur, min(2.0, clip_dur))

    # Center the burst window around the estimated peak
    peak_time = clip_dur * peak_ratio
    half = burst_dur / 2.0
    burst_start = peak_time - half
    burst_end = peak_time + half

    # Clamp to clip boundaries
    if burst_start < 0:
        burst_start = 0.0
        burst_end = min(burst_dur, clip_dur)
    if burst_end > clip_dur:
        burst_end = clip_dur
        burst_start = max(0.0, clip_dur - burst_dur)

    # Absolute master times
    master_burst_start = clip.t_start + burst_start
    master_burst_end = clip.t_start + burst_end

    game_label = f"{clip.date}__{clip.home}_vs_{clip.away}"

    return BurstSpec(
        clip=clip,
        game_label=game_label,
        burst_start=round(burst_start, 2),
        burst_end=round(burst_end, 2),
        burst_duration=round(burst_end - burst_start, 2),
        peak_ratio=peak_ratio,
        profile_label=clip.label,
        master_burst_start=round(master_burst_start, 2),
        master_burst_end=round(master_burst_end, 2),
    )


def compute_all_bursts(
    games: List[GameSummary],
    min_score: float = 0.0,
) -> List[Tuple[GameSummary, List[BurstSpec]]]:
    """Compute burst windows for every clip in every game.

    Returns: list of (game, [bursts]) sorted by game date then clip score.
    """
    results = []
    for game in games:
        bursts = []
        ranked = sorted(game.all_clips, key=lambda c: (-c.social_score, -c.priority))
        for clip in ranked:
            if clip.social_score < min_score:
                continue
            burst = compute_burst_window(clip)
            bursts.append(burst)
        results.append((game, bursts))
    return results


def build_burst_montage_script(
    game_bursts: List[Tuple[GameSummary, List[BurstSpec]]],
    config: dict,
    output_path: Path,
    slate_dir: Path,
    portrait_root: str = r"D:\Projects\soccer-video\out\portrait_reels",
    slate_duration: float = 1.5,
    out_w: int = 1080,
    out_h: int = 1920,
    crf: int = 18,
) -> str:
    """Generate a PowerShell script to extract bursts and assemble the montage.

    Two-pass approach:
      Pass 1: FFmpeg extracts each burst from its portrait reel into a temp file
      Pass 2: Concat all burst clips + slates into the final montage

    Clip source priority: portrait reel > atomic clip (landscape fallback).
    Uses each clip's native frame rate instead of a global fps.
    """
    lines = [
        "# ═══════════════════════════════════════════════════════════",
        "# TSC Season Burst Montage — Build Script",
        f"# Generated: {datetime.now().isoformat(timespec='seconds')}",
        "# ═══════════════════════════════════════════════════════════",
        "",
        '$ErrorActionPreference = "Stop"',
        "",
        f"$outW = {out_w}",
        f"$outH = {out_h}",
        f"$crf = {crf}",
        "",
        "# Portrait reel root (preferred source — polished 1080x1920)",
        f'$portraitRoot = "{portrait_root}"',
        "",
        "# Working directory for extracted bursts",
        '$burstDir = Join-Path $PSScriptRoot "burst_clips"',
        'if (!(Test-Path $burstDir)) { New-Item -ItemType Directory -Force $burstDir | Out-Null }',
        "",
        '$concatEntries = @()',
        '$extractCount = 0',
        '$skipCount = 0',
        "",
    ]

    total_bursts = 0

    for game, bursts in game_bursts:
        if not bursts:
            continue

        away_name = opponent_display_name(game.away, config)
        game_date = format_game_date(game.date)

        lines.append(f"# ─── {game_date}: TSC vs {away_name} ({len(bursts)} bursts) ───")
        lines.append("")

        # Slate for this game
        slate_file = f"slates\\{game.game_label}__slate.mp4"
        lines.append(f'$slateFile = Join-Path $PSScriptRoot "{slate_file}"')
        lines.append('if (Test-Path $slateFile) { $concatEntries += "file \'$slateFile\'" }')
        lines.append("")

        for i, burst in enumerate(bursts, 1):
            total_bursts += 1

            # Atomic clip path (landscape fallback)
            atomic_path = burst.clip_path
            if not atomic_path:
                atomic_path = (
                    f"D:\\Projects\\soccer-video\\out\\atomic_clips\\"
                    f"{burst.game_label}\\{burst.clip.clip_folder}"
                )

            safe_name = re.sub(r'[^\w\-.]', '_', f"{game.game_label}__{burst.clip.idx}")
            burst_file = f"$burstDir\\{safe_name}__burst.mp4"

            # Portrait reel search — tiered, newest-first
            # 1. Game subfolder (most specific)
            # 2. clean/ subfolder
            # 3. Recursive fallback
            # Always picks newest file to avoid stale renders
            clip_stem = burst.clip_stem
            clip_fps = burst.fps
            portrait_filter = f"{clip_stem}__portrait__FINAL*.mp4"

            lines.append(f"# Clip {burst.clip.idx}: {burst.clip.label} "
                         f"(score={burst.clip.social_score:.1f}, "
                         f"burst={burst.burst_duration:.1f}s @ {burst.burst_start:.1f}s, "
                         f"fps={clip_fps:.3f})")
            lines.append(f'$burstOut = "{burst_file}"')
            lines.append(f'$srcClip = $null')
            lines.append(f'$isPortrait = $false')
            lines.append(f'# Tiered portrait search: game subfolder > clean/ > recursive')
            lines.append(f'$gameDir = Join-Path $portraitRoot "{burst.game_label}"')
            lines.append(f'$cleanDir = Join-Path $portraitRoot "clean"')
            lines.append(f'$pFilter = "{portrait_filter}"')
            lines.append(f'$portraitHits = @()')
            lines.append(f'if (Test-Path $gameDir) {{')
            lines.append(f'  $portraitHits = @(Get-ChildItem -Path $gameDir -Filter $pFilter -ErrorAction SilentlyContinue)')
            lines.append(f'}}')
            lines.append(f'if ($portraitHits.Count -eq 0 -and (Test-Path $cleanDir)) {{')
            lines.append(f'  $portraitHits = @(Get-ChildItem -Path $cleanDir -Filter $pFilter -ErrorAction SilentlyContinue)')
            lines.append(f'}}')
            lines.append(f'if ($portraitHits.Count -eq 0) {{')
            lines.append(f'  $portraitHits = @(Get-ChildItem -Path $portraitRoot -Recurse -Filter $pFilter -ErrorAction SilentlyContinue)')
            lines.append(f'}}')
            lines.append(f'if ($portraitHits.Count -gt 0) {{')
            lines.append(f'  # Pick newest render to avoid stale files')
            lines.append(f'  $newest = $portraitHits | Sort-Object LastWriteTime -Descending | Select-Object -First 1')
            lines.append(f'  $srcClip = $newest.FullName')
            lines.append(f'  $isPortrait = $true')
            lines.append(f'  if ($portraitHits.Count -gt 1) {{')
            lines.append(f'    Write-Host "  [{burst.clip.idx}] Found $($portraitHits.Count) portrait renders, using newest: $($newest.Name)"')
            lines.append(f'  }}')
            lines.append(f'}} else {{')
            lines.append(f'  $srcClip = "{atomic_path}"')
            lines.append(f'}}')
            lines.append(f'if (Test-Path $srcClip) {{')
            lines.append(f'  if ($isPortrait) {{')
            lines.append(f'    # Portrait reel: already 1080x1920, just trim at native fps')
            lines.append(f'    ffmpeg -hide_banner -loglevel warning -y `')
            lines.append(f'      -ss {burst.burst_start:.2f} -t {burst.burst_duration:.2f} `')
            lines.append(f'      -i $srcClip `')
            lines.append(f'      -r {clip_fps:.3f} `')
            lines.append(f'      -c:v libx264 -crf $crf -preset fast -an `')
            lines.append(f'      $burstOut')
            lines.append(f'  }} else {{')
            lines.append(f'    # Landscape fallback: scale + letterbox to portrait')
            lines.append(f'    Write-Warning "Using landscape fallback for clip {burst.clip.idx}"')
            lines.append(f'    ffmpeg -hide_banner -loglevel warning -y `')
            lines.append(f'      -ss {burst.burst_start:.2f} -t {burst.burst_duration:.2f} `')
            lines.append(f'      -i $srcClip `')
            lines.append(f'      -vf "scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,'
                         f'pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1" `')
            lines.append(f'      -r {clip_fps:.3f} `')
            lines.append(f'      -c:v libx264 -crf $crf -preset fast -an `')
            lines.append(f'      $burstOut')
            lines.append(f'  }}')
            lines.append(f'  $concatEntries += "file \'$burstOut\'"')
            lines.append(f'  $extractCount++')
            lines.append(f'}} else {{')
            lines.append(f'  Write-Warning "Missing: $srcClip"')
            lines.append(f'  $skipCount++')
            lines.append(f'}}')
            lines.append("")

    # Pass 2: concat
    lines.extend([
        "# ═══════════════════════════════════════════════════════════",
        "# PASS 2: Assemble all bursts + slates into final montage",
        "# ═══════════════════════════════════════════════════════════",
        "",
        '$concatFile = Join-Path $PSScriptRoot "burst_concat_list.txt"',
        '$concatEntries | Out-File -FilePath $concatFile -Encoding ascii',
        "",
        f'$output = Join-Path $PSScriptRoot "{output_path.name}"',
        '$outputDir = Split-Path -Parent $output',
        'if (!(Test-Path $outputDir)) { New-Item -ItemType Directory -Force $outputDir | Out-Null }',
        "",
        'Write-Host ""',
        'Write-Host "Assembling $extractCount bursts ($skipCount skipped)..."',
        "",
        'ffmpeg -y -f concat -safe 0 -i $concatFile `',
        f'  -c:v libx264 -crf $crf -preset medium `',
        f'  -c:a aac -b:a 128k -ar 48000 `',
        '  -movflags +faststart `',
        '  $output',
        "",
        'Write-Host ""',
        'Write-Host "Done! Montage: $output"',
        'Write-Host "  Bursts extracted: $extractCount"',
        'Write-Host "  Skipped (missing): $skipCount"',
        "",
        "# Cleanup burst clips (uncomment to keep them)",
        '# Remove-Item $burstDir -Recurse -Force',
        '# Remove-Item $concatFile -Force',
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Burst CLI commands
# ---------------------------------------------------------------------------

def cmd_burst_plan(args):
    """Preview the burst montage plan — shows every clip with its trim window."""
    config = load_config()
    clips = load_all_clips(
        atomic_index_path=Path(args.atomic_index),
        catalog_dir=Path(args.catalog_dir),
        legacy_csv_path=Path(args.legacy_csv) if args.legacy_csv else None,
    )
    if not clips:
        print("No clips found.", file=sys.stderr)
        return 1

    games = group_by_game(clips, config)

    # Filter to a single game if requested
    if args.game:
        filt = args.game.upper().replace(" ", "_").replace("-", "")
        games = [g for g in games if filt in g.game_label.upper().replace("-", "")]
        if not games:
            print(f"No game matching '{args.game}'. Available:", file=sys.stderr)
            all_games = group_by_game(clips, config)
            for g in all_games:
                print(f"  {g.game_label}", file=sys.stderr)
            return 1

    game_bursts = compute_all_bursts(games, min_score=float(args.min_score))

    total_bursts = 0
    total_burst_dur = 0.0
    total_orig_dur = 0.0

    print(f"\n  {'═' * 72}")
    print(f"  TSC SEASON BURST MONTAGE PLAN")
    print(f"  {'═' * 72}")

    for game, bursts in game_bursts:
        if not bursts:
            continue
        away_name = opponent_display_name(game.away, config)
        date_fmt = format_game_date(game.date)

        print(f"\n  ┌─ {date_fmt}: TSC vs {away_name} ({len(bursts)} bursts)")
        print(f"  │  {'Clip':>4}  {'Label':<28} {'Score':>5}  "
              f"{'Orig':>6}  {'Burst':>6}  {'Window'}")
        print(f"  │  {'─'*4}  {'─'*28} {'─'*5}  {'─'*6}  {'─'*6}  {'─'*16}")

        for burst in bursts:
            c = burst.clip
            total_bursts += 1
            total_burst_dur += burst.burst_duration
            total_orig_dur += c.duration

            label_trunc = c.label[:28]
            print(f"  │  {c.idx:>4}  {label_trunc:<28} {c.social_score:>5.1f}  "
                  f"{c.duration:>5.1f}s  {burst.burst_duration:>5.1f}s  "
                  f"{burst.burst_start:.1f}–{burst.burst_end:.1f}s")

        print(f"  └─")

    n_games = sum(1 for _, b in game_bursts if b)
    slate_dur = n_games * 1.5
    est_total = total_burst_dur + slate_dur

    print(f"\n  {'═' * 72}")
    print(f"  📊 MONTAGE SUMMARY")
    print(f"  {'─' * 72}")
    print(f"  Games:                {n_games}")
    print(f"  Total bursts:         {total_bursts}")
    print(f"  Original clip time:   {total_orig_dur:.0f}s ({total_orig_dur/60:.1f} min)")
    print(f"  Burst time:           {total_burst_dur:.0f}s ({total_burst_dur/60:.1f} min)")
    print(f"  Compression ratio:    {total_orig_dur/total_burst_dur:.1f}× "
          f"(trimmed to {total_burst_dur/total_orig_dur*100:.0f}%)")
    print(f"  + Transition slates:  {slate_dur:.0f}s ({n_games} × 1.5s)")
    print(f"  ─────────────────────────────────")
    print(f"  Est. montage length:  {est_total:.0f}s ({est_total/60:.1f} min)")
    print(f"  {'═' * 72}\n")

    return 0


def cmd_burst_build(args):
    """Generate burst extraction + assembly scripts."""
    config = load_config()
    clips = load_all_clips(
        atomic_index_path=Path(args.atomic_index),
        catalog_dir=Path(args.catalog_dir),
        legacy_csv_path=Path(args.legacy_csv) if args.legacy_csv else None,
    )
    if not clips:
        print("No clips found.", file=sys.stderr)
        return 1

    games = group_by_game(clips, config)

    # Filter to a single game if requested
    if args.game:
        filt = args.game.upper().replace(" ", "_").replace("-", "")
        games = [g for g in games if filt in g.game_label.upper().replace("-", "")]
        if not games:
            print(f"No game matching '{args.game}'.", file=sys.stderr)
            return 1

    game_bursts = compute_all_bursts(games, min_score=float(args.min_score))
    slates = build_slate_specs(games, config)

    out_dir = ROOT / "out" / "compilation"
    out_dir.mkdir(parents=True, exist_ok=True)
    slate_dir = out_dir / "slates"
    slate_dir.mkdir(parents=True, exist_ok=True)

    # Generate FFmpeg slates (shorter 1.5s for montage pace)
    for slate in slates:
        slate.duration = 1.5
        slate_path = slate_dir / f"{slate.game.game_label}__slate.mp4"
        try:
            generate_ffmpeg_slate(slate, slate_path)
        except Exception as e:
            print(f"  [warn] Slate failed: {slate.game.game_label}: {e}")

    # Save burst manifest
    manifest_data = {
        "title": "TSC 2025-26 Season Burst Montage",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "burst_duration_range": "3.0-4.5s",
        "method": "heuristic_label_based_v1",
        "games": [],
    }
    total_bursts = 0
    for game, bursts in game_bursts:
        away_name = opponent_display_name(game.away, config)
        game_data = {
            "game_label": game.game_label,
            "opponent": away_name,
            "date": game.date,
            "total_clips": game.total_clips,
            "bursts": [],
        }
        for b in bursts:
            total_bursts += 1
            game_data["bursts"].append({
                "clip_idx": b.clip.idx,
                "label": b.clip.label,
                "social_score": b.clip.social_score,
                "original_duration": b.clip.duration,
                "burst_start": b.burst_start,
                "burst_end": b.burst_end,
                "burst_duration": b.burst_duration,
                "peak_ratio": b.peak_ratio,
                "clip_path": b.clip_path,
                "master_burst_start": b.master_burst_start,
                "master_burst_end": b.master_burst_end,
            })
        manifest_data["games"].append(game_data)

    manifest_data["total_bursts"] = total_bursts

    manifest_path = out_dir / "burst_montage_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"  Manifest: {manifest_path}")

    # Generate build script
    portrait_root = getattr(args, 'portrait_root', r"D:\Projects\soccer-video\out\portrait_reels")
    output_video = out_dir / "TSC_Season_BurstMontage_2025-26.mp4"
    script = build_burst_montage_script(
        game_bursts, config, output_video, slate_dir,
        portrait_root=portrait_root,
    )
    script_path = out_dir / "Build-BurstMontage.ps1"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Build script: {script_path}")
    print(f"  Total bursts: {total_bursts}")
    print(f"\n  Run Build-BurstMontage.ps1 on your PC to assemble the montage.")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Season compilation: best of each game with branded transitions",
    )
    sub = parser.add_subparsers(dest="command")

    # Shared arguments for all subcommands
    def _add_source_args(p):
        p.add_argument("--atomic-index", type=str,
                        default=str(ROOT / "out" / "catalog" / "atomic_index.csv"),
                        help="Path to atomic_index.csv (canonical clip catalog)")
        p.add_argument("--catalog-dir", type=str,
                        default=str(ROOT / "out" / "catalog"),
                        help="Path to catalog dir containing per-game events_selected.csv")
        p.add_argument("--legacy-csv", type=str, default=None,
                        help="Path to AtomicClips.All.csv (fallback only)")

    p_plan = sub.add_parser("plan", help="Preview the compilation plan")
    _add_source_args(p_plan)

    p_build = sub.add_parser("build", help="Generate slates + build script")
    _add_source_args(p_build)
    p_build.add_argument("--portrait-root", type=str, default=None,
                         help="Root of portrait_reels directory tree")

    p_canva = sub.add_parser("canva-slates", help="Generate Canva slate prompts")
    _add_source_args(p_canva)

    p_burst_plan = sub.add_parser("burst-plan",
                                   help="Preview the burst montage plan")
    _add_source_args(p_burst_plan)
    p_burst_plan.add_argument("--min-score", type=float, default=0.0,
                               help="Minimum social_score to include a clip")
    p_burst_plan.add_argument("--game", type=str, default=None,
                               help="Filter to a single game (fuzzy match, e.g. 'NEOFC' or '2026-02-23__TSC_vs_NEOFC')")
    p_burst_plan.add_argument("--portrait-root", type=str,
                               default=r"D:\Projects\soccer-video\out\portrait_reels",
                               help="Root of portrait_reels directory tree")

    p_burst_build = sub.add_parser("burst-build",
                                    help="Generate burst extraction + assembly script")
    _add_source_args(p_burst_build)
    p_burst_build.add_argument("--min-score", type=float, default=0.0,
                                help="Minimum social_score to include a clip")
    p_burst_build.add_argument("--game", type=str, default=None,
                                help="Filter to a single game (fuzzy match, e.g. 'NEOFC' or '2026-02-23__TSC_vs_NEOFC')")
    p_burst_build.add_argument("--portrait-root", type=str,
                                default=r"D:\Projects\soccer-video\out\portrait_reels",
                                help="Root of portrait_reels directory tree")

    args = parser.parse_args()
    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "build":
        return cmd_build(args)
    elif args.command == "canva-slates":
        return cmd_canva_slates(args)
    elif args.command == "burst-plan":
        return cmd_burst_plan(args)
    elif args.command == "burst-build":
        return cmd_burst_build(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
