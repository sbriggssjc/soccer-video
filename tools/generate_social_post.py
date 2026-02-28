#!/usr/bin/env python3
"""Generate ready-to-post Instagram content for finalized portrait reels.

Usage:
    # Generate post for a specific highlight annotation:
    python tools/generate_social_post.py social/highlight_notes/2026-02-23__TSC_vs_NEOFC__002.json

    # Scan for all annotations that don't have posts yet:
    python tools/generate_social_post.py --scan

    # Dry-run (print to console, don't write file):
    python tools/generate_social_post.py social/highlight_notes/2026-02-23__TSC_vs_NEOFC__002.json --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOCIAL_DIR = ROOT / "social"
NOTES_DIR = SOCIAL_DIR / "highlight_notes"
POSTS_DIR = SOCIAL_DIR / "posts"
CONFIG_PATH = SOCIAL_DIR / "social_config.json"
ROSTER_PATH = ROOT / "roster.csv"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_roster() -> dict[str, dict]:
    """Return roster keyed by player name (case-insensitive first-name match too)."""
    players: dict[str, dict] = {}
    with open(ROSTER_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["PlayerName"].strip()
            players[name.lower()] = row
            # Also index by first name for partial matches
            first = name.split()[0].lower()
            if first not in players:
                players[first] = row
    return players


def resolve_handle(player_name: str, roster: dict[str, dict]) -> str | None:
    """Look up Instagram handle for a player name."""
    key = player_name.strip().lower()
    entry = roster.get(key)
    if not entry:
        # Try first name
        first = key.split()[0]
        entry = roster.get(first)
    if entry:
        handle = entry.get("InstagramHandle", "").strip()
        return handle if handle else None
    return None


def parse_game_label(label: str) -> dict:
    """Extract date, home team, away team from game_label like '2026-02-23__TSC_vs_NEOFC'."""
    parts = label.split("__")
    date_str = parts[0]
    matchup = parts[1] if len(parts) > 1 else ""
    teams = matchup.split("_vs_")
    return {
        "date": date_str,
        "home": teams[0] if teams else "",
        "away": teams[1] if len(teams) > 1 else "",
    }


def format_date(date_str: str) -> str:
    """Convert '2026-02-23' to '02.23.2026'."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%m.%d.%Y")
    except ValueError:
        return date_str


def pick_hashtags(config: dict, action_types: list[str], max_tags: int = 20) -> list[str]:
    """Select hashtags from config based on action types."""
    ht = config["hashtags"]
    tags = list(ht["always"])
    tags.extend(ht["broad"][:4])
    tags.extend(ht["regional"])

    # Add action-specific tags
    action_map = ht.get("action", {})
    for action in action_types:
        action_tags = action_map.get(action, [])
        for t in action_tags:
            if t not in tags:
                tags.append(t)

    return tags[:max_tags]


def build_player_line(players_involved: list[dict], roster: dict[str, dict]) -> str:
    """Build the narrative line with player names/handles."""
    parts = []
    for p in players_involved:
        name = p["name"]
        pos = p.get("position", "")
        action = p.get("action", "")
        handle = resolve_handle(name, roster)
        display = handle if handle else name
        pos_label = f" ({pos})" if pos else ""
        parts.append(f"{display}{pos_label} {action}")
    return "\n".join(f"  {line}" for line in parts)


def build_tag_line(players_involved: list[dict], config: dict, roster: dict[str, dict]) -> str:
    """Build the @-mention line for tagging."""
    mentions = list(config.get("always_tag", []))
    for p in players_involved:
        handle = resolve_handle(p["name"], roster)
        if handle and handle not in mentions:
            mentions.append(handle)

    # Add opponent if available
    return " ".join(mentions)


def generate_post(note_path: Path, dry_run: bool = False) -> Path | None:
    """Generate an Instagram post from a highlight annotation file."""
    config = load_config()
    roster = load_roster()

    with open(note_path, encoding="utf-8") as f:
        note = json.load(f)

    game_info = parse_game_label(note["game_label"])
    date_fmt = format_date(game_info["date"])
    opponent_key = game_info["away"]
    opponent = config.get("opponents", {}).get(opponent_key, {})
    opponent_name = opponent.get("name", opponent_key)
    opponent_ig = opponent.get("instagram", "")

    action_types = note.get("action_types", [])
    hashtags = pick_hashtags(config, action_types)

    # Build the caption
    hook = note.get("caption_hook", note.get("playtag", "Highlight"))
    narrative = note.get("narrative", "")

    # Player action breakdown
    player_lines = build_player_line(note["players_involved"], roster)

    # Tag line
    tag_line = build_tag_line(note["players_involved"], config, roster)

    caption = textwrap.dedent(f"""\
        {hook}

        {config["club"]["short_name"]} vs {opponent_name}
        {date_fmt}

        {narrative}

        {tag_line}

        {" ".join(hashtags)}""")

    # Build the full post output
    post = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_annotation": str(note_path),
        "video_file": note.get("video_file", ""),
        "platform": "instagram",
        "caption": caption,
        "tags_in_video": [t for t in config.get("always_tag", [])],
        "hashtags": hashtags,
        "player_handles": [
            resolve_handle(p["name"], roster)
            for p in note["players_involved"]
            if resolve_handle(p["name"], roster)
        ],
        "play_breakdown": player_lines,
    }

    if dry_run:
        print("=" * 60)
        print("INSTAGRAM POST — READY TO COPY/PASTE")
        print("=" * 60)
        print()
        print("VIDEO:", post["video_file"])
        print()
        print("--- CAPTION (copy below this line) ---")
        print(caption)
        print("--- END CAPTION ---")
        print()
        print("TAG IN VIDEO:", ", ".join(post["tags_in_video"]))
        print()
        print("PLAY BREAKDOWN:")
        print(player_lines)
        print()
        return None

    # Write post file
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = note_path.stem
    out_path = POSTS_DIR / f"{stem}__instagram.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(post, f, indent=2, ensure_ascii=False)

    # Also write a plain-text caption file for easy copy/paste
    txt_path = POSTS_DIR / f"{stem}__caption.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(caption)

    print(f"Post generated: {out_path}")
    print(f"Caption file:   {txt_path}")
    return out_path


def scan_for_new(dry_run: bool = False) -> list[Path]:
    """Find highlight annotations that don't have posts yet."""
    generated = []
    if not NOTES_DIR.exists():
        print("No highlight_notes directory found.")
        return generated

    for note_file in sorted(NOTES_DIR.glob("*.json")):
        post_file = POSTS_DIR / f"{note_file.stem}__instagram.json"
        if not post_file.exists():
            print(f"New annotation found: {note_file.name}")
            result = generate_post(note_file, dry_run=dry_run)
            if result:
                generated.append(result)
    if not generated and not dry_run:
        print("No new annotations to process.")
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate Instagram post content from highlight annotations"
    )
    parser.add_argument(
        "annotation",
        nargs="?",
        help="Path to a highlight annotation JSON file",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan for all annotations without posts and generate them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print post to console instead of writing files",
    )
    args = parser.parse_args()

    if args.scan:
        scan_for_new(dry_run=args.dry_run)
    elif args.annotation:
        path = Path(args.annotation)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        generate_post(path, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
