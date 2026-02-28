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
import re
import sys
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


def resolve_player(player_name: str, roster: dict[str, dict]) -> dict | None:
    """Look up full player info from roster."""
    key = player_name.strip().lower()
    entry = roster.get(key)
    if not entry:
        first = key.split()[0]
        entry = roster.get(first)
    return entry


def resolve_handle(player_name: str, roster: dict[str, dict]) -> str | None:
    """Look up Instagram handle for a player name."""
    entry = resolve_player(player_name, roster)
    if entry:
        handle = entry.get("InstagramHandle", "").strip()
        return handle if handle else None
    return None


def expand_player_refs(narrative: str, roster: dict[str, dict]) -> str:
    """Expand {PlayerName} placeholders to (No. number - FirstName) format.

    Uses 'No.' instead of '#' to avoid Instagram counting them as hashtags.
    E.g. '{Charlotte Robison}' -> '(No. 13 - Charlotte)'
    """
    def replace_match(m: re.Match) -> str:
        name = m.group(1)
        entry = resolve_player(name, roster)
        if entry:
            number = entry.get("PlayerNumber", "").strip()
            first_name = entry["PlayerName"].strip().split()[0]
            if number:
                return f"(No. {number} - {first_name})"
            return f"({first_name})"
        return f"({name})"

    return re.sub(r"\{([^}]+)\}", replace_match, narrative)


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


def pick_hashtags(config: dict, action_types: list[str]) -> list[str]:
    """Select hashtags from config based on action types. Instagram max = 5."""
    max_tags = config.get("posting", {}).get("max_hashtags", 5)
    ht = config["hashtags"]

    # Start with the always tags (up to 3)
    tags = list(ht["always"][:3])

    # Add action-specific tags to fill remaining slots
    action_map = ht.get("action", {})
    for action in action_types:
        for t in action_map.get(action, []):
            if t not in tags and len(tags) < max_tags:
                tags.append(t)

    # If still under limit, fill from broad
    for t in ht.get("broad", []):
        if t not in tags and len(tags) < max_tags:
            tags.append(t)

    return tags[:max_tags]


def build_tag_line(players_involved: list[dict], config: dict, roster: dict[str, dict]) -> str:
    """Build the @-mention line for tagging."""
    mentions = list(config.get("always_tag", []))
    for p in players_involved:
        handle = resolve_handle(p["name"], roster)
        if handle and handle not in mentions:
            mentions.append(handle)
    return " ".join(mentions)


def generate_post(note_path: Path, dry_run: bool = False) -> Path | None:
    """Generate an Instagram post from a highlight annotation file."""
    config = load_config()
    roster = load_roster()

    with open(note_path, encoding="utf-8") as f:
        note = json.load(f)

    game_info = parse_game_label(note["game_label"])

    # Use event_date if provided, otherwise fall back to game_label date
    date_str = note.get("event_date", game_info["date"])
    date_fmt = format_date(date_str)

    # Team names and colors
    home = game_info["home"]
    opponent_key = game_info["away"]
    opponent = config.get("opponents", {}).get(opponent_key, {})
    opponent_name = opponent.get("name", opponent_key)
    opponent_ig = opponent.get("instagram", "")

    home_color = note.get("home_color", "")
    away_color = note.get("away_color", "")
    matchup = f"{config['club']['short_name']}"
    if home_color:
        matchup += f" {home_color}"
    matchup += f" vs {opponent_name}"
    if away_color:
        matchup += f" {away_color}"

    # Event/tournament name
    event_name = note.get("event_name", "")

    action_types = note.get("action_types", [])
    hashtags = pick_hashtags(config, action_types)

    # Hook line — include coach personal handle
    hook = note.get("caption_hook", note.get("playtag", "Highlight"))
    coach_personal = config.get("coach", {}).get("instagram_personal", "")
    if coach_personal and "Coach" in hook and f"({coach_personal})" not in hook:
        hook = hook.replace("Coach", f"Coach ({coach_personal})")

    # Narrative with player refs expanded to (#number - Name)
    narrative = note.get("narrative", "")
    narrative = expand_player_refs(narrative, roster)

    # Tag line
    tag_line = build_tag_line(note["players_involved"], config, roster)
    if opponent_ig:
        tag_line += f" {opponent_ig}"

    # Assemble caption
    lines = [hook, matchup]
    if event_name:
        lines.append(event_name)
    lines.append(date_fmt)
    lines.append(narrative)
    lines.append(tag_line)
    lines.append(" ".join(hashtags))

    caption = "\n".join(lines)

    # Build the full post output
    post = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_annotation": str(note_path),
        "video_file": note.get("video_file", ""),
        "platform": "instagram",
        "caption": caption,
        "tags_in_video": list(config.get("always_tag", [])),
        "hashtags": hashtags,
        "player_handles": [
            resolve_handle(p["name"], roster)
            for p in note["players_involved"]
            if resolve_handle(p["name"], roster)
        ],
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
