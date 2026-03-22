"""
Add new clips from March 1st games to AtomicClips.All.csv
Games: TSC vs North OKC, TSC vs OK Celtic
"""
import csv, os, re
from pathlib import Path

INDEX = r"D:\Projects\soccer-video\AtomicClips.All.csv"
ATOMIC_DIR = Path(r"D:\Projects\soccer-video\out\atomic_clips")
REEL_DIR = Path(r"D:\Projects\soccer-video\out\portrait_reels")

GAMES = [
    {"folder": "2026-03-01__TSC_vs_North_OKC", "date": "2026-03-01", "home": "TSC", "away": "North_OKC"},
    {"folder": "2026-03-01__TSC_vs_OK_Celtic",  "date": "2026-03-01", "home": "TSC", "away": "OK_Celtic"},
]

# Filename pattern: 001__Corner__69-79.mp4
CLIP_RE = re.compile(r'^(\d{3})__(.+?)__(\d+)-(\d+)\.mp4$')

FIELDNAMES = [
    "key", "idx", "date", "home", "away", "label",
    "t_start", "t_end", "clip_folder", "clip_folder_path",
    "proxy_exists", "transforms_exists", "stabilized_exists",
    "branded_exists", "branded_brand", "branded_post", "branded_paths",
    "proxy_path", "transforms_path", "stabilized_path"
]

def normalize_label(label):
    """Convert filename label to index format: uppercase, & -> AND, spaces -> _"""
    s = label.replace("&", "AND").replace(",", "")
    s = re.sub(r'\s+', '_', s.strip())
    return s.upper()

def parse_clips(game):
    clip_dir = ATOMIC_DIR / game["folder"]
    reel_dir = REEL_DIR / game["folder"]
    
    # Get set of clips that have FINAL renders
    finals = set()
    if reel_dir.exists():
        for f in reel_dir.iterdir():
            if f.name.endswith("__portrait__FINAL.mp4"):
                m = re.match(r'^(\d{3})__', f.name)
                if m:
                    finals.add(m.group(1))
    
    rows = []
    for f in sorted(clip_dir.iterdir()):
        if f.is_dir():
            continue
        m = CLIP_RE.match(f.name)
        if not m:
            continue
        
        idx = m.group(1)
        label_raw = m.group(2)
        t_start = m.group(3)
        t_end = m.group(4)
        
        label_norm = normalize_label(label_raw)
        key = f"{idx}|{game['date']}|{game['home']}|{game['away']}|{label_norm}|t{t_start}.00|t{t_end}.00"
        
        has_final = idx in finals
        clip_folder = f.stem  # filename without extension
        clip_folder_path = str(clip_dir / clip_folder)
        
        # Portrait FINAL path
        final_path = ""
        if has_final:
            final_name = f"{idx}__{label_raw}__{t_start}-{t_end}__portrait__FINAL.mp4"
            final_path = str(reel_dir / final_name)
        
        row = {
            "key": key,
            "idx": idx,
            "date": game["date"],
            "home": game["home"],
            "away": game["away"],
            "label": label_norm,
            "t_start": f"t{t_start}.00",
            "t_end": f"t{t_end}.00",
            "clip_folder": clip_folder,
            "clip_folder_path": str(clip_dir),
            "proxy_exists": "False",
            "transforms_exists": "False",
            "stabilized_exists": str(has_final),
            "branded_exists": "False",
            "branded_brand": "False",
            "branded_post": "False",
            "branded_paths": "",
            "proxy_path": str(f),
            "transforms_path": "",
            "stabilized_path": final_path,
        }
        rows.append(row)
    
    return rows

# Read existing index
with open(INDEX, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    existing = list(reader)

existing_keys = {r["key"] for r in existing}
print(f"Existing index: {len(existing)} rows, {len(existing_keys)} unique keys")

# Parse new clips
new_rows = []
for game in GAMES:
    clips = parse_clips(game)
    added = 0
    skipped = 0
    for row in clips:
        if row["key"] in existing_keys:
            skipped += 1
        else:
            new_rows.append(row)
            existing_keys.add(row["key"])
            added += 1
    print(f"{game['folder']}: {len(clips)} clips found, {added} new, {skipped} already in index")

if not new_rows:
    print("No new clips to add.")
else:
    # Append new rows
    all_rows = existing + new_rows
    with open(INDEX, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\nAdded {len(new_rows)} new rows. Total: {len(all_rows)} rows.")
    print("\nNew entries:")
    for r in new_rows:
        has_final = "FINAL" if r["stabilized_path"] else "no render"
        print(f"  {r['idx']} | {r['date']} | {r['away']:12s} | {r['label']:30s} | {r['t_start']}-{r['t_end']} | {has_final}")
