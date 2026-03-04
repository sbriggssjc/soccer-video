"""Add Feb 23 NEOFC (all 34 clips) to atomic index."""
import csv, os, re
from pathlib import Path

INDEX = r"D:\Projects\soccer-video\AtomicClips.All.csv"
GAME_FOLDER = "2026-02-23__TSC_vs_NEOFC"
ATOMIC_DIR = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME_FOLDER}")
REEL_DIR = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME_FOLDER}")

GAME = {"folder": GAME_FOLDER, "date": "2026-02-23", "home": "TSC", "away": "NEOFC"}

# Filename: 009__2026-02-23__TSC_vs_NEOFC__CORNER__t850.00-t856.00.mp4
CLIP_RE = re.compile(r'^(\d{3})__\d{4}-\d{2}-\d{2}__\w+_vs_\w+__(.+?)__t([\d.]+)-t([\d.]+)\.mp4$')

FIELDNAMES = [
    "key", "idx", "date", "home", "away", "label",
    "t_start", "t_end", "clip_folder", "clip_folder_path",
    "proxy_exists", "transforms_exists", "stabilized_exists",
    "branded_exists", "branded_brand", "branded_post", "branded_paths",
    "proxy_path", "transforms_path", "stabilized_path"
]

def normalize_label(label):
    s = label.replace("&", "AND").replace(",", "")
    s = re.sub(r'\s+', '_', s.strip())
    return s.upper()

# Get FINAL renders
finals = set()
for f in REEL_DIR.glob("*__portrait__FINAL.mp4"):
    m = re.match(r'^(\d{3})__', f.name)
    if m:
        finals.add(m.group(1))
print(f"Found {len(finals)} FINAL renders")

# Read existing index
with open(INDEX, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    existing = list(reader)
existing_keys = {r["key"] for r in existing}
print(f"Existing index: {len(existing)} rows")

# Parse clips
new_rows = []
for f in sorted(ATOMIC_DIR.iterdir()):
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
    key = f"{idx}|{GAME['date']}|{GAME['home']}|{GAME['away']}|{label_norm}|t{t_start}|t{t_end}"

    if key in existing_keys:
        print(f"  SKIP {idx} (already in index)")
        continue

    has_final = idx in finals
    final_path = ""
    if has_final:
        final_files = list(REEL_DIR.glob(f"{idx}__*__portrait__FINAL.mp4"))
        if final_files:
            final_path = str(final_files[0])

    row = {
        "key": key, "idx": idx, "date": GAME["date"],
        "home": GAME["home"], "away": GAME["away"], "label": label_norm,
        "t_start": f"t{t_start}", "t_end": f"t{t_end}",
        "clip_folder": f.stem, "clip_folder_path": str(ATOMIC_DIR),
        "proxy_exists": "False", "transforms_exists": "False",
        "stabilized_exists": str(has_final), "branded_exists": "False",
        "branded_brand": "False", "branded_post": "False", "branded_paths": "",
        "proxy_path": str(f), "transforms_path": "", "stabilized_path": final_path,
    }
    new_rows.append(row)
    existing_keys.add(key)

if new_rows:
    all_rows = existing + new_rows
    with open(INDEX, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAdded {len(new_rows)} entries. Total: {len(all_rows)}")
else:
    print("\nNo new entries to add.")
