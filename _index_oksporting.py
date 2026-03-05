"""Add OK Sporting Nov 22 (4 clips) to atomic index and clean Desktop."""
import csv, os, re
from pathlib import Path

INDEX = r"D:\Projects\soccer-video\AtomicClips.All.csv"
GAME_FOLDER = "2025-11-22__TSC_vs_OK_Sporting"
ATOMIC_DIR = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME_FOLDER}")
REEL_DIR = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME_FOLDER}")
DESKTOP = Path(r"C:\Users\scott\Desktop")
PREFIX = "oks_"

GAME = {"date": "2025-11-22", "home": "TSC", "away": "OK_Sporting"}

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

finals = {}
for f in REEL_DIR.glob("*__portrait__FINAL.mp4"):
    m = re.match(r'^(\d{3})__', f.name)
    if m: finals[m.group(1)] = str(f)
print(f"Found {len(finals)} FINAL renders")

with open(INDEX, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    existing = list(reader)
existing_keys = {r["key"] for r in existing}
print(f"Existing index: {len(existing)} rows")

new_rows = []
for f in sorted(ATOMIC_DIR.iterdir()):
    if f.is_dir(): continue
    m = CLIP_RE.match(f.name)
    if not m: continue
    idx, label_raw, t_start, t_end = m.group(1), m.group(2), m.group(3), m.group(4)
    label_norm = normalize_label(label_raw)
    key = f"{idx}|{GAME['date']}|{GAME['home']}|{GAME['away']}|{label_norm}|t{t_start}|t{t_end}"
    if key in existing_keys: continue
    has_final = idx in finals
    row = {
        "key": key, "idx": idx, "date": GAME["date"],
        "home": GAME["home"], "away": GAME["away"], "label": label_norm,
        "t_start": f"t{t_start}", "t_end": f"t{t_end}",
        "clip_folder": f.stem, "clip_folder_path": str(ATOMIC_DIR),
        "proxy_exists": "False", "transforms_exists": "False",
        "stabilized_exists": str(has_final), "branded_exists": "False",
        "branded_brand": "False", "branded_post": "False", "branded_paths": "",
        "proxy_path": str(f), "transforms_path": "",
        "stabilized_path": finals.get(idx, ""),
    }
    new_rows.append(row)
    existing_keys.add(key)

if new_rows:
    all_rows = existing + new_rows
    with open(INDEX, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Added {len(new_rows)} entries. Total: {len(all_rows)}")
else:
    print("No new entries to add.")

removed = 0
for i in range(1, 5):
    for pat in [f"review_{PREFIX}{i:03d}.csv", f"filmstrip_{PREFIX}{i:03d}.jpg"]:
        fp = DESKTOP / pat
        if fp.exists():
            os.remove(fp); removed += 1; print(f"  Removed {pat}")
print(f"Cleaned {removed} files from Desktop")
