"""Add Feb 21 Greenwood to atomic index and clean Desktop CSVs."""
import csv, os, re
from pathlib import Path

INDEX = r"D:\Projects\soccer-video\AtomicClips.All.csv"
GAME_FOLDER = "2026-02-21__TSC_vs_Greenwood"
ATOMIC_DIR = Path(rf"D:\Projects\soccer-video\out\atomic_clips\{GAME_FOLDER}")
REEL_DIR = Path(rf"D:\Projects\soccer-video\out\portrait_reels\{GAME_FOLDER}")
DESKTOP = Path(r"C:\Users\scott\Desktop")

GAME = {"folder": GAME_FOLDER, "date": "2026-02-21", "home": "TSC", "away": "Greenwood"}

# Filename pattern with decimal timestamps: 001__Free Kick & Goal__551.43-567.97.mp4
CLIP_RE = re.compile(r'^(\d{3})__(.+?)__([\d.]+)-([\d.]+)\.mp4$')

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

# Read existing index
with open(INDEX, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    existing = list(reader)
existing_keys = {r["key"] for r in existing}

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
    print(f"Added {len(new_rows)} entries to index. Total: {len(all_rows)}")
else:
    print("No new entries to add.")

# Clean Desktop CSVs
removed = 0
for i in range(1, 21):
    for pat in [f"review_gw21_{i:03d}.csv", f"filmstrip_gw21_{i:03d}.jpg"]:
        f = DESKTOP / pat
        if f.exists():
            try:
                os.remove(f)
                removed += 1
                print(f"  Removed {pat}")
            except Exception as e:
                print(f"  FAILED {pat}: {e}")
print(f"\nCleaned {removed} files from Desktop")
