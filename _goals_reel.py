"""Goals-only highlight reel — chronological order."""
import subprocess, json, csv, shutil
from pathlib import Path

GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
BURST_DIR = Path(r"D:\Projects\soccer-video\out\bursts") / GAME
REEL_DIR = Path(r"D:\Projects\soccer-video\out\reels")
DESKTOP = Path(r"C:\Users\scott\Desktop")
CSV_PATH = BURST_DIR / "burst_overrides.csv"
REEL_DIR.mkdir(parents=True, exist_ok=True)

# Read CSV, filter to goals only, chronological order
rows = []
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        if "goal" in r["label"].lower():
            rows.append({
                "idx": int(r["clip"]),
                "label": r["label"],
                "burst_start": float(r["burst_start"]),
                "burst_end": float(r["burst_end"]),
            })
rows.sort(key=lambda c: c["idx"])
print(f"Found {len(rows)} goal clips (chronological):\n")
for i, r in enumerate(rows, 1):
    dur = r["burst_end"] - r["burst_start"]
    print(f"  {i}. [{r['idx']:02d}] {r['label']} "
          f"({r['burst_start']:.1f}-{r['burst_end']:.1f}s, {dur:.1f}s)")

# Use existing burst clips — already extracted with correct windows
# Normalize for concat
NORM_DIR = BURST_DIR / "normalized_goals"
NORM_DIR.mkdir(exist_ok=True)
norm_paths = []
print("\nNormalizing goal clips...")
for r in rows:
    src = BURST_DIR / f"burst_{r['idx']:02d}__{r['label'].replace(' ', '_')}.mp4"
    if not src.exists():
        print(f"  [{r['idx']:02d}] MISSING: {src.name}"); continue
    norm = NORM_DIR / f"norm_{r['idx']:02d}.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-i", str(src),
        "-vf", "fps=24,scale=1080:1920:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(norm),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [{r['idx']:02d}] ERROR: {res.stderr[-200:]}")
    else:
        norm_paths.append(norm)
        print(f"  [{r['idx']:02d}] OK")

# Concat list
concat_file = BURST_DIR / "concat_goals.txt"
with open(concat_file, "w") as f:
    for p in norm_paths:
        f.write(f"file '{p}'\n")

# Assemble
reel_name = f"{GAME}__goals_reel.mp4"
reel_path = REEL_DIR / reel_name
print(f"\nAssembling goals reel -> {reel_path}")
cmd = [
    "ffmpeg", "-hide_banner", "-y",
    "-f", "concat", "-safe", "0", "-i", str(concat_file),
    "-c:v", "libx264", "-preset", "slow", "-crf", "17",
    "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart", str(reel_path),
]
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode != 0:
    print(f"ERROR: {res.stderr[-500:]}")
else:
    dur = json.loads(subprocess.check_output(
        ["ffprobe","-v","error","-show_entries","format=duration",
         "-of","json",str(reel_path)],
        text=True, stderr=subprocess.DEVNULL
    ))["format"]["duration"]
    print(f"\nGoals reel complete! Duration: {float(dur):.1f}s")
    desktop_reel = DESKTOP / reel_name
    shutil.copy2(str(reel_path), str(desktop_reel))
    print(f"Copied to Desktop: {reel_name}")
    print(f"\n{'='*50}")
    print("DONE!")
    print(f"{'='*50}")
