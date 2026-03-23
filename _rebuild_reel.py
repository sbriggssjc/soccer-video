"""Re-extract bursts from CSV overrides and rebuild highlight reel."""
import subprocess, json, csv, re, shutil
from pathlib import Path

GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
SRC_DIR = Path(r"D:\Projects\soccer-video\out\enhanced") / GAME
BURST_DIR = Path(r"D:\Projects\soccer-video\out\bursts") / GAME
REEL_DIR = Path(r"D:\Projects\soccer-video\out\reels")
DESKTOP = Path(r"C:\Users\scott\Desktop")
CSV_PATH = BURST_DIR / "burst_overrides.csv"

PRIORITY = [
    (5, ["GOAL"]), (4, ["SHOT"]),
    (3, ["CROSS", "SAVE", "FREE KICK", "CORNER"]),
    (2, ["BUILD", "DRIBBL", "PRESSURE", "THROUGH"]),
    (1, ["DEFENSE", "TACKLE", "BLOCK", "CLEAR"]),
]
def pri(label):
    u = label.upper()
    return max((s for s, toks in PRIORITY if any(t in u for t in toks)), default=0)

# Find source clip by index number
FNAME_RE = re.compile(r"^(\d+)__(.+?)__[\d.]+-[\d.]+(?:_portrait_FINAL)?\.mp4$", re.I)
def find_src(idx):
    for f in SRC_DIR.glob("*.mp4"):
        m = FNAME_RE.match(f.name)
        if m and int(m.group(1)) == idx:
            return f
    return None

# Read CSV overrides
rows = []
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        idx = int(r["clip"])
        bs = float(r["burst_start"])
        be = float(r["burst_end"])
        label = r["label"]
        rows.append({"idx": idx, "label": label, "burst_start": bs,
                      "burst_end": be, "burst_dur": round(be - bs, 3),
                      "pri": pri(label)})

print(f"Loaded {len(rows)} burst overrides from CSV\n")
print(f"{'#':>3} | {'Start':>6} | {'End':>6} | {'Dur':>5} | Label")
print("-" * 55)
total = 0
for r in rows:
    print(f"{r['idx']:3d} | {r['burst_start']:6.1f} | {r['burst_end']:6.1f} | "
          f"{r['burst_dur']:5.1f} | {r['label']}")
    total += r["burst_dur"]
print(f"\nTotal burst content: {total:.1f}s")

# Step 1: Re-extract all bursts
print(f"\n--- Re-extracting bursts ---\n")
for r in rows:
    src = find_src(r["idx"])
    if not src:
        print(f"  [{r['idx']:02d}] SOURCE NOT FOUND!"); continue
    out = BURST_DIR / f"burst_{r['idx']:02d}__{r['label'].replace(' ', '_')}.mp4"
    if out.exists():
        out.unlink()
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{r['burst_start']:.3f}",
        "-i", str(src),
        "-t", f"{r['burst_dur']:.3f}",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [{r['idx']:02d}] ERROR: {res.stderr[-200:]}")
    else:
        print(f"  [{r['idx']:02d}] OK ({r['burst_dur']:.1f}s)")
print("Burst extraction complete.")

# Step 2: Impact sandwich pacing
if len(rows) > 2:
    by_pri = sorted(rows, key=lambda c: (-c["pri"], c["idx"]))
    opener = by_pri[0]
    closer = by_pri[1]
    middle = [c for c in rows if c is not opener and c is not closer]
    middle.sort(key=lambda c: c["idx"])
    ordered = [opener] + middle + [closer]
else:
    ordered = rows

print(f"\n--- Pacing order ---\n")
for i, c in enumerate(ordered):
    tag = " << OPENER" if i == 0 else (" << CLOSER" if i == len(ordered)-1 else "")
    print(f"  {i+1:2d}. [{c['idx']:02d}] {c['label']} "
          f"(pri={c['pri']}, {c['burst_dur']:.1f}s){tag}")

# Step 3: Normalize for concat
NORM_DIR = BURST_DIR / "normalized"
NORM_DIR.mkdir(exist_ok=True)
print(f"\n--- Normalizing for concat ---\n")
norm_paths = []
for c in ordered:
    src = BURST_DIR / f"burst_{c['idx']:02d}__{c['label'].replace(' ', '_')}.mp4"
    norm = NORM_DIR / f"norm_{c['idx']:02d}.mp4"
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
        print(f"  [{c['idx']:02d}] NORM ERROR: {res.stderr[-200:]}")
    else:
        norm_paths.append(norm)
        print(f"  [{c['idx']:02d}] OK")

# Step 4: Concat into reel
REEL_DIR.mkdir(parents=True, exist_ok=True)
concat_file = BURST_DIR / "concat_list.txt"
with open(concat_file, "w") as f:
    for p in norm_paths:
        f.write(f"file '{p}'\n")

reel_path = REEL_DIR / f"{GAME}__highlight_reel.mp4"
print(f"\n--- Assembling reel -> {reel_path} ---\n")
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
    print(f"REEL ERROR: {res.stderr[-500:]}")
else:
    dur = json.loads(subprocess.check_output(
        ["ffprobe","-v","error","-show_entries","format=duration",
         "-of","json",str(reel_path)],
        text=True, stderr=subprocess.DEVNULL
    ))["format"]["duration"]
    print(f"Highlight reel complete! Duration: {float(dur):.1f}s")
    desktop_reel = DESKTOP / f"{GAME}__highlight_reel.mp4"
    shutil.copy2(str(reel_path), str(desktop_reel))
    print(f"Copied to Desktop: {desktop_reel.name}")
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
