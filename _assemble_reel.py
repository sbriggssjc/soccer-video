"""Assemble highlight reel from burst clips using ffmpeg concat demuxer.
Much simpler and more reliable than xfade filter chains for 19 clips.

Impact-sandwich pacing: best goal opens, second-best closes,
rest in chronological order.
"""
import subprocess, json, re, shutil
from pathlib import Path

GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
BURST_DIR = Path(r"D:\Projects\soccer-video\out\bursts") / GAME
REEL_DIR = Path(r"D:\Projects\soccer-video\out\reels")
DESKTOP = Path(r"C:\Users\scott\Desktop")
REEL_DIR.mkdir(parents=True, exist_ok=True)

# Priority rules for pacing
PRIORITY = [
    (5, ["GOAL"]), (4, ["SHOT"]),
    (3, ["CROSS", "SAVE", "FREE KICK", "CORNER"]),
    (2, ["BUILD", "DRIBBL", "PRESSURE", "THROUGH"]),
    (1, ["DEFENSE", "TACKLE", "BLOCK", "CLEAR"]),
]

def pri(label):
    u = label.upper()
    return max((s for s, toks in PRIORITY if any(t in u for t in toks)), default=0)

# Discover burst clips
RE = re.compile(r"^burst_(\d+)__(.+)\.mp4$")
bursts = []
for f in sorted(BURST_DIR.glob("burst_*.mp4")):
    m = RE.match(f.name)
    if m:
        idx = int(m.group(1))
        label = m.group(2).replace("_", " ").replace("&", "&")
        bursts.append({"idx": idx, "label": label, "path": f, "pri": pri(label)})
bursts.sort(key=lambda c: c["idx"])
print(f"Found {len(bursts)} burst clips")

# Impact sandwich pacing
if len(bursts) > 2:
    by_pri = sorted(bursts, key=lambda c: (-c["pri"], c["idx"]))
    opener = by_pri[0]
    closer = by_pri[1]
    middle = [c for c in bursts if c is not opener and c is not closer]
    middle.sort(key=lambda c: c["idx"])
    ordered = [opener] + middle + [closer]
else:
    ordered = bursts

print("\nPacing order:")
for i, c in enumerate(ordered):
    tag = " << OPENER" if i == 0 else (" << CLOSER" if i == len(ordered)-1 else "")
    print(f"  {i+1:2d}. [{c['idx']:02d}] {c['label']} (pri={c['pri']}){tag}")

# Step 1: Normalize all bursts to identical format (needed for concat)
NORM_DIR = BURST_DIR / "normalized"
NORM_DIR.mkdir(exist_ok=True)
print("\nNormalizing burst clips to consistent format...")
norm_paths = []
for c in ordered:
    norm_path = NORM_DIR / f"norm_{c['idx']:02d}.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-i", str(c["path"]),
        "-vf", "fps=24,scale=1080:1920:force_original_aspect_ratio=decrease,"
               "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(norm_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR normalizing {c['idx']:02d}: {r.stderr[-200:]}")
    else:
        norm_paths.append(norm_path)
        print(f"  [{c['idx']:02d}] OK")
print(f"Normalized {len(norm_paths)} clips")

# Step 2: Write concat file list
concat_file = BURST_DIR / "concat_list.txt"
with open(concat_file, "w") as f:
    for p in norm_paths:
        f.write(f"file '{p}'\n")
print(f"\nConcat list: {concat_file}")

# Step 3: Assemble reel with concat demuxer
reel_path = REEL_DIR / f"{GAME}__highlight_reel.mp4"
print(f"Assembling reel -> {reel_path}")
cmd = [
    "ffmpeg", "-hide_banner", "-y",
    "-f", "concat", "-safe", "0",
    "-i", str(concat_file),
    "-c:v", "libx264", "-preset", "slow", "-crf", "17",
    "-profile:v", "high", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k",
    "-movflags", "+faststart",
    str(reel_path),
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(f"REEL ERROR: {r.stderr[-500:]}")
else:
    dur = json.loads(subprocess.check_output(
        ["ffprobe","-v","error","-show_entries","format=duration",
         "-of","json",str(reel_path)],
        text=True, stderr=subprocess.DEVNULL
    ))["format"]["duration"]
    print(f"\nHighlight reel complete! Duration: {float(dur):.1f}s")
    
    # Copy to Desktop
    desktop_reel = DESKTOP / f"{GAME}__highlight_reel.mp4"
    shutil.copy2(str(reel_path), str(desktop_reel))
    print(f"Copied to Desktop: {desktop_reel.name}")
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
