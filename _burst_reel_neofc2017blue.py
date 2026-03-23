"""Generate burst highlights + game highlight reel for
2026-03-21__TSC_vs_NEOFC_2017_Blue.

Burst = the 3-4.5s peak action segment from each clip (goal moment,
shot on target, key cross, etc.). Heuristic: action climax lives in
the final 20-35% of the clip depending on play type.

Outputs:
  1. Individual burst clips  -> out/bursts/{GAME}/
  2. Game highlight reel     -> out/reels/{GAME}__highlight_reel.mp4

Usage: python _burst_reel_neofc2017blue.py
       python _burst_reel_neofc2017blue.py --bursts-only
       python _burst_reel_neofc2017blue.py --reel-only
"""
import subprocess, json, sys, re, math
from pathlib import Path

GAME = "2026-03-21__TSC_vs_NEOFC_2017_Blue"
PYTHON = r"C:\Users\scott\AppData\Local\Programs\Python\Python313\python.exe"

SRC_DIR = Path(r"D:\Projects\soccer-video\out\enhanced") / GAME
BURST_DIR = Path(r"D:\Projects\soccer-video\out\bursts") / GAME
REEL_DIR = Path(r"D:\Projects\soccer-video\out\reels")
DESKTOP = Path(r"C:\Users\scott\Desktop")

# Burst duration range
MIN_BURST = 3.0
MAX_BURST = 4.5

# Crossfade between bursts in the reel
CROSSFADE = 0.3  # seconds

# ---------- Priority & peak-location heuristics ----------

PRIORITY_RULES = [
    (5, ["GOAL"]),
    (4, ["SHOT"]),
    (3, ["CROSS", "SAVE", "FREE KICK", "CORNER"]),
    (2, ["BUILD", "DRIBBL", "PRESSURE", "THROUGH"]),
    (1, ["DEFENSE", "TACKLE", "BLOCK", "CLEAR"]),
]

def compute_priority(label):
    upper = label.upper()
    best = 0
    for score, tokens in PRIORITY_RULES:
        if any(tok in upper for tok in tokens):
            best = max(best, score)
    return best

def peak_ratio(label):
    """Where in the clip the peak action is (0.0=start, 1.0=end).
    Goals/shots climax near the end; defensive plays peak earlier."""
    upper = label.upper()
    if "GOAL" in upper:
        return 0.82
    if "SHOT" in upper:
        return 0.80
    if "CROSS" in upper:
        return 0.75
    if "SAVE" in upper or "CORNER" in upper or "FREE KICK" in upper:
        return 0.78
    return 0.70  # build/pressure/defense

def burst_duration(label, clip_dur):
    """Choose burst length based on action type and clip length."""
    upper = label.upper()
    if "GOAL" in upper:
        dur = 4.5
    elif "SHOT" in upper:
        dur = 4.0
    elif "CROSS" in upper or "DRIBBL" in upper:
        dur = 3.5
    else:
        dur = 3.0
    # Clamp: burst can't exceed 80% of clip
    dur = min(dur, clip_dur * 0.8)
    dur = max(MIN_BURST, min(MAX_BURST, dur))
    return round(dur, 2)

# ---------- ffprobe helper ----------

def probe_duration(path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "json", str(path)]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    return float(json.loads(out)["format"]["duration"])

# ---------- Clip discovery ----------

FNAME_RE = re.compile(
    r"^(?P<idx>\d+)__(?P<label>.+?)__[\d.]+-[\d.]+(?:_portrait_FINAL)?\.mp4$",
    re.IGNORECASE)

def discover():
    clips = []
    for mp4 in sorted(SRC_DIR.glob("*.mp4")):
        m = FNAME_RE.match(mp4.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        label = m.group("label").strip()
        dur = probe_duration(mp4)
        pri = compute_priority(label)
        pr = peak_ratio(label)
        bd = burst_duration(label, dur)
        # Center burst around peak moment
        peak_t = dur * pr
        bs = max(0, peak_t - bd / 2)
        be = bs + bd
        if be > dur:
            be = dur
            bs = max(0, be - bd)
        clips.append({
            "idx": idx, "label": label, "path": mp4,
            "duration": dur, "priority": pri,
            "burst_start": round(bs, 3),
            "burst_end": round(be, 3),
            "burst_dur": round(be - bs, 3),
        })
    clips.sort(key=lambda c: c["idx"])
    return clips

# ---------- Extract burst clips ----------

def extract_bursts(clips):
    BURST_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for c in clips:
        out_name = f"burst_{c['idx']:02d}__{c['label'].replace(' ', '_')}.mp4"
        out = BURST_DIR / out_name
        cmd = [
            "ffmpeg", "-hide_banner", "-y",
            "-ss", f"{c['burst_start']:.3f}",
            "-i", str(c["path"]),
            "-t", f"{c['burst_dur']:.3f}",
            "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out),
        ]
        print(f"  [{c['idx']:02d}] {c['label']}: "
              f"{c['burst_start']:.1f}s-{c['burst_end']:.1f}s "
              f"({c['burst_dur']:.1f}s burst) pri={c['priority']}")
        r = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print(f"    ERROR: {r.stderr[-200:]}", file=sys.stderr)
        else:
            out_paths.append(out)
    return out_paths

# ---------- Assemble highlight reel ----------

def order_for_pacing(clips):
    """Impact sandwich: best opener, chronological middle, strong closer."""
    if len(clips) <= 2:
        return sorted(clips, key=lambda c: (-c["priority"], c["idx"]))
    by_pri = sorted(clips, key=lambda c: (-c["priority"], c["idx"]))
    opener = by_pri[0]
    closer = by_pri[1]
    middle = [c for c in clips if c is not opener and c is not closer]
    middle.sort(key=lambda c: c["idx"])
    return [opener] + middle + [closer]

def build_reel(burst_paths, clips, output):
    """Concatenate burst clips with crossfade transitions."""
    if not burst_paths:
        print("No bursts to assemble!")
        return
    REEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Order bursts using impact sandwich pacing
    ordered = order_for_pacing(clips)
    ordered_paths = []
    for c in ordered:
        bp = BURST_DIR / f"burst_{c['idx']:02d}__{c['label'].replace(' ', '_')}.mp4"
        if bp.exists():
            ordered_paths.append((bp, c))

    n = len(ordered_paths)
    if n == 0:
        print("No burst files found!")
        return

    # Build ffmpeg filter graph with xfade transitions
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    
    # Add all inputs
    for bp, _ in ordered_paths:
        cmd += ["-i", str(bp)]
    
    filters = []
    # Prep each input: ensure consistent format
    for i in range(n):
        filters.append(
            f"[{i}:v]fps=24,scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p[v{i}]"
        )
        filters.append(
            f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]"
        )

    # Chain xfade transitions
    if n == 1:
        filters.append("[v0]copy[vout]")
        filters.append("[a0]acopy[aout]")
    else:
        # Get durations for offset calculation
        durations = []
        for bp, _ in ordered_paths:
            durations.append(probe_duration(bp))
        
        cur_v = "v0"
        cur_a = "a0"
        timeline = durations[0]
        
        for i in range(1, n):
            cf = min(CROSSFADE, durations[i-1]/2, durations[i]/2)
            offset = max(timeline - cf - 0.001, 0.0)
            out_v = f"xv{i}"
            out_a = f"xa{i}"
            filters.append(
                f"[{cur_v}][v{i}]xfade=transition=fade:"
                f"duration={cf:.3f}:offset={offset:.3f}[{out_v}]"
            )
            filters.append(
                f"[{cur_a}][a{i}]acrossfade=d={cf:.3f}:"
                f"curve1=tri:curve2=tri[{out_a}]"
            )
            cur_v = out_v
            cur_a = out_a
            timeline += durations[i] - cf
        
        filters.append(f"[{cur_v}]copy[vout]")
        filters.append(f"[{cur_a}]volume=1.0[aout]")

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]",
        "-r", "24",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output),
    ]

    est_dur = sum(d for d in durations) - CROSSFADE * (n - 1)
    print(f"\n[reel] Assembling {n} bursts -> {output}")
    print(f"[reel] Estimated duration: {est_dur:.1f}s")
    print(f"[reel] Pacing order:")
    for i, (_, c) in enumerate(ordered_paths):
        tag = ""
        if i == 0: tag = " << OPENER"
        elif i == n-1: tag = " << CLOSER"
        print(f"  {i+1:2d}. [{c['idx']:02d}] {c['label']} "
              f"(pri={c['priority']}, {c['burst_dur']:.1f}s){tag}")

    r = subprocess.run(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        print(f"\n[reel] FFMPEG ERROR:\n{r.stderr[-500:]}", file=sys.stderr)
        return None
    
    print(f"\n[reel] Done! -> {output}")
    return output

# ---------- Main ----------

def main():
    import shutil
    bursts_only = "--bursts-only" in sys.argv
    reel_only = "--reel-only" in sys.argv

    print("=" * 60)
    print(f"BURST HIGHLIGHTS + REEL: {GAME}")
    print("=" * 60)

    # Discover clips
    print(f"\nScanning {SRC_DIR} ...")
    clips = discover()
    print(f"Found {len(clips)} clips\n")

    if not clips:
        print("No clips found!", file=sys.stderr)
        return 1

    # Show burst plan
    total_burst = 0
    print(f"{'#':>3} | {'Pri':>3} | {'Clip':>6} | {'Burst':>12} | "
          f"{'Dur':>5} | Label")
    print("-" * 70)
    for c in clips:
        print(f"{c['idx']:3d} |  {c['priority']}  | "
              f"{c['duration']:5.1f}s | "
              f"{c['burst_start']:5.1f}-{c['burst_end']:5.1f}s | "
              f"{c['burst_dur']:4.1f}s | {c['label']}")
        total_burst += c["burst_dur"]
    print(f"\nTotal burst content: {total_burst:.1f}s "
          f"(from {sum(c['duration'] for c in clips):.1f}s source)")

    # Extract bursts
    if not reel_only:
        print(f"\n--- Extracting burst clips to {BURST_DIR} ---\n")
        burst_paths = extract_bursts(clips)
        print(f"\nExtracted {len(burst_paths)} burst clips")
    else:
        burst_paths = sorted(BURST_DIR.glob("burst_*.mp4"))
        print(f"\nUsing {len(burst_paths)} existing burst clips")

    if bursts_only:
        # Copy bursts to Desktop
        for bp in sorted(BURST_DIR.glob("burst_*.mp4")):
            dest = DESKTOP / bp.name
            shutil.copy2(str(bp), str(dest))
            print(f"  -> Desktop: {bp.name}")
        print("\nDone! Burst clips on Desktop.")
        return 0

    # Build reel
    print(f"\n--- Building highlight reel ---\n")
    reel_name = f"{GAME}__highlight_reel.mp4"
    reel_path = REEL_DIR / reel_name
    result = build_reel(burst_paths, clips, reel_path)

    if result:
        # Copy reel to Desktop
        desktop_reel = DESKTOP / reel_name
        shutil.copy2(str(result), str(desktop_reel))
        print(f"\n{'=' * 60}")
        print(f"DONE! Highlight reel on Desktop:")
        print(f"  {desktop_reel}")
        print(f"{'=' * 60}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
