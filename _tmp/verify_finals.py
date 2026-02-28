import glob, os, subprocess, json

REEL_DIR = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
LOG = r"D:\Projects\soccer-video\_tmp\final_report.txt"

def probe(path):
    p = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,duration,nb_frames',
         '-of', 'json', path],
        capture_output=True, text=True
    )
    try:
        s = json.loads(p.stdout)["streams"][0]
        return {
            "w": int(s.get("width", 0)),
            "h": int(s.get("height", 0)),
            "dur": float(s.get("duration", 0)),
            "frames": int(s.get("nb_frames", 0))
        }
    except:
        return {"w": 0, "h": 0, "dur": 0, "frames": 0}

pattern = os.path.join(REEL_DIR, "*__portrait__FINAL.mp4")
finals = sorted(glob.glob(pattern))

total_size = 0
total_dur = 0
errors = []

with open(LOG, "w") as f:
    f.write(f"PORTRAIT REEL FINAL VERIFICATION REPORT\n")
    f.write(f"Game: 2026-02-23 TSC vs NEOFC\n")
    f.write(f"Total clips: {len(finals)}\n")
    f.write(f"{'='*90}\n\n")
    f.write(f"{'#':>3}  {'Resolution':>10}  {'Duration':>8}  {'Size MB':>8}  {'Frames':>6}  Filename\n")
    f.write(f"{'-'*90}\n")
    
    for i, fp in enumerate(finals, 1):
        info = probe(fp)
        sz = os.path.getsize(fp) / (1024*1024)
        total_size += sz
        total_dur += info["dur"]
        name = os.path.basename(fp)
        
        ok = info["w"] == 2160 and info["h"] == 3840
        flag = "" if ok else " *** BAD RESOLUTION ***"
        if not ok:
            errors.append(name)
        
        f.write(f"{i:3d}  {info['w']}x{info['h']:>4}  {info['dur']:7.1f}s  {sz:7.1f}  {info['frames']:>6}  {name[:60]}{flag}\n")
    
    f.write(f"\n{'='*90}\n")
    f.write(f"SUMMARY\n")
    f.write(f"  Total clips:    {len(finals)}\n")
    f.write(f"  Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)\n")
    f.write(f"  Total size:     {total_size:.1f} MB ({total_size/1024:.2f} GB)\n")
    f.write(f"  Avg clip size:  {total_size/len(finals):.1f} MB\n")
    f.write(f"  Resolution errors: {len(errors)}\n")
    if errors:
        for e in errors:
            f.write(f"    - {e}\n")
    else:
        f.write(f"  ALL CLIPS VERIFIED: 2160x3840 (4K portrait)\n")
