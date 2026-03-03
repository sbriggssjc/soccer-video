"""Probe all FINAL clips to check for frame count / duration issues."""
import subprocess, os, re

d = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_Greenwood"
files = sorted([f for f in os.listdir(d) if "FINAL" in f])

for f in files:
    fp = os.path.join(d, f)
    clip_num = f[:3]
    
    # Get source clip info
    src_dir = r"D:\Projects\soccer-video\out\atomic_clips\2026-02-23__TSC_vs_Greenwood"
    src_files = [s for s in os.listdir(src_dir) if s.startswith(clip_num)]
    src_fp = os.path.join(src_dir, src_files[0]) if src_files else None
    
    # Probe the FINAL output
    r = subprocess.run([
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,r_frame_rate,width,height,duration",
        "-show_entries", "format=duration",
        "-print_format", "flat", fp
    ], capture_output=True, text=True, timeout=10)
    
    lines = r.stdout.strip().split("\n")
    info = {}
    for line in lines:
        if "=" in line:
            key, val = line.split("=", 1)
            info[key.strip()] = val.strip().strip('"')
    
    fps = info.get("streams.stream.0.r_frame_rate", "?")
    w = info.get("streams.stream.0.width", "?")
    h = info.get("streams.stream.0.height", "?")
    dur = info.get("format.duration", "?")
    nb = info.get("streams.stream.0.nb_frames", "?")
    
    sz = os.path.getsize(fp) / (1024*1024)
    
    # Flag potential issues
    flags = []
    if fps != "24/1" and fps != "24000/1001":
        flags.append(f"FPS={fps}")
    if w != "1080" or h != "1920":
        flags.append(f"DIMS={w}x{h}")
    
    flag_str = " *** " + ", ".join(flags) if flags else ""
    print(f"{clip_num}: {w}x{h} {fps} {nb}frames {float(dur):.1f}s {sz:.1f}MB{flag_str}")
