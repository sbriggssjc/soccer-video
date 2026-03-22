"""Wait for master queue to finish, then render NEOFC clip 008."""
import time, os, sys, csv, json, subprocess, traceback, tempfile, shutil
import numpy as np
from pathlib import Path

os.chdir(r"D:\Projects\soccer-video")
STATUS = r"D:\Projects\soccer-video\_tmp\render_008_status.txt"
MASTER_STATUS = r"D:\Projects\soccer-video\_tmp\master_queue_status.txt"
PYTHON = sys.executable
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()
GAME = "2026-02-23__TSC_vs_NEOFC"
FRAME_W = 1920; FPS = 30; DEFAULT_CY = 540

def log(msg):
    with open(STATUS, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

with open(STATUS, "w") as f:
    f.write("Waiting for master queue...\n")

# Poll for master queue completion
while True:
    time.sleep(30)
    try:
        with open(MASTER_STATUS, "r") as f:
            content = f.read()
        if "MASTER QUEUE COMPLETE" in content or "EXCEPTION" in content:
            log("Master queue finished!")
            break
        lines = [l for l in content.strip().split("\n") if l.strip()]
        if lines: log(f"Queue: {lines[-1].strip()}")
    except: pass

clip_num = "008"
csv_path = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
log(f"Starting NEOFC clip {clip_num}...")

try:
    clips_dir = Path(f"out/atomic_clips/{GAME}")
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    stem = clip_file.stem
    telem_dir = Path("out/telemetry")
    
    ball_cands = list(telem_dir.glob(f"{clip_num}__{GAME}*.ball.jsonl"))
    ball_jsonl = ball_cands[0] if ball_cands else None
    yolo_cands = [p for p in list(telem_dir.glob(f"{clip_num}__{GAME}*.yolo_ball.jsonl")) +
                  list(telem_dir.glob(f"{clip_num}__{GAME}*.yolo_ball.yolov8x.jsonl"))
                  if not str(p).endswith("bak")]
    yolo_jsonl = yolo_cands[0] if yolo_cands else None
    tracker_cands = [p for p in list(telem_dir.glob(f"{clip_num}__{GAME}*.tracker_ball.jsonl")) +
                     list(telem_dir.glob(f"{clip_num}__{GAME}*.tracker_ball.yolov8x.jsonl"))
                     if not str(p).endswith("bak")]
    tracker_jsonl = tracker_cands[0] if tracker_cands else None
    
    if ball_jsonl is None and yolo_jsonl:
        ball_jsonl = yolo_jsonl.parent / yolo_jsonl.name.replace(".yolo_ball", ".ball").replace(".yolov8x", "")
    elif ball_jsonl is None:
        raise FileNotFoundError("No telemetry")
    
    reels_dir = Path(f"out/portrait_reels/{GAME}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"
    render_log = reels_dir / f"{clip_num}__rerender.log"
    
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    anchors = {int(r["frame"]): float(r["ball_x_pct"]) for r in rows}
    
    diag_file = next(reels_dir.glob(f"{clip_num}__*__portrait.diag.csv"))
    with open(diag_file, "r") as f:
        total_frames = sum(1 for _ in f) - 1
    
    mx = max(anchors.keys())
    if mx < total_frames - 1:
        anchors[total_frames - 1] = anchors[mx]
    
    aframes = sorted(anchors.keys())
    acx = [anchors[f] / 100.0 * FRAME_W for f in aframes]
    cx_interp = np.clip(np.interp(np.arange(total_frames), aframes, acx), 0, FRAME_W)
    
    log(f"{len(anchors)} anchors, {total_frames} frames")
    
    for path in [ball_jsonl, yolo_jsonl, tracker_jsonl]:
        if path and path.exists():
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists(): shutil.copy2(path, bak)
    
    with open(ball_jsonl, "w") as f:
        for i in range(total_frames):
            f.write(json.dumps({"frame": i, "t": round(i/FPS,6),
                "cx": round(float(cx_interp[i]),2), "cy": DEFAULT_CY,
                "w": 30, "h": 30, "conf": 0.90}) + "\n")
    
    yo = yolo_jsonl or (ball_jsonl.parent / ball_jsonl.name.replace(".ball.jsonl", ".yolo_ball.jsonl"))
    with open(yo, "w") as f:
        for i in range(total_frames):
            f.write(json.dumps({"frame": i, "t": round(i/FPS,6),
                "cx": round(float(cx_interp[i]),2), "cy": DEFAULT_CY, "conf": 0.90}) + "\n")
    
    if tracker_jsonl:
        with open(tracker_jsonl, "w") as f: pass
    
    t_total = time.time()
    
    log("Step 1: Portrait render...")
    r = subprocess.run([PYTHON, "tools/render_follow_unified.py",
        "--in", str(clip_file), "--src", str(clip_file),
        "--out", str(portrait), "--portrait", "1080x1920",
        "--preset", "cinematic", "--fps", "24", "--diagnostics",
        "--use-ball-telemetry", "--keep-scratch"],
        capture_output=True, text=True, timeout=600)
    with open(render_log, "w") as f:
        f.write(r.stdout or "")
        if r.stderr: f.write("\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        log(f"FAIL render rc={r.returncode}"); sys.exit(1)
    log(f"Step 1 done: {portrait.stat().st_size/(1024*1024):.1f} MB")
    
    trf = f"neofc{clip_num}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0: log("FAIL vidstab detect"); sys.exit(1)
    log("Step 2 done")
    
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{clip_num}_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0: log("FAIL vidstab transform"); sys.exit(1)
    log("Step 3 done")
    
    fa = str(final.resolve())
    if os.path.exists(fa): os.remove(fa)
    r3 = subprocess.run([FFMPEG, "-y", "-i", stab_tmp, "-vf",
        "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "copy", fa], capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r3.returncode != 0: log("FAIL 4K upscale"); sys.exit(1)
    log("Step 4 done")
    
    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if os.path.exists(tmp): os.remove(tmp)
    
    total = time.time() - t_total
    fmb = os.path.getsize(fa) / (1024*1024)
    
    clean = Path("out/portrait_reels/clean") / f"{stem}__CINEMATIC_portrait_FINAL.mp4"
    shutil.copy2(fa, clean)
    shutil.copy2(fa, rf"C:\Users\scott\Desktop\clip_{clip_num}.mp4")
    log(f"COMPLETE: {total/60:.1f}m, {fmb:.1f} MB (copied to clean + Desktop)")

except Exception as e:
    with open(STATUS, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
