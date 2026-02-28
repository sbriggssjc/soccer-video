"""Queue: render NEOFC clips 005 and 006 after Greenwood batch completes.
Each clip gets its own CSV, telemetry patching, and full pipeline.
"""
import time, os, sys, csv, json, subprocess, traceback, tempfile, shutil
import numpy as np
from pathlib import Path

os.chdir(r"D:\Projects\soccer-video")
BATCH_RESULT = r"D:\Projects\soccer-video\_tmp\batch_greenwood_0221_result.txt"
STATUS_FILE = r"D:\Projects\soccer-video\_tmp\render_queue_status.txt"
PYTHON = sys.executable
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()
GAME = "2026-02-23__TSC_vs_NEOFC"
FRAME_W = 1920
FPS = 30
DEFAULT_CY = 540

def log(msg):
    with open(STATUS_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg)

def render_clip(clip_num):
    """Full pipeline for one NEOFC clip: patch telemetry + render + vidstab + 4K."""
    csv_path = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
    result_file = rf"D:\Projects\soccer-video\_tmp\apply_render_{clip_num}_result.txt"
    
    clips_dir = Path(f"out/atomic_clips/{GAME}")
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    stem = clip_file.stem
    
    telem_dir = Path("out/telemetry")
    
    # Find telemetry files
    ball_candidates = list(telem_dir.glob(f"{clip_num}__{GAME}*.ball.jsonl"))
    ball_jsonl = ball_candidates[0] if ball_candidates else None
    
    yolo_candidates = list(telem_dir.glob(f"{clip_num}__{GAME}*.yolo_ball.jsonl")) + \
                      list(telem_dir.glob(f"{clip_num}__{GAME}*.yolo_ball.yolov8x.jsonl"))
    yolo_candidates = [p for p in yolo_candidates if not str(p).endswith("bak")]
    yolo_jsonl = yolo_candidates[0] if yolo_candidates else None
    
    tracker_candidates = list(telem_dir.glob(f"{clip_num}__{GAME}*.tracker_ball.jsonl")) + \
                         list(telem_dir.glob(f"{clip_num}__{GAME}*.tracker_ball.yolov8x.jsonl"))
    tracker_candidates = [p for p in tracker_candidates if not str(p).endswith("bak")]
    tracker_jsonl = tracker_candidates[0] if tracker_candidates else None
    
    if ball_jsonl is None and yolo_jsonl is not None:
        ball_jsonl = yolo_jsonl.parent / yolo_jsonl.name.replace(".yolo_ball", ".ball").replace(".yolov8x", "")
    elif ball_jsonl is None:
        raise FileNotFoundError(f"No telemetry for clip {clip_num}")
    
    reels_dir = Path(f"out/portrait_reels/{GAME}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"
    render_log = reels_dir / f"{clip_num}__rerender.log"
    
    with open(result_file, "w") as f:
        f.write(f"Apply + render clip {clip_num}\n")
    
    def rlog(msg):
        with open(result_file, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        log(f"  [{clip_num}] {msg}")
    
    # Read CSV
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    anchors = {}
    for row in rows:
        frame = int(row["frame"])
        pct = float(row["ball_x_pct"])
        anchors[frame] = pct
    
    diag_file = next(reels_dir.glob(f"{clip_num}__*__portrait.diag.csv"))
    with open(diag_file, "r") as f:
        total_frames = sum(1 for _ in f) - 1
    
    max_anchor = max(anchors.keys())
    if max_anchor < total_frames - 1:
        anchors[total_frames - 1] = anchors[max_anchor]
    
    anchor_frames = sorted(anchors.keys())
    anchor_cx = [anchors[f] / 100.0 * FRAME_W for f in anchor_frames]
    all_frames = np.arange(total_frames)
    cx_interp = np.clip(np.interp(all_frames, anchor_frames, anchor_cx), 0, FRAME_W)
    
    rlog(f"Anchors: {len(anchors)} points, {total_frames} total frames")
    
    # Backup + patch telemetry
    telem_files = [ball_jsonl]
    if yolo_jsonl: telem_files.append(yolo_jsonl)
    if tracker_jsonl: telem_files.append(tracker_jsonl)
    for path in telem_files:
        bak = path.with_suffix(path.suffix + ".bak")
        if path.exists() and not bak.exists():
            shutil.copy2(path, bak)
    
    with open(ball_jsonl, "w") as f:
        for i in range(total_frames):
            r = {"frame": i, "t": round(i/FPS, 6),
                 "cx": round(float(cx_interp[i]), 2), "cy": DEFAULT_CY,
                 "w": 30, "h": 30, "conf": 0.90}
            f.write(json.dumps(r) + "\n")
    
    yolo_out = yolo_jsonl if yolo_jsonl else (ball_jsonl.parent / ball_jsonl.name.replace(".ball.jsonl", ".yolo_ball.jsonl"))
    with open(yolo_out, "w") as f:
        for i in range(total_frames):
            r = {"frame": i, "t": round(i/FPS, 6),
                 "cx": round(float(cx_interp[i]), 2), "cy": DEFAULT_CY,
                 "conf": 0.90}
            f.write(json.dumps(r) + "\n")
    
    if tracker_jsonl:
        with open(tracker_jsonl, "w") as f:
            pass
    
    rlog("Telemetry patched")
    t_total = time.time()
    
    # Step 1: Portrait render
    rlog("Step 1: Portrait render...")
    cmd = [PYTHON, "tools/render_follow_unified.py",
           "--in", str(clip_file), "--src", str(clip_file),
           "--out", str(portrait), "--portrait", "1080x1920",
           "--preset", "cinematic", "--fps", "24", "--diagnostics",
           "--use-ball-telemetry", "--keep-scratch"]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    with open(render_log, "w") as f:
        f.write(result.stdout or "")
        if result.stderr: f.write("\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0:
        rlog(f"FAIL render rc={result.returncode}")
        return False
    rlog(f"Step 1 done: {time.time()-t0:.0f}s, {portrait.stat().st_size/(1024*1024):.1f} MB")
    
    # Step 2: vidstab detect
    rlog("Step 2: vidstab detect...")
    t0 = time.time()
    trf = f"neofc{clip_num}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0:
        rlog("FAIL vidstab detect"); return False
    rlog(f"Step 2 done: {time.time()-t0:.0f}s")
    
    # Step 3: vidstab transform
    rlog("Step 3: vidstab transform...")
    t0 = time.time()
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{clip_num}_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0:
        rlog("FAIL vidstab transform"); return False
    rlog(f"Step 3 done: {time.time()-t0:.0f}s")
    
    # Step 4: 4K upscale
    rlog("Step 4: 4K upscale...")
    t0 = time.time()
    final_abs = str(final.resolve())
    if os.path.exists(final_abs): os.remove(final_abs)
    r3 = subprocess.run([FFMPEG, "-y", "-i", stab_tmp, "-vf",
        "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "copy", final_abs],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r3.returncode != 0:
        rlog("FAIL 4K upscale"); return False
    rlog(f"Step 4 done: {time.time()-t0:.0f}s")
    
    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if os.path.exists(tmp): os.remove(tmp)
    
    total = time.time() - t_total
    fmb = os.path.getsize(final_abs) / (1024*1024)
    rlog(f"SUCCESS - Total: {total:.0f}s ({total/60:.1f}m), {fmb:.1f} MB")
    
    # Copy to clean folder + Desktop
    clean_name = f"{stem}__CINEMATIC_portrait_FINAL.mp4"
    clean_dst = Path("out/portrait_reels/clean") / clean_name
    shutil.copy2(final_abs, clean_dst)
    desktop_dst = rf"C:\Users\scott\Desktop\clip_{clip_num}.mp4"
    shutil.copy2(final_abs, desktop_dst)
    rlog(f"Copied to clean + Desktop ({fmb:.1f} MB)")
    return True

# ── Main ──
try:
    with open(STATUS_FILE, "w") as f:
        f.write("Render queue started\n")
    
    # Wait for Greenwood batch
    log("Waiting for Greenwood batch to complete...")
    while True:
        time.sleep(30)
        try:
            with open(BATCH_RESULT, "r") as f:
                content = f.read()
            if "BATCH COMPLETE" in content or "EXCEPTION" in content:
                log("Greenwood batch finished!")
                break
            lines = [l for l in content.strip().split("\n") if l.strip()]
            if lines:
                log(f"Batch: {lines[-1].strip()}")
        except Exception as e:
            log(f"Error reading batch: {e}")
    
    # Render clip 005
    log("=== Starting NEOFC clip 005 ===")
    ok5 = render_clip("005")
    log(f"Clip 005: {'SUCCESS' if ok5 else 'FAILED'}")
    
    # Render clip 006
    log("=== Starting NEOFC clip 006 ===")
    ok6 = render_clip("006")
    log(f"Clip 006: {'SUCCESS' if ok6 else 'FAILED'}")
    
    log(f"\nQUEUE COMPLETE: 005={'OK' if ok5 else 'FAIL'}, 006={'OK' if ok6 else 'FAIL'}")

except Exception as e:
    with open(STATUS_FILE, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
