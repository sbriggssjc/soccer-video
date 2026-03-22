"""Re-render clip 004 v3: wider zoom during fast movement.
Changes from v2:
  - --zoom-min 0.72 (was 1.0) - allows wider FOV during fast action
  - --zoom-out-max 1.5 (was default) - more aggressive auto zoom-out
  - vidstab zoom=1 (was 3) - less post-render zoom
"""
import os, sys, csv, json, traceback, subprocess, time, tempfile, shutil
import numpy as np

CLIP_NUM = "004"
GAME = "2026-02-23__TSC_vs_NEOFC"
CSV_PATH = rf"C:\Users\scott\Desktop\review_{CLIP_NUM}.csv"
RESULT = rf"D:\Projects\soccer-video\_tmp\apply_render_{CLIP_NUM}_v3_result.txt"
os.makedirs(r"D:\Projects\soccer-video\_tmp", exist_ok=True)
os.chdir(r"D:\Projects\soccer-video")

FRAME_W = 1920
FPS = 30
DEFAULT_CY = 540
PYTHON = sys.executable
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()

from pathlib import Path
clips_dir = Path(f"out/atomic_clips/{GAME}")
clip_file = next(clips_dir.glob(f"{CLIP_NUM}__*.mp4"))
stem = clip_file.stem

telem_dir = Path("out/telemetry")
ball_jsonl = next(telem_dir.glob(f"{CLIP_NUM}__{GAME}*.ball.jsonl"))
yolo_candidates = list(telem_dir.glob(f"{CLIP_NUM}__{GAME}*.yolo_ball.jsonl")) + \
                  list(telem_dir.glob(f"{CLIP_NUM}__{GAME}*.yolo_ball.yolov8x.jsonl"))
yolo_jsonl = yolo_candidates[0] if yolo_candidates else None
tracker_candidates = list(telem_dir.glob(f"{CLIP_NUM}__{GAME}*.tracker_ball.jsonl")) + \
                     list(telem_dir.glob(f"{CLIP_NUM}__{GAME}*.tracker_ball.yolov8x.jsonl"))
tracker_jsonl = tracker_candidates[0] if tracker_candidates else None

reels_dir = Path(f"out/portrait_reels/{GAME}")
portrait = reels_dir / f"{stem}__portrait.mp4"
final = reels_dir / f"{stem}__portrait__FINAL.mp4"
render_log = reels_dir / f"{CLIP_NUM}__rerender.log"

def log(msg):
    with open(RESULT, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

try:
    with open(RESULT, "w") as f:
        f.write(f"Re-render clip {CLIP_NUM} v3 (wider zoom)\n")

    # Read review CSV
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    anchors = {}
    for row in rows:
        frame = int(row["frame"])
        pct = float(row["ball_x_pct"])
        anchors[frame] = pct

    diag_file = next(reels_dir.glob(f"{CLIP_NUM}__*__portrait.diag.csv"))
    with open(diag_file, "r") as f:
        total_frames = sum(1 for _ in f) - 1

    max_anchor = max(anchors.keys())
    if max_anchor < total_frames - 1:
        anchors[total_frames - 1] = anchors[max_anchor]

    anchor_frames = sorted(anchors.keys())
    anchor_cx = [anchors[f] / 100.0 * FRAME_W for f in anchor_frames]
    all_frames = np.arange(total_frames)
    cx_interp = np.clip(np.interp(all_frames, anchor_frames, anchor_cx), 0, FRAME_W)

    log(f"Anchors: {len(anchors)} points, {total_frames} total frames")

    # Patch telemetry
    telem_files = [ball_jsonl]
    if yolo_jsonl: telem_files.append(yolo_jsonl)
    if tracker_jsonl: telem_files.append(tracker_jsonl)
    for path in telem_files:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

    with open(ball_jsonl, "w") as f:
        for i in range(total_frames):
            row = {"frame": i, "t": round(i/FPS, 6),
                   "cx": round(float(cx_interp[i]), 2), "cy": DEFAULT_CY,
                   "w": 30, "h": 30, "conf": 0.90}
            f.write(json.dumps(row) + "\n")

    yolo_out = yolo_jsonl if yolo_jsonl else (ball_jsonl.parent / ball_jsonl.name.replace(".ball.jsonl", ".yolo_ball.jsonl"))
    with open(yolo_out, "w") as f:
        for i in range(total_frames):
            row = {"frame": i, "t": round(i/FPS, 6),
                   "cx": round(float(cx_interp[i]), 2), "cy": DEFAULT_CY,
                   "conf": 0.90}
            f.write(json.dumps(row) + "\n")

    if tracker_jsonl:
        with open(tracker_jsonl, "w") as f:
            pass

    log("Telemetry patched")
    t_total = time.time()

    # Step 1: Portrait render with WIDER ZOOM settings
    log("STEP 1: Portrait render (wider zoom)...")
    cmd = [PYTHON, "tools/render_follow_unified.py",
           "--in", str(clip_file), "--src", str(clip_file),
           "--out", str(portrait), "--portrait", "1080x1920",
           "--preset", "cinematic", "--fps", "24", "--diagnostics",
           "--use-ball-telemetry", "--keep-scratch",
           "--zoom-min", "0.72",
           "--zoom-out-max", "1.5",
           ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    with open(render_log, "w") as f:
        f.write(result.stdout or "")
        if result.stderr: f.write("\n--- STDERR ---\n" + result.stderr)
    if result.returncode != 0:
        log(f"FAIL render rc={result.returncode}")
        log((result.stderr or "")[-2000:])
        sys.exit(1)
    pmb = portrait.stat().st_size / (1024*1024)
    log(f"STEP 1 DONE: {time.time()-t0:.0f}s, {pmb:.1f} MB")

    # Step 2: vidstab detect
    log("STEP 2: vidstab detect...")
    t0 = time.time()
    trf = f"neofc{CLIP_NUM}_v3_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0:
        log(f"FAIL vidstab detect"); sys.exit(1)
    log(f"STEP 2 DONE: {time.time()-t0:.0f}s")

    # Step 3: vidstab transform — REDUCED ZOOM (1 instead of 3)
    log("STEP 3: vidstab transform (zoom=1)...")
    t0 = time.time()
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{CLIP_NUM}_v3_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=1",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0:
        log(f"FAIL vidstab transform"); sys.exit(1)
    log(f"STEP 3 DONE: {time.time()-t0:.0f}s")

    # Step 4: 4K upscale
    log("STEP 4: 4K upscale...")
    t0 = time.time()
    final_abs = str(final.resolve())
    if os.path.exists(final_abs): os.remove(final_abs)
    r3 = subprocess.run([FFMPEG, "-y", "-i", stab_tmp, "-vf",
        "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "copy", final_abs],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r3.returncode != 0:
        log(f"FAIL 4K upscale"); sys.exit(1)
    log(f"STEP 4 DONE: {time.time()-t0:.0f}s")

    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if os.path.exists(tmp): os.remove(tmp)

    total = time.time() - t_total
    fmb = os.path.getsize(final_abs) / (1024*1024)
    log(f"\nSUCCESS - Total: {total:.0f}s ({total/60:.1f}m), {fmb:.1f} MB")
    log(f"Changes: --zoom-min 0.72, --zoom-out-max 1.5, vidstab zoom=1")

except Exception as e:
    with open(RESULT, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
