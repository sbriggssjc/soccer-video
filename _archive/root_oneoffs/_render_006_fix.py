"""Re-render NEOFC clip 006 with manual anchors. Handles missing telemetry."""
import time, os, sys, csv, json, subprocess, traceback, tempfile, shutil
import numpy as np
from pathlib import Path

os.chdir(r"D:\Projects\soccer-video")
RESULT = r"D:\Projects\soccer-video\_tmp\render_006_fix_result.txt"
PYTHON = sys.executable
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()
GAME = "2026-02-23__TSC_vs_NEOFC"
FRAME_W = 1920; FPS = 30; DEFAULT_CY = 540
clip_num = "006"

def log(msg):
    with open(RESULT, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

try:
    with open(RESULT, "w") as f:
        f.write(f"Re-render clip {clip_num} (telemetry fix)\n")

    csv_path = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
    clips_dir = Path(f"out/atomic_clips/{GAME}")
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    stem = clip_file.stem
    telem_dir = Path("out/telemetry")
    reels_dir = Path(f"out/portrait_reels/{GAME}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"
    render_log = reels_dir / f"{clip_num}__rerender.log"

    # Build telemetry paths from existing files
    all_telem = list(telem_dir.glob(f"{clip_num}__{GAME}*"))
    log(f"Found {len(all_telem)} telemetry files: {[p.name for p in all_telem]}")

    has_yolov8x = any("yolov8x" in str(p) for p in all_telem)
    if has_yolov8x:
        base = None
        for p in all_telem:
            if "yolo_ball.yolov8x" in p.name and not p.name.endswith("bak"):
                base = p; break
        if base is None:
            for p in all_telem:
                if "yolo_ball.yolov8x.jsonl.1280bak" in p.name:
                    base = p.parent / p.name.replace(".1280bak", "")
                    break
        if base is None:
            base = telem_dir / f"{stem}.yolo_ball.yolov8x.jsonl"
        yolo_jsonl = base
        ball_jsonl = telem_dir / base.name.replace(".yolo_ball.yolov8x.jsonl", ".ball.jsonl")
        tracker_jsonl = telem_dir / base.name.replace(".yolo_ball.yolov8x.jsonl", ".tracker_ball.yolov8x.jsonl")
    else:
        ball_jsonl = telem_dir / f"{stem}.ball.jsonl"
        yolo_jsonl = telem_dir / f"{stem}.yolo_ball.jsonl"
        tracker_jsonl = telem_dir / f"{stem}.tracker_ball.jsonl"

    log(f"ball: {ball_jsonl.name}")
    log(f"yolo: {yolo_jsonl.name}")
    log(f"tracker: {tracker_jsonl.name}")

    # Check if CSV still exists on Desktop
    if not os.path.exists(csv_path):
        # Try tmp copy
        csv_path = rf"D:\Projects\soccer-video\_tmp\review_{clip_num}.csv"
    log(f"CSV: {csv_path}")

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    anchors = {int(r["frame"]): float(r["ball_x_pct"]) for r in rows}

    diag_file = next(reels_dir.glob(f"{clip_num}__*__portrait.diag.csv"), None)
    if diag_file is None:
        # No diag - estimate from clip duration
        import re
        m = re.search(r"t([\d.]+)-t([\d.]+)", stem)
        duration = float(m.group(2)) - float(m.group(1))
        total_frames = int(duration * FPS)
        log(f"No diag CSV, estimated {total_frames} frames from duration")
    else:
        with open(diag_file, "r") as f:
            total_frames = sum(1 for _ in f) - 1

    mx = max(anchors.keys())
    if mx < total_frames - 1:
        anchors[total_frames - 1] = anchors[mx]

    aframes = sorted(anchors.keys())
    acx = [anchors[f] / 100.0 * FRAME_W for f in aframes]
    cx_interp = np.clip(np.interp(np.arange(total_frames), aframes, acx), 0, FRAME_W)

    log(f"{len(anchors)} anchors, {total_frames} frames")

    # Backup existing files
    for path in [ball_jsonl, yolo_jsonl, tracker_jsonl]:
        if path.exists():
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists(): shutil.copy2(path, bak)

    # Write telemetry
    with open(ball_jsonl, "w") as f:
        for i in range(total_frames):
            f.write(json.dumps({"frame": i, "t": round(i/FPS,6),
                "cx": round(float(cx_interp[i]),2), "cy": DEFAULT_CY,
                "w": 30, "h": 30, "conf": 0.90}) + "\n")

    with open(yolo_jsonl, "w") as f:
        for i in range(total_frames):
            f.write(json.dumps({"frame": i, "t": round(i/FPS,6),
                "cx": round(float(cx_interp[i]),2), "cy": DEFAULT_CY, "conf": 0.90}) + "\n")

    with open(tracker_jsonl, "w") as f:
        pass

    log("Telemetry written")
    t_total = time.time()

    # Step 1
    log("Step 1: Portrait render...")
    r = subprocess.run([PYTHON, "tools/render_follow_unified.py",
        "--in", str(clip_file), "--src", str(clip_file),
        "--out", str(portrait), "--portrait", "1080x1920",
        "--preset", "cinematic", "--fps", "24", "--diagnostics",
        "--use-ball-telemetry", "--keep-scratch"],
        capture_output=True, text=True, timeout=1800)
    with open(render_log, "w") as f:
        f.write(r.stdout or "")
        if r.stderr: f.write("\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        log(f"FAIL render rc={r.returncode}")
        log((r.stderr or "")[-1000:])
        sys.exit(1)
    log(f"Step 1 done: {portrait.stat().st_size/(1024*1024):.1f} MB")

    # Step 2
    trf = f"neofc{clip_num}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0: log("FAIL vidstab detect"); sys.exit(1)
    log("Step 2 done")

    # Step 3
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{clip_num}_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0: log("FAIL vidstab transform"); sys.exit(1)
    log("Step 3 done")

    # Step 4
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

    # Clean up intermediates - only keep FINAL
    if portrait.exists(): portrait.unlink()
    if render_log.exists(): render_log.unlink()
    for dg in reels_dir.glob(f"{clip_num}__*diag*"):
        dg.unlink()

    total = time.time() - t_total
    fmb = os.path.getsize(fa) / (1024*1024)
    log(f"COMPLETE: {total/60:.1f}m, {fmb:.1f} MB")
    log(f"Saved to: {fa}")

except Exception as e:
    with open(RESULT, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
