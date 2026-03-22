"""Master queue: finish Greenwood 0221 batch, then render NEOFC 005/006/007."""
import time, os, sys, csv, json, subprocess, traceback, tempfile, shutil
import numpy as np
from pathlib import Path

os.chdir(r"D:\Projects\soccer-video")
STATUS = r"D:\Projects\soccer-video\_tmp\master_queue_status.txt"
PYTHON = sys.executable
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()
FRAME_W = 1920
FPS = 30
DEFAULT_CY = 540

def log(msg):
    with open(STATUS, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

def run_cmd(cmd, timeout=600, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd or os.getcwd())

# ── Greenwood finisher ──
def finalize_greenwood_clip(clip_path, clip_num):
    """Run vidstab + 4K for one Greenwood clip. Skip if FINAL exists."""
    game = "2026-02-21__TSC_vs_Greenwood"
    reels = Path(f"out/portrait_reels/{game}")
    stem = clip_path.stem
    portrait = reels / f"{stem}__portrait.mp4"
    final = reels / f"{stem}__portrait__FINAL.mp4"
    
    if final.exists():
        log(f"  [{clip_num}] SKIPPED (FINAL exists)")
        return True
    
    if not portrait.exists():
        # Need full render
        log(f"  [{clip_num}] Portrait missing, running full render...")
        cmd = [PYTHON, "tools/render_follow_unified.py",
               "--in", str(clip_path), "--src", str(clip_path),
               "--out", str(portrait), "--portrait", "1080x1920",
               "--preset", "cinematic", "--fps", "24", "--diagnostics",
               "--use-ball-telemetry", "--keep-scratch"]
        t0 = time.time()
        r = run_cmd(cmd)
        if r.returncode != 0:
            log(f"  [{clip_num}] FAIL render"); return False
        log(f"  [{clip_num}] Portrait done: {time.time()-t0:.0f}s")
    
    # vidstab detect
    t0 = time.time()
    trf = f"gw{clip_num}_transforms.trf"
    r1 = run_cmd([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0:
        log(f"  [{clip_num}] FAIL vidstab detect"); return False
    log(f"  [{clip_num}] vidstab detect: {time.time()-t0:.0f}s")
    
    # vidstab transform
    t0 = time.time()
    stab_tmp = os.path.join(TEMP_DIR, f"gw{clip_num}_stab.mp4")
    r2 = run_cmd([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp], cwd=TEMP_DIR)
    if r2.returncode != 0:
        log(f"  [{clip_num}] FAIL vidstab transform"); return False
    log(f"  [{clip_num}] vidstab transform: {time.time()-t0:.0f}s")
    
    # 4K upscale
    t0 = time.time()
    fa = str(final.resolve())
    if os.path.exists(fa): os.remove(fa)
    r3 = run_cmd([FFMPEG, "-y", "-i", stab_tmp, "-vf",
        "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "copy", fa], cwd=TEMP_DIR)
    if r3.returncode != 0:
        log(f"  [{clip_num}] FAIL 4K upscale"); return False
    log(f"  [{clip_num}] 4K upscale: {time.time()-t0:.0f}s")
    
    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if os.path.exists(tmp): os.remove(tmp)
    
    fmb = os.path.getsize(fa) / (1024*1024)
    log(f"  [{clip_num}] COMPLETE: {fmb:.1f} MB")
    return True

# ── NEOFC renderer ──
def render_neofc_clip(clip_num):
    """Full pipeline for one NEOFC clip with manual anchors."""
    game = "2026-02-23__TSC_vs_NEOFC"
    csv_path = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
    
    clips_dir = Path(f"out/atomic_clips/{game}")
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    stem = clip_file.stem
    telem_dir = Path("out/telemetry")
    
    ball_cands = list(telem_dir.glob(f"{clip_num}__{game}*.ball.jsonl"))
    ball_jsonl = ball_cands[0] if ball_cands else None
    yolo_cands = [p for p in list(telem_dir.glob(f"{clip_num}__{game}*.yolo_ball.jsonl")) +
                  list(telem_dir.glob(f"{clip_num}__{game}*.yolo_ball.yolov8x.jsonl"))
                  if not str(p).endswith("bak")]
    yolo_jsonl = yolo_cands[0] if yolo_cands else None
    tracker_cands = [p for p in list(telem_dir.glob(f"{clip_num}__{game}*.tracker_ball.jsonl")) +
                     list(telem_dir.glob(f"{clip_num}__{game}*.tracker_ball.yolov8x.jsonl"))
                     if not str(p).endswith("bak")]
    tracker_jsonl = tracker_cands[0] if tracker_cands else None
    
    if ball_jsonl is None and yolo_jsonl:
        ball_jsonl = yolo_jsonl.parent / yolo_jsonl.name.replace(".yolo_ball", ".ball").replace(".yolov8x", "")
    elif ball_jsonl is None:
        log(f"  [{clip_num}] No telemetry found!"); return False
    
    reels_dir = Path(f"out/portrait_reels/{game}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"
    render_log = reels_dir / f"{clip_num}__rerender.log"
    
    # Read CSV
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
    
    log(f"  [{clip_num}] {len(anchors)} anchors, {total_frames} frames")
    
    # Backup + patch
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
    
    # Step 1
    log(f"  [{clip_num}] Step 1: Portrait render...")
    cmd = [PYTHON, "tools/render_follow_unified.py",
           "--in", str(clip_file), "--src", str(clip_file),
           "--out", str(portrait), "--portrait", "1080x1920",
           "--preset", "cinematic", "--fps", "24", "--diagnostics",
           "--use-ball-telemetry", "--keep-scratch"]
    r = run_cmd(cmd)
    with open(render_log, "w") as f:
        f.write(r.stdout or "")
        if r.stderr: f.write("\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        log(f"  [{clip_num}] FAIL render"); return False
    log(f"  [{clip_num}] Step 1 done: {portrait.stat().st_size/(1024*1024):.1f} MB")
    
    # Step 2
    trf = f"neofc{clip_num}_transforms.trf"
    r1 = run_cmd([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0:
        log(f"  [{clip_num}] FAIL vidstab detect"); return False
    log(f"  [{clip_num}] Step 2 done")
    
    # Step 3
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{clip_num}_stab.mp4")
    r2 = run_cmd([FFMPEG, "-y", "-i", str(portrait.resolve()), "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp], cwd=TEMP_DIR)
    if r2.returncode != 0:
        log(f"  [{clip_num}] FAIL vidstab transform"); return False
    log(f"  [{clip_num}] Step 3 done")
    
    # Step 4
    fa = str(final.resolve())
    if os.path.exists(fa): os.remove(fa)
    r3 = run_cmd([FFMPEG, "-y", "-i", stab_tmp, "-vf",
        "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "copy", fa], cwd=TEMP_DIR)
    if r3.returncode != 0:
        log(f"  [{clip_num}] FAIL 4K upscale"); return False
    log(f"  [{clip_num}] Step 4 done")
    
    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if os.path.exists(tmp): os.remove(tmp)
    
    total = time.time() - t_total
    fmb = os.path.getsize(fa) / (1024*1024)
    
    # Copy to clean + desktop
    clean = Path("out/portrait_reels/clean") / f"{stem}__CINEMATIC_portrait_FINAL.mp4"
    shutil.copy2(fa, clean)
    shutil.copy2(fa, rf"C:\Users\scott\Desktop\clip_{clip_num}.mp4")
    log(f"  [{clip_num}] COMPLETE: {total/60:.1f}m, {fmb:.1f} MB (copied to clean + Desktop)")
    return True

# ── MAIN ──
try:
    with open(STATUS, "w") as f:
        f.write("Master queue started\n")
    
    # Part 1: Finish Greenwood
    gw_game = "2026-02-21__TSC_vs_Greenwood"
    gw_clips = sorted(Path(f"out/atomic_clips/{gw_game}").glob("*.mp4"))
    gw_reels = Path(f"out/portrait_reels/{gw_game}")
    
    gw_todo = []
    for c in gw_clips:
        cn = c.name.split("__")[0]
        final_check = list(gw_reels.glob(f"{cn}__*__portrait__FINAL.mp4"))
        if not final_check:
            gw_todo.append((c, cn))
    
    log(f"Greenwood: {len(gw_todo)} clips remaining")
    gw_ok = 0
    for clip_path, clip_num in gw_todo:
        log(f"Greenwood clip {clip_num}...")
        if finalize_greenwood_clip(clip_path, clip_num):
            gw_ok += 1
        else:
            log(f"  [{clip_num}] FAILED - continuing")
    log(f"Greenwood done: {gw_ok}/{len(gw_todo)} succeeded\n")
    
    # Part 2: NEOFC clips
    for cn in ["005", "006", "007"]:
        log(f"=== NEOFC clip {cn} ===")
        ok = render_neofc_clip(cn)
        log(f"NEOFC {cn}: {'SUCCESS' if ok else 'FAILED'}\n")
    
    log("MASTER QUEUE COMPLETE")

except Exception as e:
    with open(STATUS, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
