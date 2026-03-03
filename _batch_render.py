"""Batch direct crop renderer for NEOFC clips 006-034 (30fps re-render with recovered anchors)."""
import time, os, sys, csv, re, subprocess, traceback, tempfile, shutil, gc
import numpy as np
import cv2
from pathlib import Path

os.chdir(r"D:\Projects\soccer-video")
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()
GAME = "2026-02-23__TSC_vs_NEOFC"
SKIP_4K_UPSCALE = True

SRC_W = 1920; SRC_H = 1080
PORT_W = 1080; PORT_H = 1920
FPS_OUT = 30; FPS_SRC = 30

# All clips use zoom=1 now (wider framing)
ZOOM_OVERRIDE = {}
DEFAULT_ZOOM = 1

CLIP_NUMS = [f"{i:03d}" for i in range(6, 35)]

RESULT_FILE = r"D:\Projects\soccer-video\_tmp\batch_render_result.txt"
os.makedirs(r"D:\Projects\soccer-video\_tmp", exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except OSError:
        pass  # stdout redirect may be invalid; result file is the real log
    with open(RESULT_FILE, "a") as f:
        f.write(line + "\n")

def render_clip(clip_num):
    zoom = ZOOM_OVERRIDE.get(clip_num, DEFAULT_ZOOM)
    csv_path = rf"C:\Users\scott\Desktop\review_neofc_{clip_num}.csv"
    clips_dir = Path(f"out/atomic_clips/{GAME}")
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    stem = clip_file.stem
    reels_dir = Path(f"out/portrait_reels/{GAME}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"

    log(f"--- CLIP {clip_num} (zoom={zoom}) ---")
    log(f"File: {clip_file.name}")

    # Get actual frame count via OpenCV
    cap_probe = cv2.VideoCapture(str(clip_file))
    total_src_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_src_frames / FPS_SRC
    cap_probe.release()
    m = re.search(r"t?([\d.]+)-t?([\d.]+)", stem)
    if m:
        fname_dur = float(m.group(2)) - float(m.group(1))
        log(f"Actual: {total_src_frames} frames, {duration:.2f}s (filename said {fname_dur}s)")
    else:
        log(f"Actual: {total_src_frames} frames, {duration:.2f}s")

    # Read anchors
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    anchors = {}
    for r in rows:
        val = r["camera_x_pct"].strip()
        if val:
            anchors[int(r["frame"])] = float(val)
    # Extend last anchor to end
    mx = max(anchors.keys())
    if mx < total_src_frames - 1:
        anchors[total_src_frames - 1] = anchors[mx]

    aframes = sorted(anchors.keys())
    acx_pct = [anchors[f] for f in aframes]
    all_cx_pct = np.interp(np.arange(total_src_frames), aframes, acx_pct)

    CROP_W = int(SRC_H * PORT_W / PORT_H)
    CROP_W = CROP_W + (CROP_W % 2)
    CROP_H = SRC_H
    half_crop = CROP_W / 2

    all_cx_px = all_cx_pct / 100.0 * SRC_W
    all_cx_px = np.clip(all_cx_px, half_crop, SRC_W - half_crop)
    all_crop_x = (all_cx_px - half_crop).astype(int)

    log(f"{len(anchors)} anchors, crop {CROP_W}x{CROP_H}")
    t_clip = time.time()

    # Step 1: Direct crop via OpenCV -> pipe to ffmpeg
    cap = cv2.VideoCapture(str(clip_file))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w != SRC_W or actual_h != SRC_H:
        CROP_W = int(actual_h * PORT_W / PORT_H)
        CROP_W = CROP_W + (CROP_W % 2)
        CROP_H = actual_h
        half_crop = CROP_W / 2
        all_cx_px = all_cx_pct / 100.0 * actual_w
        all_cx_px = np.clip(all_cx_px, half_crop, actual_w - half_crop)
        all_crop_x = (all_cx_px - half_crop).astype(int)

    total_out_frames = int(duration * FPS_OUT)
    out_to_src = np.round(np.linspace(0, total_src_frames - 1, total_out_frames)).astype(int)

    portrait_tmp = os.path.join(TEMP_DIR, f"batch{clip_num}_portrait.mp4")
    ffmpeg_proc = subprocess.Popen([
        FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{CROP_W}x{CROP_H}", "-r", str(FPS_OUT),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p", portrait_tmp
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_idx = 0; out_idx = 0; frames_written = 0
    while out_idx < total_out_frames:
        target_src = out_to_src[out_idx]
        while frame_idx < target_src:
            cap.grab()
            frame_idx += 1
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        cx = int(all_crop_x[min(target_src, len(all_crop_x)-1)])
        cx = max(0, min(cx, actual_w - CROP_W))
        cropped = frame[0:CROP_H, cx:cx+CROP_W]
        ffmpeg_proc.stdin.write(cropped.tobytes())
        frames_written += 1
        out_idx += 1
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait(timeout=600)
    log(f"Wrote {frames_written} frames")

    # Scale to portrait + add audio
    portrait_str = str(portrait.resolve())
    if portrait.exists(): portrait.unlink()
    out_duration = frames_written / FPS_OUT

    r = subprocess.run([FFMPEG, "-y",
        "-i", portrait_tmp, "-i", str(clip_file),
        "-filter_complex", f"[0:v]scale={PORT_W}:{PORT_H}:flags=lanczos[v]",
        "-map", "[v]", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k",
        "-t", f"{out_duration:.3f}",
        portrait_str], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
    if r.returncode != 0:
        log(f"FAIL encode step 1"); return False
    os.remove(portrait_tmp)
    log(f"Step 1 done: {portrait.stat().st_size/(1024*1024):.1f} MB")

    # Step 2: vidstab detect
    trf = f"batch{clip_num}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", portrait_str, "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0: log("FAIL vidstab detect"); return False
    log("Step 2 done")

    # Step 3: vidstab transform — output directly to D: to avoid cross-drive move
    stab_tmp = str((reels_dir / f"{clip_num}__stab_tmp.mp4").resolve())
    stderr_log = os.path.join(TEMP_DIR, f"batch{clip_num}_ffmpeg_stderr.txt")
    with open(stderr_log, "w") as ef:
        r2 = subprocess.run([FFMPEG, "-y", "-i", portrait_str, "-vf",
            f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom={zoom}",
            "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-c:a", "aac", "-b:a", "128k", stab_tmp],
            stdout=subprocess.DEVNULL, stderr=ef, timeout=600, cwd=TEMP_DIR)
    log(f"Step 3 ffmpeg returned {r2.returncode}")
    if r2.returncode != 0:
        # Read last 500 chars of stderr
        try:
            with open(stderr_log) as ef:
                err = ef.read()[-500:]
            log(f"FAIL vidstab transform: {err}")
        except: log("FAIL vidstab transform (no stderr)")
        return False
    log(f"Step 3 done: {os.path.getsize(stab_tmp)/(1024*1024):.1f} MB")

    # Step 4: rename stab to FINAL
    fa = str(final.resolve())
    try:
        if os.path.exists(fa): os.remove(fa)
        os.rename(stab_tmp, fa)
        log("Step 4 done (renamed to FINAL)")
    except Exception as e:
        log(f"Step 4 rename failed: {e}, trying copy")
        shutil.copy2(stab_tmp, fa)
        os.remove(stab_tmp)
        log("Step 4 done (copy+delete)")

    # Cleanup intermediates
    for tmp in [os.path.join(TEMP_DIR, trf)]:
        if os.path.exists(tmp): os.remove(tmp)
    if portrait.exists(): portrait.unlink()
    for dg in reels_dir.glob(f"{clip_num}__*diag*"): dg.unlink()
    for rl in reels_dir.glob(f"{clip_num}__*rerender*"): rl.unlink()

    total = time.time() - t_clip
    fmb = os.path.getsize(fa) / (1024*1024)
    log(f"COMPLETE {clip_num}: {total/60:.1f}m, {fmb:.1f} MB")
    log(f"Saved to: {fa}")
    return True

# --- Main batch loop ---
with open(RESULT_FILE, "w") as f:
    f.write(f"Batch render started {time.strftime('%H:%M:%S')}\n")

t_batch = time.time()
results = {}
for cn in CLIP_NUMS:
    try:
        ok = render_clip(cn)
        results[cn] = "OK" if ok else "FAIL"
    except Exception as e:
        log(f"EXCEPTION {cn}: {e}")
        traceback.print_exc()
        results[cn] = f"ERROR: {e}"
    gc.collect()

log(f"\n=== BATCH SUMMARY ===")
for cn, status in results.items():
    log(f"  {cn}: {status}")
log(f"Total batch time: {(time.time()-t_batch)/60:.1f}m")
log("BATCH DONE")
