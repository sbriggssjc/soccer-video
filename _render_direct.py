"""Direct crop renderer - reusable for any clip.
Bypasses render_follow_unified entirely.
Reads camera_x_pct anchors, interpolates per-frame, crops portrait directly.
Then vidstab + 4K upscale.
"""
import time, os, sys, csv, re, subprocess, traceback, tempfile
import numpy as np

CLIP_NUM = "016"
SKIP_4K_UPSCALE = True  # Stay at 1080x1920 for cleaner image

os.chdir(r"D:\Projects\soccer-video")
RESULT = rf"D:\Projects\soccer-video\_tmp\render_direct_{CLIP_NUM}_result.txt"
FFMPEG = "ffmpeg"
TEMP_DIR = tempfile.gettempdir()
GAME = "2026-02-23__TSC_vs_NEOFC"

SRC_W = 1920; SRC_H = 1080
PORT_W = 1080; PORT_H = 1920
FPS_OUT = 24; FPS_SRC = 30

def log(msg):
    with open(RESULT, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

try:
    with open(RESULT, "w") as f:
        f.write(f"Direct crop render clip {CLIP_NUM}\n")

    from pathlib import Path
    csv_path = rf"C:\Users\scott\Desktop\review_{CLIP_NUM}.csv"
    clips_dir = Path(f"out/atomic_clips/{GAME}")
    clip_file = next(clips_dir.glob(f"{CLIP_NUM}__*.mp4"))
    stem = clip_file.stem
    reels_dir = Path(f"out/portrait_reels/{GAME}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"

    # Use ffprobe for actual duration/frame count (filename can be inaccurate)
    probe = subprocess.run([FFMPEG.replace("ffmpeg","ffprobe"), "-v", "quiet",
        "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,duration,r_frame_rate",
        "-print_format", "csv=p=0", str(clip_file)],
        capture_output=True, text=True, timeout=60)
    probe_parts = probe.stdout.strip().split(",")
    # fallback to filename if probe fails
    m = re.search(r"t([\d.]+)-t([\d.]+)", stem)
    fname_duration = float(m.group(2)) - float(m.group(1))
    
    import cv2
    cap_probe = cv2.VideoCapture(str(clip_file))
    actual_frame_count = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap_probe.get(cv2.CAP_PROP_FPS)
    cap_probe.release()
    
    total_src_frames = actual_frame_count
    duration = total_src_frames / FPS_SRC
    log(f"Clip: {clip_file.name}")
    log(f"Actual: {total_src_frames} frames, {duration:.2f}s (filename said {fname_duration}s)")

    # Read anchors
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    anchors = {}
    for r in rows:
        val = r["camera_x_pct"].strip()
        if val:
            anchors[int(r["frame"])] = float(val)

    mx = max(anchors.keys())
    if mx < total_src_frames - 1:
        anchors[total_src_frames - 1] = anchors[mx]

    aframes = sorted(anchors.keys())
    acx_pct = [anchors[f] for f in aframes]
    all_cx_pct = np.interp(np.arange(total_src_frames), aframes, acx_pct)

    # Crop dimensions: 9:16 from 16:9 source (must be even for codec)
    CROP_W = int(SRC_H * PORT_W / PORT_H)  # 607
    CROP_W = CROP_W + (CROP_W % 2)  # round up to even -> 608
    CROP_H = SRC_H  # 1080
    half_crop = CROP_W / 2

    all_cx_px = all_cx_pct / 100.0 * SRC_W
    all_cx_px = np.clip(all_cx_px, half_crop, SRC_W - half_crop)
    all_crop_x = (all_cx_px - half_crop).astype(int)

    log(f"{len(anchors)} anchors, crop {CROP_W}x{CROP_H}")

    t_total = time.time()

    # Step 1: Direct crop - read with OpenCV, pipe to ffmpeg
    log("Step 1: Direct crop...")
    cap = cv2.VideoCapture(str(clip_file))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"Source dims: {actual_w}x{actual_h}")
    
    # Recalc crop if source isn't exactly 1920x1080
    if actual_w != SRC_W or actual_h != SRC_H:
        CROP_W = int(actual_h * PORT_W / PORT_H)
        CROP_W = CROP_W + (CROP_W % 2)
        CROP_H = actual_h
        half_crop = CROP_W / 2
        all_cx_px = all_cx_pct / 100.0 * actual_w
        all_cx_px = np.clip(all_cx_px, half_crop, actual_w - half_crop)
        all_crop_x = (all_cx_px - half_crop).astype(int)
        log(f"Recalculated crop: {CROP_W}x{CROP_H}")

    total_out_frames = int(duration * FPS_OUT)
    out_to_src = np.round(np.linspace(0, total_src_frames - 1, total_out_frames)).astype(int)

    portrait_tmp = os.path.join(TEMP_DIR, f"neofc{CLIP_NUM}_portrait_direct.mp4")
    
    # Pipe raw frames to ffmpeg for reliable encoding
    ffmpeg_proc = subprocess.Popen([
        FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{CROP_W}x{CROP_H}", "-r", str(FPS_OUT),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "17",
        "-pix_fmt", "yuv420p", portrait_tmp
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame_idx = 0
    out_idx = 0
    frames_written = 0
    while out_idx < total_out_frames:
        target_src = out_to_src[out_idx]
        while frame_idx < target_src:
            cap.grab()
            frame_idx += 1
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        cx = int(all_crop_x[min(target_src, len(all_crop_x)-1)])
        cx = max(0, min(cx, actual_w - CROP_W))  # safety clamp
        cropped = frame[0:CROP_H, cx:cx+CROP_W]
        ffmpeg_proc.stdin.write(cropped.tobytes())
        frames_written += 1
        out_idx += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait(timeout=600)
    log(f"Wrote {frames_written} frames")

    # Scale to portrait dimensions + add audio (trimmed to match video)
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
        portrait_str], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        log(f"FAIL encode: {(r.stderr or '')[-500:]}"); sys.exit(1)
    os.remove(portrait_tmp)
    log(f"Step 1 done: {portrait.stat().st_size/(1024*1024):.1f} MB")

    # Step 2: vidstab detect
    trf = f"neofc{CLIP_NUM}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", portrait_str, "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0: log("FAIL vidstab detect"); sys.exit(1)
    log("Step 2 done")

    # Step 3: vidstab transform
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{CLIP_NUM}_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", portrait_str, "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0: log("FAIL vidstab transform"); sys.exit(1)
    log("Step 3 done")

    # Step 4: 4K upscale (optional)
    fa = str(final.resolve())
    if os.path.exists(fa): os.remove(fa)
    if not SKIP_4K_UPSCALE:
        r3 = subprocess.run([FFMPEG, "-y", "-i", stab_tmp, "-vf",
            "scale=iw*2:ih*2:flags=lanczos,hqdn3d=2:1:2:3,unsharp=5:5:0.5:5:5:0.0",
            "-c:v", "libx264", "-preset", "slow", "-crf", "17",
            "-c:a", "copy", fa], capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
        if r3.returncode != 0: log("FAIL 4K upscale"); sys.exit(1)
        log("Step 4 done")
    else:
        import shutil
        shutil.move(stab_tmp, fa)
        stab_tmp = None
        log("Step 4 skipped (no 4K upscale)")

    for tmp in [os.path.join(TEMP_DIR, trf), stab_tmp]:
        if tmp and os.path.exists(tmp): os.remove(tmp)

    # Clean up intermediates
    if portrait.exists(): portrait.unlink()
    for dg in reels_dir.glob(f"{CLIP_NUM}__*diag*"): dg.unlink()
    for rl in reels_dir.glob(f"{CLIP_NUM}__*rerender*"): rl.unlink()

    total = time.time() - t_total
    fmb = os.path.getsize(fa) / (1024*1024)
    log(f"COMPLETE: {total/60:.1f}m, {fmb:.1f} MB")
    log(f"Saved to: {fa}")

except Exception as e:
    with open(RESULT, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
