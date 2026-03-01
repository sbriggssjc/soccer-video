"""Clip 006 v4 - Direct crop renderer.
Bypasses render_follow_unified entirely.
Reads camera_x_pct anchors, interpolates per-frame, crops portrait directly.
Then vidstab + 4K upscale as before.
"""
import time, os, sys, csv, re, subprocess, traceback, tempfile
import numpy as np

os.chdir(r"D:\Projects\soccer-video")
RESULT = r"D:\Projects\soccer-video\_tmp\render_006_v4_result.txt"
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
TEMP_DIR = tempfile.gettempdir()

GAME = "2026-02-23__TSC_vs_NEOFC"
clip_num = "006"

# Portrait dimensions
PORT_W = 1080
PORT_H = 1920
SRC_W = 1920
SRC_H = 1080

# Output fps (cinematic)
FPS_OUT = 24
FPS_SRC = 30

def log(msg):
    with open(RESULT, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

try:
    with open(RESULT, "w") as f:
        f.write(f"Direct crop render clip {clip_num} v4\n")

    from pathlib import Path
    csv_path = rf"C:\Users\scott\Desktop\review_{clip_num}.csv"
    clips_dir = Path(f"out/atomic_clips/{GAME}")
    clip_file = next(clips_dir.glob(f"{clip_num}__*.mp4"))
    stem = clip_file.stem
    reels_dir = Path(f"out/portrait_reels/{GAME}")
    portrait = reels_dir / f"{stem}__portrait.mp4"
    final = reels_dir / f"{stem}__portrait__FINAL.mp4"

    # Parse duration from filename
    m = re.search(r"t([\d.]+)-t([\d.]+)", stem)
    duration = float(m.group(2)) - float(m.group(1))
    total_src_frames = int(duration * FPS_SRC)
    log(f"Clip: {clip_file.name}")
    log(f"Duration: {duration}s, {total_src_frames} source frames")

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

    # Interpolate camera_x for every source frame
    aframes = sorted(anchors.keys())
    acx_pct = [anchors[f] for f in aframes]
    all_cx_pct = np.interp(np.arange(total_src_frames), aframes, acx_pct)

    # Convert percentage to pixel crop position
    # Portrait crop is PORT_W wide (1080) from a SRC_W (1920) source
    # The crop_x is the LEFT edge of the crop window
    # camera_x_pct is where the CENTER of the crop should be
    half_crop = PORT_W / 2  # 540
    all_cx_px = all_cx_pct / 100.0 * SRC_W
    # Clamp center so crop stays within frame
    all_cx_px = np.clip(all_cx_px, half_crop, SRC_W - half_crop)
    all_crop_x = (all_cx_px - half_crop).astype(int)

    log(f"{len(anchors)} anchors interpolated to {total_src_frames} frames")
    log(f"Crop X range: {all_crop_x.min()}-{all_crop_x.max()}")

    # For the vertical crop: center vertically
    # SRC_H=1080, PORT_H=1920 - we can't get 1920 from 1080!
    # Portrait mode means we crop a tall narrow strip and SCALE it up.
    # Actually: the source is 1920x1080 landscape. Portrait is 1080x1920.
    # The aspect ratio of portrait is 9:16. From 1920x1080 (16:9), 
    # a 9:16 crop would be: height=1080, width=1080*(9/16)=607.5 -> 608px wide
    # Then scale 608x1080 -> 1080x1920
    
    # So actual crop from source: 608 wide x 1080 tall (full height)
    CROP_W = int(SRC_H * PORT_W / PORT_H)  # 1080 * 1080/1920 = 607.5 -> 608
    CROP_H = SRC_H  # 1080 (full height)
    
    half_crop_actual = CROP_W / 2  # 304
    all_cx_px2 = all_cx_pct / 100.0 * SRC_W
    all_cx_px2 = np.clip(all_cx_px2, half_crop_actual, SRC_W - half_crop_actual)
    all_crop_x2 = (all_cx_px2 - half_crop_actual).astype(int)
    
    log(f"Actual crop: {CROP_W}x{CROP_H} from {SRC_W}x{SRC_H}")
    log(f"Crop X range (actual): {all_crop_x2.min()}-{all_crop_x2.max()}")

    t_total = time.time()

    # Step 1: Direct crop with per-frame positioning using sendcmd
    # We'll use ffmpeg's crop filter with per-frame x positions
    # via the sendcmd/commands filter approach, or simpler: use
    # a Python frame-by-frame approach with OpenCV
    
    log("Step 1: Direct crop with OpenCV...")
    
    # Check if cv2 is available
    try:
        import cv2
    except ImportError:
        log("Installing opencv-python...")
        subprocess.run([sys.executable, "-m", "pip", "install", 
            "opencv-python", "--break-system-packages", "-q"],
            capture_output=True, timeout=120)
        import cv2
    
    cap = cv2.VideoCapture(str(clip_file))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {clip_file}")
    
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"Source: {src_fps}fps, {src_frame_count} frames")
    
    # Frame sampling: convert 30fps source to 24fps output
    # For each output frame, pick the nearest source frame
    total_out_frames = int(duration * FPS_OUT)
    out_to_src = np.round(np.linspace(0, total_src_frames - 1, total_out_frames)).astype(int)
    
    # Write portrait frames via OpenCV
    portrait_tmp = os.path.join(TEMP_DIR, f"neofc{clip_num}_portrait_direct.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(portrait_tmp, fourcc, FPS_OUT, (CROP_W, CROP_H))
    
    frame_idx = 0
    out_idx = 0
    frames_written = 0
    
    while out_idx < total_out_frames:
        target_src = out_to_src[out_idx]
        
        # Skip to target frame if needed
        while frame_idx < target_src:
            cap.grab()
            frame_idx += 1
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Crop
        cx = int(all_crop_x2[min(target_src, len(all_crop_x2)-1)])
        cropped = frame[0:CROP_H, cx:cx+CROP_W]
        writer.write(cropped)
        frames_written += 1
        out_idx += 1
    
    cap.release()
    writer.release()
    log(f"Wrote {frames_written} frames to temp portrait")
    
    # Re-encode with ffmpeg for proper codec + add audio
    portrait_str = str(portrait.resolve())
    if portrait.exists(): portrait.unlink()
    
    r = subprocess.run([FFMPEG, "-y",
        "-i", portrait_tmp,
        "-i", str(clip_file),
        "-filter_complex",
        f"[0:v]scale={PORT_W}:{PORT_H}:flags=lanczos[v]",
        "-map", "[v]", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k",
        "-r", str(FPS_OUT),
        portrait_str],
        capture_output=True, text=True, timeout=300)
    
    if r.returncode != 0:
        log(f"FAIL encode: {(r.stderr or '')[-500:]}")
        sys.exit(1)
    
    os.remove(portrait_tmp)
    log(f"Step 1 done: {portrait.stat().st_size/(1024*1024):.1f} MB")

    # Step 2: vidstab detect
    trf = f"neofc{clip_num}_transforms.trf"
    r1 = subprocess.run([FFMPEG, "-y", "-i", portrait_str, "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result='{trf}'",
        "-f", "null", "-"], capture_output=True, text=True, timeout=300, cwd=TEMP_DIR)
    if r1.returncode != 0: log("FAIL vidstab detect"); sys.exit(1)
    log("Step 2 done")

    # Step 3: vidstab transform
    stab_tmp = os.path.join(TEMP_DIR, f"neofc{clip_num}_stab.mp4")
    r2 = subprocess.run([FFMPEG, "-y", "-i", portrait_str, "-vf",
        f"vidstabtransform=input='{trf}':smoothing=15:interpol=bicubic:crop=black:zoom=3",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-c:a", "aac", "-b:a", "128k", stab_tmp],
        capture_output=True, text=True, timeout=600, cwd=TEMP_DIR)
    if r2.returncode != 0: log("FAIL vidstab transform"); sys.exit(1)
    log("Step 3 done")

    # Step 4: 4K upscale
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

    # Clean up intermediates
    if portrait.exists(): portrait.unlink()
    for dg in reels_dir.glob(f"{clip_num}__*diag*"):
        dg.unlink()
    for rl in reels_dir.glob(f"{clip_num}__*rerender*"):
        rl.unlink()

    total = time.time() - t_total
    fmb = os.path.getsize(fa) / (1024*1024)
    log(f"COMPLETE: {total/60:.1f}m, {fmb:.1f} MB")
    log(f"Saved to: {fa}")

except Exception as e:
    with open(RESULT, "a") as f:
        f.write(f"\nEXCEPTION: {e}\n{traceback.format_exc()}")
