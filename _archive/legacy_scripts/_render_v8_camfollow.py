#!/usr/bin/env python3
"""v8 renderer: Camera-motion-based framing.

KEY INSIGHT: The broadcast cameraman IS tracking the ball. The camera pan
direction and speed is the strongest signal for where the ball is.

Strategy:
- Estimate camera velocity per frame (how fast is the camera panning?)
- When camera pans right, the ball is right-of-center (and vice versa)
- The crop leads slightly in the direction of camera motion
- Center of the frame is the default position (cameraman centers the ball)
- Only use VERY high confidence YOLO (>=0.40) as gentle nudges, not drivers

This avoids all the false-positive YOLO problems by treating camera motion
as the primary signal.
"""
import sys, time, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

CLIP = (REPO / "out" / "atomic_clips" / "2026-02-23__TSC_vs_NEOFC"
        / "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4")
OUT_DIR = REPO / "out" / "portrait_reels" / "2026-02-23__TSC_vs_NEOFC"
OUT_VIDEO = OUT_DIR / "002__camfollow_v8.mp4"
OUT_DIAG  = OUT_DIR / "002__camfollow_v8.diag.csv"

SRC_W, SRC_H = 1920, 1080
CROP_W = 608
OUT_W, OUT_H = 1080, 1920
FPS_OUT = 24
FRAME_CENTER = SRC_W / 2  # 960

def estimate_camera_motion(video_path):
    """Get per-frame camera displacement and velocity."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[CAM] Estimating camera motion for {n_frames} frames...")

    fp = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
    lk = dict(winSize=(21,21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    cam_dx = np.zeros(n_frames, dtype=np.float64)
    ret, prev = cap.read()
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    for fi in range(1, n_frames):
        ret, frame = cap.read()
        if not ret: break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(prev_g, **fp)
        if pts is not None and len(pts) >= 10:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, pts, None, **lk)
            gp = pts[st.ravel()==1]; gn = nxt[st.ravel()==1]
            if len(gp) >= 6:
                M, _ = cv2.estimateAffinePartial2D(gp, gn, method=cv2.RANSAC)
                if M is not None:
                    cam_dx[fi] = M[0, 2]
        prev_g = g
    cap.release()

    cam_cum_x = np.cumsum(cam_dx)
    print(f"[CAM] Total pan: {cam_cum_x[-1]:.0f}px")
    print(f"[CAM] Per-frame dx range: {cam_dx.min():.1f} to {cam_dx.max():.1f}")
    return n_frames, cam_dx, cam_cum_x

def build_camera_following_path(n_frames, cam_dx):
    """Build crop path based on camera motion.

    The cameraman centers the ball in the broadcast frame. When the camera
    pans, it means the ball has moved and the cameraman is catching up.
    During a pan, the ball is AHEAD of center in the pan direction.

    We offset the crop center from frame center based on:
    1. Camera velocity (how fast is it panning?) - more pan = more offset
    2. Smoothed velocity to avoid jitter

    The offset formula: crop_x = frame_center + velocity_lead
    where velocity_lead pushes the crop in the direction of camera pan
    """
    from scipy.ndimage import gaussian_filter1d

    # Smooth the camera velocity to get a clean pan signal
    # sigma=5 (~0.17s at 30fps) removes jitter but preserves pan direction
    smooth_dx = gaussian_filter1d(cam_dx, sigma=5.0)

    # The "lead" factor: how many pixels to offset per px/frame of velocity
    # At typical pan speed of ~7 px/frame, this gives ~140px offset (moderate)
    LEAD_FACTOR = 20.0

    # Calculate crop center: frame center + velocity-based lead
    crop_cx = np.full(n_frames, FRAME_CENTER)
    velocity_lead = smooth_dx * LEAD_FACTOR
    # Clamp the lead to prevent extreme offsets
    velocity_lead = np.clip(velocity_lead, -350, 350)
    crop_cx += velocity_lead

    # Smooth the final path to avoid any remaining jitter
    crop_cx = gaussian_filter1d(crop_cx, sigma=3.0)

    # Clamp to valid crop range
    hw = CROP_W / 2
    crop_cx = np.clip(crop_cx, hw, SRC_W - hw)

    print(f"[PATH] Crop center range: {crop_cx.min():.0f} - {crop_cx.max():.0f}")
    print(f"[PATH] Velocity lead range: {velocity_lead.min():.0f} to {velocity_lead.max():.0f}")

    return crop_cx

def render_portrait(video_path, crop_cx, out_path, diag_path):
    import subprocess, cv2
    n_frames = len(crop_cx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dec_cmd = ["ffmpeg","-hide_banner","-loglevel","error",
               "-i",str(video_path),"-f","rawvideo","-pix_fmt","rgb24","-"]
    enc_cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
               "-f","rawvideo","-pix_fmt","rgb24",
               "-s",f"{OUT_W}x{OUT_H}","-r",str(FPS_OUT),"-i","-",
               "-c:v","libx264","-preset","fast","-crf","20",
               "-pix_fmt","yuv420p",str(out_path)]
    dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, bufsize=SRC_W*SRC_H*3*2)
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=OUT_W*OUT_H*3*2)
    fb = SRC_W*SRC_H*3; t0 = time.time()
    diag = ["frame,crop_cx"]
    print(f"[RENDER] Rendering {n_frames} frames -> {out_path.name}")
    for fi in range(n_frames):
        raw = dec.stdout.read(fb)
        if len(raw) < fb: break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(SRC_H, SRC_W, 3)
        cx = int(round(crop_cx[fi]))
        x0 = cx - CROP_W//2; x1 = x0 + CROP_W
        crop = frame[:, x0:x1, :]
        scaled = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4)
        enc.stdin.write(scaled.tobytes())
        diag.append(f"{fi},{cx}")
        if fi % 100 == 0: print(f"  f{fi}/{n_frames} | {time.time()-t0:.1f}s")
    dec.stdout.close(); enc.stdin.close(); dec.wait(); enc.wait()
    print(f"[RENDER] Done in {time.time()-t0:.1f}s")
    print(f"[RENDER] Output: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    with open(diag_path, "w") as f: f.write("\n".join(diag))

def main():
    print("="*70)
    print("v8 CAMERA-FOLLOWING RENDERER")
    print("  Primary signal: camera pan velocity (cameraman tracks the ball)")
    print("  No YOLO dependency - avoids all false detection issues")
    print("="*70)
    n_frames, cam_dx, cam_cum_x = estimate_camera_motion(CLIP)
    crop_cx = build_camera_following_path(n_frames, cam_dx)
    render_portrait(CLIP, crop_cx, OUT_VIDEO, OUT_DIAG)
    print("\n" + "="*70 + "\nCOMPLETE\n" + "="*70)

if __name__ == "__main__":
    main()
