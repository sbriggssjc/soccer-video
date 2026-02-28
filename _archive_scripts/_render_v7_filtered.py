#!/usr/bin/env python3
"""v7 renderer: Two-pass filtered trajectory + unrestricted ball-following.

Pass 1: Run YOLO detection (reuse v6 data)
Pass 2: Filter detections using trajectory coherence in world-space
  - Establish main trajectory using high-confidence detections only
  - Then include lower-conf detections ONLY if they're near the main trajectory
  - Remove isolated false positives that jump far from the trajectory
"""
import sys, time, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

CLIP = (REPO / "out" / "atomic_clips" / "2026-02-23__TSC_vs_NEOFC"
        / "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4")
OUT_DIR = REPO / "out" / "portrait_reels" / "2026-02-23__TSC_vs_NEOFC"
OUT_VIDEO = OUT_DIR / "002__filtered_v7.mp4"
OUT_DIAG  = OUT_DIR / "002__filtered_v7.diag.csv"

SRC_W, SRC_H = 1920, 1080
CROP_W = 608
OUT_W, OUT_H = 1080, 1920
FPS_OUT = 24
SMOOTH_SIGMA = 2.0  # slightly more smoothing for visual comfort

YOLO_CACHE = REPO / "_v7_yolo_cache.json"

def run_fresh_yolo(video_path):
    """Detect ball using BallTracker at full resolution. Caches results."""
    import cv2, json as _json

    # Check cache first
    if YOLO_CACHE.is_file():
        with open(YOLO_CACHE) as f:
            data = _json.load(f)
        detections = {int(k): tuple(v) for k, v in data["detections"].items()}
        n_frames = data["n_frames"]
        print(f"[YOLO] Loaded {len(detections)} cached detections from {YOLO_CACHE.name}")
        return detections, n_frames

    from soccer_highlights.ball_tracker import BallTracker
    model_path = REPO / "yolov8n.pt"
    print(f"[YOLO] BallTracker model=yolov8n.pt conf>=0.10 imgsz=1920")
    tracker = BallTracker(weights_path=model_path, min_conf=0.10,
                          device="cpu", input_size=1920, smooth_alpha=0.25, max_gap=12)
    if not tracker.is_ready:
        print(f"[YOLO] Not ready: {tracker.failure_reason}"); return {}, 0
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[YOLO] Scanning {n_frames} frames...")
    detections = {}
    t0 = time.time()
    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret: break
        track = tracker.update(fi, frame)
        if track is not None:
            detections[fi] = (track.raw_cx, track.raw_cy, track.raw_conf)
        if fi % 50 == 0:
            elapsed = time.time() - t0
            rate = (fi+1)/elapsed if elapsed > 0 else 0
            print(f"  f{fi}/{n_frames} | {len(detections)} dets | {rate:.1f} fps")
    cap.release()
    pct = 100*len(detections)/n_frames if n_frames else 0
    print(f"[YOLO] Done: {len(detections)}/{n_frames} ({pct:.1f}%) in {time.time()-t0:.1f}s")
    # Save cache
    with open(YOLO_CACHE, "w") as f:
        _json.dump({"n_frames": n_frames,
                    "detections": {str(k): list(v) for k, v in detections.items()}}, f)
    print(f"[YOLO] Cached to {YOLO_CACHE.name}")
    return detections, n_frames

def estimate_camera_motion(video_path, n_frames):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    print(f"[CAM] Estimating camera motion...")
    fp = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
    lk = dict(winSize=(21,21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    cam_dx = np.zeros(n_frames, dtype=np.float64)
    cam_dy = np.zeros(n_frames, dtype=np.float64)
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
                    cam_dx[fi] = M[0,2]; cam_dy[fi] = M[1,2]
        prev_g = g
    cap.release()
    cx = np.cumsum(cam_dx); cy = np.cumsum(cam_dy)
    print(f"[CAM] Total pan: dx={cx[-1]:.0f}px, dy={cy[-1]:.0f}px")
    return cx, cy

def filter_trajectory(detections, n_frames, cam_cum_x, cam_cum_y):
    """Two-pass filter: establish main trajectory, then remove outliers.
    
    Pass 1: Use only HIGH confidence detections (>=0.20) to establish 
            the 'spine' of the ball trajectory in world-space.
    Pass 2: Include lower-confidence detections ONLY if they're within
            a reasonable distance of the spine trajectory.
    """
    from scipy.interpolate import PchipInterpolator
    import math

    # Convert ALL detections to world-space
    all_dets = {}
    for fi in sorted(detections.keys()):
        cx, cy, conf = detections[fi]
        wx = cx + cam_cum_x[fi]
        wy = cy + cam_cum_y[fi]
        all_dets[fi] = (wx, wy, conf)

    # Pass 1: High-confidence spine (conf >= 0.20)
    HIGH_CONF = 0.20
    spine_frames = []
    spine_wx = []
    spine_wy = []
    for fi in sorted(all_dets.keys()):
        wx, wy, conf = all_dets[fi]
        if conf >= HIGH_CONF:
            spine_frames.append(fi)
            spine_wx.append(wx)
            spine_wy.append(wy)

    spine_frames = np.array(spine_frames)
    spine_wx = np.array(spine_wx)
    spine_wy = np.array(spine_wy)
    print(f"[FILTER] Pass 1: {len(spine_frames)} high-conf (>={HIGH_CONF}) detections")

    if len(spine_frames) < 3:
        print("[FILTER] Too few high-conf detections, using all")
        return detections

    # Build spine trajectory via PCHIP
    spine_interp_x = PchipInterpolator(spine_frames, spine_wx, extrapolate=True)
    spine_interp_y = PchipInterpolator(spine_frames, spine_wy, extrapolate=True)

    # Pass 2: Include low-conf detections only if near spine
    MAX_DIST = 200  # pixels in world-space - ball shouldn't be >200px from spine
    filtered = {}
    n_kept_low = 0
    n_rejected = 0

    for fi in sorted(all_dets.keys()):
        wx, wy, conf = all_dets[fi]
        if conf >= HIGH_CONF:
            # Always keep high-conf
            filtered[fi] = detections[fi]
        else:
            # Check distance from spine
            expected_wx = float(spine_interp_x(fi))
            expected_wy = float(spine_interp_y(fi))
            dist = math.hypot(wx - expected_wx, wy - expected_wy)
            if dist <= MAX_DIST:
                filtered[fi] = detections[fi]
                n_kept_low += 1
            else:
                n_rejected += 1

    print(f"[FILTER] Pass 2: kept {len(filtered)} total "
          f"({len(spine_frames)} high + {n_kept_low} low-conf near spine)")
    print(f"[FILTER] Rejected {n_rejected} false detections")
    return filtered

def build_crop_path(detections, n_frames, cam_cum_x, cam_cum_y):
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d
    fi_list, wx_list = [], []
    for fi in sorted(detections.keys()):
        cx, cy, conf = detections[fi]
        fi_list.append(fi)
        wx_list.append(cx + cam_cum_x[fi])
    fi_arr = np.array(fi_list); wx = np.array(wx_list)
    print(f"[PATH] {len(fi_arr)} anchors, world X: {wx.min():.0f}-{wx.max():.0f}")
    if len(fi_arr) < 2:
        print("[PATH] ERROR: <2 detections"); return np.full(n_frames, SRC_W/2)
    allf = np.arange(n_frames)
    path_wx = PchipInterpolator(fi_arr, wx, extrapolate=True)(allf)
    path_fx = path_wx - cam_cum_x
    path_fx = gaussian_filter1d(path_fx, sigma=SMOOTH_SIGMA)
    hw = CROP_W/2
    path_fx = np.clip(path_fx, hw, SRC_W - hw)
    # gaps
    gaps = []
    prev = -1
    for fi in sorted(fi_arr):
        if prev >= 0 and fi-prev > 5: gaps.append((prev,fi,fi-prev))
        prev = fi
    gaps.sort(key=lambda g: g[2], reverse=True)
    print("[PATH] Largest gaps:")
    for s,e,l in gaps[:5]: print(f"  f{s}->f{e}: {l} frames ({l/30:.1f}s)")
    return path_fx

def render_portrait(video_path, crop_cx, detections, out_path, diag_path):
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
    fb = SRC_W*SRC_H*3; t0=time.time(); bic=0; td=0
    diag = ["frame,crop_cx,yolo_x,yolo_conf,ball_in_crop"]
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
        yx = yc = None; ic = ""
        if fi in detections:
            td += 1; yx = detections[fi][0]; yc = detections[fi][2]
            if x0 <= yx <= x1: bic += 1; ic = "Y"
            else: ic = "N"
        diag.append(f"{fi},{cx},{yx or ''},{yc or ''},{ic}")
        if fi % 100 == 0: print(f"  f{fi}/{n_frames} | {time.time()-t0:.1f}s")
    dec.stdout.close(); enc.stdin.close(); dec.wait(); enc.wait()
    elapsed = time.time()-t0
    bp = 100*bic/td if td else 0
    print(f"[RENDER] Done in {elapsed:.1f}s | BIC: {bic}/{td} ({bp:.1f}%)")
    print(f"[RENDER] Output: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    with open(diag_path, "w") as f: f.write("\n".join(diag))

def main():
    print("="*70)
    print("v7 FILTERED TRAJECTORY RENDERER")
    print(f"  Two-pass filter: high-conf spine + proximity check")
    print(f"  Smoothing: sigma={SMOOTH_SIGMA} | Speed limits: NONE")
    print("="*70)

    detections, n_frames = run_fresh_yolo(CLIP)
    cam_cx, cam_cy = estimate_camera_motion(CLIP, n_frames)

    # Two-pass filtering
    filtered = filter_trajectory(detections, n_frames, cam_cx, cam_cy)

    crop_cx = build_crop_path(filtered, n_frames, cam_cx, cam_cy)
    render_portrait(CLIP, crop_cx, filtered, OUT_VIDEO, OUT_DIAG)
    print("\n" + "="*70 + "\nCOMPLETE\n" + "="*70)

if __name__ == "__main__":
    main()
