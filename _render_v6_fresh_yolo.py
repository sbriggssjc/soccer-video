#!/usr/bin/env python3
"""v6 renderer: Fresh YOLO detection using BallTracker + unrestricted ball-following."""
import sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

CLIP = (REPO / "out" / "atomic_clips" / "2026-02-23__TSC_vs_NEOFC"
        / "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.mp4")
OUT_DIR = REPO / "out" / "portrait_reels" / "2026-02-23__TSC_vs_NEOFC"
OUT_VIDEO = OUT_DIR / "002__fresh_yolo_v6.mp4"
OUT_DIAG  = OUT_DIR / "002__fresh_yolo_v6.diag.csv"

SRC_W, SRC_H = 1920, 1080
CROP_W = 608
OUT_W, OUT_H = 1080, 1920
FPS_OUT = 24
YOLO_MODEL = "yolov8n.pt"
MIN_CONF = 0.10
SMOOTH_SIGMA = 1.5

def run_fresh_yolo(video_path):
    import cv2
    from soccer_highlights.ball_tracker import BallTracker
    model_path = REPO / YOLO_MODEL
    print(f"[YOLO] BallTracker model={model_path.name} conf>={MIN_CONF} imgsz=1920")
    tracker = BallTracker(weights_path=model_path, min_conf=MIN_CONF,
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
    elapsed = time.time() - t0
    pct = 100*len(detections)/n_frames if n_frames else 0
    print(f"[YOLO] Done: {len(detections)}/{n_frames} ({pct:.1f}%) in {elapsed:.1f}s")
    return detections, n_frames

def estimate_camera_motion(video_path, n_frames):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    print(f"[CAM] Estimating camera motion for {n_frames} frames...")
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

def build_crop_path(detections, n_frames, cam_cum_x, cam_cum_y):
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter1d, median_filter
    fi_list, wx_list, wy_list = [], [], []
    for fi in sorted(detections.keys()):
        cx, cy, conf = detections[fi]
        fi_list.append(fi)
        wx_list.append(cx + cam_cum_x[fi])
        wy_list.append(cy + cam_cum_y[fi])
    fi_arr = np.array(fi_list); wx = np.array(wx_list); wy = np.array(wy_list)
    print(f"[PATH] {len(fi_arr)} anchors, world X: {wx.min():.0f}-{wx.max():.0f}")
    if len(fi_arr) > 10:
        wxm = median_filter(wx, size=min(7,len(wx)))
        wym = median_filter(wy, size=min(7,len(wy)))
        tx = max(np.std(np.abs(wx-wxm))*3, 100)
        ty = max(np.std(np.abs(wy-wym))*3, 100)
        mask = (np.abs(wx-wxm)<tx)&(np.abs(wy-wym)<ty)
        nr = np.sum(~mask)
        if nr: print(f"[PATH] Removed {nr} outliers")
        fi_arr=fi_arr[mask]; wx=wx[mask]; wy=wy[mask]
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
    print(f"[RENDER] Done in {elapsed:.1f}s")
    print(f"[RENDER] BIC: {bic}/{td} ({bp:.1f}%)")
    print(f"[RENDER] Output: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    with open(diag_path, "w") as f: f.write("\n".join(diag))

def main():
    print("="*70)
    print("v6 FRESH YOLO RENDERER (BallTracker)")
    print(f"  Model: {YOLO_MODEL} @ 1920px | conf>={MIN_CONF}")
    print(f"  Smoothing: sigma={SMOOTH_SIGMA} | Speed limits: NONE")
    print("="*70)
    detections, n_frames = run_fresh_yolo(CLIP)
    cam_cx, cam_cy = estimate_camera_motion(CLIP, n_frames)
    crop_cx = build_crop_path(detections, n_frames, cam_cx, cam_cy)
    render_portrait(CLIP, crop_cx, detections, OUT_VIDEO, OUT_DIAG)
    print("\n" + "="*70 + "\nCOMPLETE\n" + "="*70)

if __name__ == "__main__":
    main()
