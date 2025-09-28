import pandas as pd, numpy as np, cv2, os, subprocess, argparse
from scipy.signal import savgol_filter

def smooth_series(arr, fps):
    n = len(arr)
    if n < 5: return arr
    win = max(5, int(round(fps*0.5)))
    if win % 2 == 0: win += 1
    if win > n: win = (n if n % 2 == 1 else n-1)
    if win < 5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)
    # same “best” settings you liked:
    ap.add_argument("--tau", type=float, default=0.24)          # seconds look-ahead
    ap.add_argument("--slew", type=float, default=280.0)        # px/s
    ap.add_argument("--accel", type=float, default=1200.0)      # px/s^2
    ap.add_argument("--left_frac", type=float, default=0.45)    # ball position in frame (0..1 from left)
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)
    args = ap.parse_args()

    clip = args.clip
    track_csv = args.track_csv
    out_mp4 = args.out_mp4

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # maintain 608x1080 aspect for vertical
    crop_w = int(np.floor(H * (args.W_out/args.H_out) / 2) * 2)
    crop_w = min(crop_w, W)

    df = pd.read_csv(track_csv)

    # fill small gaps then smooth
    for col in ["cx","cy"]:
        s = pd.to_numeric(df[col], errors="coerce").ffill(limit=10).bfill(limit=10)
        df[col] = smooth_series(s.to_numpy(), fps)

    cx = pd.to_numeric(df["cx"], errors="coerce").to_numpy()
    if np.isnan(cx).any():
        idx = np.where(~np.isnan(cx))[0]
        cx = np.interp(np.arange(len(cx)), idx, cx[idx])

    vx = np.gradient(cx) * fps
    lead_cx = cx + args.tau * vx

    target = np.clip(lead_cx - args.left_frac*crop_w, 0, W - crop_w)

    x = np.zeros_like(target, dtype=float)
    x[0] = target[0]
    v = 0.0
    dt = 1.0 / max(fps, 1e-6)

    for i in range(1, len(target)):
        err   = target[i] - x[i-1]
        v_des = np.clip(err/dt, -args.slew, args.slew)
        dv    = np.clip(v_des - v, -args.accel*dt, args.accel*dt)
        v     = np.clip(v + dv, -args.slew, args.slew)
        x[i]  = float(np.clip(x[i-1] + v*dt, 0, W - crop_w))

    # optional debug
    dbg_dir = os.path.join(os.path.dirname(out_mp4) or ".", "..", "autoframe_work")
    os.makedirs(dbg_dir, exist_ok=True)
    pd.DataFrame({"frame":df["frame"], "t":df["time"], "cx":cx, "x":x}).to_csv(
        os.path.join(dbg_dir, os.path.splitext(os.path.basename(out_mp4))[0] + "_virtual_cam.csv"),
        index=False
    )

    # render per frame
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    tmp = os.path.join(dbg_dir, "_temp_frames_balltrack")
    os.makedirs(tmp, exist_ok=True)

    cap = cv2.VideoCapture(clip)
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        xi = int(round(x[i] if i < len(x) else x[-1]))
        xi = max(0, min(xi, W - crop_w))
        crop = bgr[:, xi:xi+int(crop_w)]
        crop = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp, f"f_{i:06d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1
    cap.release()

    subprocess.run([
        "ffmpeg","-y",
        "-framerate", str(int(round(fps))),
        "-i", os.path.join(tmp, "f_%06d.jpg"),
        "-i", clip,
        "-map","0:v","-map","1:a:0?",
        "-c:v","libx264","-preset","veryfast","-crf","19",
        "-x264-params","keyint=120:min-keyint=120:scenecut=0",
        "-pix_fmt","yuv420p","-profile:v","high","-level","4.0",
        "-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709",
        "-shortest","-movflags","+faststart",
        "-c:a","aac","-b:a","128k",
        out_mp4
    ], check=True)

    print("Wrote", out_mp4)

if __name__ == "__main__":
    main()
