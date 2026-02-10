# 03_smart_shrink.py
# Finds peak action within coarse highlights and (optionally) writes
# short, tracked clips for social media.
#
# Usage example (vertical social):
#   python 03_smart_shrink.py --video .\full_game_stabilized.mp4 ^
#       --csv .\out\highlights.csv --outcsv .\out\highlights_smart.csv ^
#       --aspect vertical --write-clips .\out\smart_vertical --pre 3 --post 5 --bias-blue

import os, csv, argparse, math
import numpy as np
import cv2

# Optional audio (helps pick the exact peak). Falls back gracefully.
def audio_envelope(video_path):
    try:
        import librosa
        y, sr = librosa.load(video_path, sr=None, mono=True)
        # onset strength is a decent "excitement" proxy
        hop = 512
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
        env = env.astype(np.float32)
        if env.size:
            env = (env - env.min()) / (env.max() - env.min() + 1e-8)
        return t, env
    except Exception:
        return None, None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def blue_mask_small(bgr_small):
    # HSV mask for blue jerseys; adjust if your blue shifts.
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    # Two ranges for robust "blue"
    lower1 = np.array([90,  40,  40], np.uint8)
    upper1 = np.array([130,255,255], np.uint8)
    m = cv2.inRange(hsv, lower1, upper1).astype(np.float32) / 255.0
    # smooth a bit
    if m.any():
        m = cv2.GaussianBlur(m, (9,9), 0)
    return m

def motion_timeseries(cap, f0, f1, scale=0.5, use_blue=False):
    """Return per-frame motion energy and per-frame timestamps (sec) between [f0, f1)."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sw, sh = int(W*scale), int(H*scale)

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, prev = cap.read()
    if not ok: 
        return np.array([]), np.array([])
    prev_small = cv2.resize(prev, (sw, sh))
    prev_gray  = cv2.GaussianBlur(cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY),(5,5),0)

    energies = []
    times = []
    blue_w = blue_mask_small(prev_small) if use_blue else None

    for f in range(f0+1, f1):
        ok, frame = cap.read()
        if not ok: break
        sm = cv2.resize(frame, (sw, sh))
        gray = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY),(5,5),0)
        diff = cv2.absdiff(gray, prev_gray).astype(np.float32)

        base = diff.sum()
        if use_blue:
            # weight motion under blue mask a bit higher so we bias toward your team’s play
            m = blue_mask_small(sm)
            if blue_w is None: blue_w = m
            w = 0.65*base + 0.35*(diff*m).sum()
        else:
            w = base

        energies.append(w)
        times.append((f)/fps)
        prev_gray = gray

    e = np.array(energies, dtype=np.float32)
    t = np.array(times, dtype=np.float32)
    if e.size:
        # light smoothing
        k = 9
        ker = np.ones(k, np.float32)/k
        e = np.convolve(e, ker, mode='same')
        e = (e - e.min())/(e.max()-e.min()+1e-8)
    return e, t

def find_peak_time(cap, start_s, end_s, audio_t=None, audio_env=None, use_blue=False):
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = clamp(int(start_s * fps), 0, total_frames-2)
    f1 = clamp(int(end_s   * fps), f0+1, total_frames-1)

    motion, tf = motion_timeseries(cap, f0, f1, scale=0.5, use_blue=use_blue)
    if motion.size == 0: 
        return (start_s + end_s)/2.0

    score = motion.copy()
    if audio_t is not None and audio_env is not None and audio_env.size and tf.size:
        # Blend in audio excitement (25%)
        ae = np.interp(tf, audio_t, audio_env)
        ae = (ae - ae.min())/(ae.max()-ae.min()+1e-8)
        score = 0.75*motion + 0.25*ae

    idx = int(np.argmax(score))
    return float(tf[idx])

def tracked_crop_writer(out_path, frames_iter, W, H, aspect='vertical', zoom=1.0, fps=24.0):
    """
    Tracks action with motion centroid and writes a cropped clip.
    aspect: 'vertical' makes 9:16 1080x1920; 'horizontal' crops tighter 16:9 and writes 1920x1080.
    zoom: for horizontal only, 1.0 = no crop, 1.4 = crop to 1371x771 then upscale back to 1920x1080
    """
    if aspect == 'vertical':
        out_w, out_h = 1080, 1920
        crop_h = H  # use full height
        crop_w = int(round(crop_h * 9/16))
        crop_w = min(crop_w, W)
    else:
        out_w, out_h = 1920, 1080
        # compute crop rect inverse of zoom
        crop_w = int(round(W / zoom))
        crop_h = int(round(H / zoom))
        crop_w = max(640, min(W, crop_w))
        crop_h = max(360, min(H, crop_h))

    # OpenCV writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    # tracker state (EMA)
    cx, cy = W//2, H//2
    alpha = 0.12

    # scaled working size for motion centroid
    sw, sh = W//2, H//2

    ok_first, prev_small = False, None

    for frame in frames_iter:
        # compute motion centroid on downscaled frames
        small = cv2.resize(frame, (sw, sh))
        gray  = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),(5,5),0)
        if not ok_first:
            prev_small = gray
            ok_first = True
        diff = cv2.absdiff(gray, prev_small)
        prev_small = gray

        # threshold + contour
        thr = max(10, int(diff.mean() + 2*diff.std()))
        _, binm = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        binm = cv2.dilate(binm, np.ones((7,7),np.uint8), iterations=1)

        # centroid
        ys, xs = np.where(binm>0)
        if xs.size:
            cx_s = np.mean(xs)
            cy_s = np.mean(ys)
            # map back to full-res
            cx = int((1-alpha)*cx + alpha* (cx_s * (W/float(sw))))
            cy = int((1-alpha)*cy + alpha* (cy_s * (H/float(sh))))
        else:
            # drift slowly toward center if no motion detected
            cx = int((1-alpha)*cx + alpha*(W//2))
            cy = int((1-alpha)*cy + alpha*(H//2))

        # crop rect centered on (cx, cy)
        x0 = int(round(cx - crop_w/2))
        y0 = int(round(cy - crop_h/2))
        x0 = clamp(x0, 0, W - crop_w)
        y0 = clamp(y0, 0, H - crop_h)
        crop = frame[y0:y0+crop_h, x0:x0+crop_w]

        # resize to output
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        vw.write(resized)

    vw.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--csv', required=True, help="input highlights.csv (start,end,score)")
    ap.add_argument('--outcsv', required=True, help="refined csv to write (start,end,score)")
    ap.add_argument('--pre', type=float, default=3.0)
    ap.add_argument('--post', type=float, default=5.0)
    ap.add_argument('--aspect', choices=['vertical','horizontal'], default='vertical')
    ap.add_argument('--zoom', type=float, default=1.3, help="horizontal only: crop&zoom factor, e.g. 1.3-1.8")
    ap.add_argument('--bias-blue', action='store_true', help="prefer motion where blue jerseys are")
    ap.add_argument('--write-clips', default=None, help="if set, write tracked clips here")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_dur = total_frames / float(fps)

    at, aenv = audio_envelope(args.video)

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    rows = []
    with open(args.csv, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    # prepare clip writer dir
    outdir = None
    if args.write_clips:
        outdir = args.write_clips
        os.makedirs(outdir, exist_ok=True)

    refined = []
    for idx, r in enumerate(rows, start=1):
        s = float(r['start']); e = float(r['end'])
        score = float(r.get('score', 0.0))

        # find best peak inside the window
        peak = find_peak_time(cap, s, e, audio_t=at, audio_env=aenv, use_blue=args.bias_blue)

        rs = clamp(peak - args.pre, 0.0, total_dur-0.05)
        re = clamp(peak + args.post, 0.0, total_dur)
        if re <= rs + 0.1:
            re = clamp(rs + 0.1, 0, total_dur)

        refined.append((rs, re, score))

        if outdir:
            # write tracked clip
            f0 = int(rs * fps)
            f1 = min(total_frames-1, int(re * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

            def frame_iter():
                for _ in range(f0, f1):
                    ok, fr = cap.read()
                    if not ok: break
                    yield fr

            clip_path = os.path.join(outdir, f"clip_{idx:04d}.mp4")
            tracked_crop_writer(
                clip_path, frame_iter(), W, H, 
                aspect=args.aspect, zoom=args.zoom, fps=fps
            )
            print(f"[{idx:04d}] {rs:.2f}-{re:.2f}s -> {clip_path}")

    # write refined CSV compatible with 04_make_highlights.py (start,end,score)
    with open(args.outcsv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start','end','score'])
        for rs, re, sc in refined:
            w.writerow([f"{rs:.2f}", f"{re:.2f}", f"{sc:.3f}"])

    cap.release()
    print(f"Wrote refined CSV: {args.outcsv}")
    if outdir:
        print(f"Wrote tracked clips to: {outdir}")

if __name__ == "__main__":
    main()
