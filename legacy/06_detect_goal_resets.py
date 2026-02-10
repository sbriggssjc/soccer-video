import argparse, csv, os, math
import numpy as np
import cv2
import librosa
from scipy.signal import find_peaks

def load_audio_peaks(path, hop_s=0.05, win_s=0.20, prom_mult=2.5, min_sep_s=12.0):
    y, sr = librosa.load(path, sr=None, mono=True)
    hop = int(sr*hop_s)
    win = int(sr*win_s)
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    # Normalize
    rms = (rms - np.median(rms)) / (np.std(rms) + 1e-8)
    prom = prom_mult  # since z-scored
    dist = int(max(1, min_sep_s / hop_s))
    peaks, props = find_peaks(rms, prominence=prom, distance=dist)
    return t[peaks], props.get("prominences", np.ones_like(peaks)), t[-1] if len(t) else 0.0

def white_line_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Field-ish green (for masking non-pitch)
    g_lo, g_hi = np.array([35, 60, 40], np.uint8), np.array([90, 255, 255], np.uint8)
    green = cv2.inRange(hsv, g_lo, g_hi)
    # White paint lines (low saturation, high value)
    w_lo, w_hi = np.array([0, 0, 200], np.uint8), np.array([179, 80, 255], np.uint8)
    white = cv2.inRange(hsv, w_lo, w_hi)
    # limit to pitch area
    white = cv2.bitwise_and(white, white, mask=green)
    white = cv2.medianBlur(white, 3)
    return white, green

def center_score_for_frame(bgr, ring_r=(0.17, 0.32), center_band=0.18):
    h, w = bgr.shape[:2]
    white, green = white_line_mask(bgr)
    # bail if not much pitch
    if np.count_nonzero(green) / (h*w) < 0.05:
        return 0.0
    # Horizontal half-way line near vertical center
    lines = cv2.HoughLinesP(white, 1, np.pi/180, threshold=70,
                            minLineLength=int(0.25*w), maxLineGap=20)
    hline_len = 0
    if lines is not None:
        yc_min, yc_max = int(h*(0.5-center_band)), int(h*(0.5+center_band))
        for x1,y1,x2,y2 in lines[:,0]:
            ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            if ang < 12 or ang > 168:  # near horizontal
                if yc_min <= y1 <= yc_max or yc_min <= y2 <= yc_max:
                    hline_len = max(hline_len, abs(x2-x1))
    hline_score = min(1.0, hline_len / (0.6*w))

    # Ring from center circle: donut of white pixels
    yy, xx = np.indices((h, w))
    cx, cy = w//2, h//2
    r = np.hypot(xx - cx, yy - cy)
    r1, r2 = ring_r[0]*min(h,w), ring_r[1]*min(h,w)
    ring_mask = ((r >= r1) & (r <= r2)).astype(np.uint8)
    ring_white = np.count_nonzero(cv2.bitwise_and(white, white, mask=ring_mask))
    ring_score = ring_white / (np.count_nonzero(ring_mask) + 1e-6)

    # Combine
    return 0.6*hline_score + 0.4*ring_score

def scan_center_scores(video, sample_fps=1.5, max_frames=None, resize_w=640):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise SystemExit(f"Could not open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n / fps if n else 0.0
    step = max(1, int(round(fps / sample_fps)))
    times, scores = [], []
    for idx in range(0, n, step):
        if max_frames and len(times) >= max_frames: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        if w > resize_w:
            scale = resize_w / w
            frame = cv2.resize(frame, (resize_w, int(h*scale)), interpolation=cv2.INTER_AREA)
        s = center_score_for_frame(frame)
        times.append(idx / fps)
        scores.append(s)
    cap.release()
    return np.array(times), np.array(scores), dur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-fps", type=float, default=1.5)
    ap.add_argument("--search-after", type=float, default=10.0, help="seconds after cheer to start looking")
    ap.add_argument("--search-until", type=float, default=80.0, help="seconds after cheer to stop looking")
    ap.add_argument("--center-thresh", type=float, default=0.08)
    ap.add_argument("--backfill", type=float, default=28.0, help="how far before kickoff to start clip")
    ap.add_argument("--end-pad", type=float, default=3.0, help="end a bit before kickoff")
    ap.add_argument("--prom-mult", type=float, default=2.5, help="cheer prominence (z-units)")
    ap.add_argument("--min-sep", type=float, default=12.0, help="min seconds between cheer peaks")
    args = ap.parse_args()

    # Audio cheers
    cheer_t, cheer_prom, _ = load_audio_peaks(args.video,
                                              prom_mult=args.prom_mult,
                                              min_sep_s=args.min_sep)

    # Center scores
    t_cent, s_cent, dur = scan_center_scores(args.video, sample_fps=args.sample_fps)

    # For each cheer, find the strongest center frame shortly after
    rows = []
    for ct, prom in zip(cheer_t, cheer_prom):
        win_lo, win_hi = ct + args.search_after, ct + args.search_until
        mask = (t_cent >= win_lo) & (t_cent <= win_hi)
        if not np.any(mask): continue
        idx = np.argmax(s_cent[mask])
        t_kick = t_cent[mask][idx]
        cscore = s_cent[mask][idx]
        if cscore < args.center_thresh: continue

        start = max(0.0, t_kick - args.backfill)
        end   = max(0.0, min(dur, t_kick - args.end_pad))
        if end <= start: continue

        # Score: reward center match and cheer strength
        score = 0.70 + 0.20*min(1.0, cscore/0.20) + 0.10*min(1.0, prom/4.0)
        rows.append({"start": f"{start:.2f}", "end": f"{end:.2f}", "score": f"{score:.3f}"})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","score"])
        w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
