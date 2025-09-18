# 02_detect_events.py
# Detects soccer-specific highlight moments:
#  - pass chains (blue->blue)
#  - shots/goals (ball speed into goal corridor + stoppage)
#  - intensity (optical flow), lightly blended with audio if available
#
# Output: out/highlights.csv  (start,end,score,event)

import os, csv, argparse, math
import numpy as np
import cv2

def clamp(v, lo, hi): return max(lo, min(hi, v))

# --------- field / team / ball masks (tweak here if needed) ----------
def mask_green(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # broad pitch green
    lower = np.array([35, 40, 40], np.uint8)
    upper = np.array([90,255,255], np.uint8)
    return cv2.inRange(hsv, lower, upper)

def mask_blue(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90,  40,  40], np.uint8)   # <-- adjust if your kit is darker/lighter
    upper = np.array([130,255,255], np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    return m

def mask_white_ball(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # white/very light (low saturation, high value)
    lower = np.array([0, 0, 190], np.uint8)
    upper = np.array([180, 60, 255], np.uint8)
    m = cv2.inRange(hsv, lower, upper)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    return m

# ---------- optional audio excitement ----------
def audio_envelope(path):
    try:
        import librosa
        y, sr = librosa.load(path, sr=None, mono=True)
        hop = 512
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
        env = env.astype(np.float32)
        if env.size: env = (env - env.min())/(env.max()-env.min()+1e-8)
        return t, env
    except Exception:
        return None, None

# ---------- tiny helpers ----------
def contour_centroids(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        m = cv2.moments(c)
        if m['m00'] > 0:
            pts.append((int(m['m10']/m['m00']), int(m['m01']/m['m00'])))
    return pts

def nearest(pt, pts):
    if not pts: return None, 1e9
    d = [ ( (pt[0]-x)**2 + (pt[1]-y)**2 )**0.5 for (x,y) in pts ]
    j = int(np.argmin(d))
    return j, d[j]

# ---------- main pass ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', default='out/highlights.csv')
    ap.add_argument('--sample-fps', type=float, default=8.0, help='analysis fps (downsample)')
    ap.add_argument('--pass-window', type=float, default=8.0, help='seconds window for pass chains')
    ap.add_argument('--passes-needed', type=int, default=3, help='blue->blue completions to trigger event')
    ap.add_argument('--pre', type=float, default=1.0, help='seconds before peak')
    ap.add_argument('--post', type=float, default=2.0, help='seconds after peak')
    ap.add_argument('--bias-blue', action='store_true', help='slightly favor blue-side intensity')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise SystemExit('Could not open video')

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = N / fps

    # downsample stride
    stride = max(1, int(round(fps / args.sample_fps)))
    samp_fps = fps / stride

    # audio (optional)
    at, aenv = audio_envelope(args.video)

    # motion init
    prev = None
    flow_mag_series = []
    times = []

    # ball / players tracking state
    ball_pos = None
    ball_pos_prev = None
    ball_speed_series = []
    ball_x_series = []
    ball_y_series = []
    owners = []  # nearest blue cluster index (or -1 if none)
    blue_pts_series = []
    moving_players_series = []
    cam_pan_speed_series = []
    ball_on_pitch_series = []

    # global motion for stoppage detection
    global_motion = []

    # goal corridors (12% screen width at edges)
    goal_w = int(0.12 * W)
    left_goal = (0, 0, goal_w, H)
    right_goal = (W-goal_w, 0, goal_w, H)

    # iterate frames
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if f % stride != 0:
            f += 1; continue
        t = f / fps

        # shrink work image
        work = cv2.resize(frame, (W//2, H//2))
        gmask = mask_green(work)

        # intensity via absdiff (fast & stable)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        if prev is None:
            prev = gray
            f += 1
            continue
        diff = cv2.absdiff(gray, prev)
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev = gray
        mag = float(np.mean(diff))
        flow_mag_series.append(mag)
        times.append(t)
        global_motion.append(mag)
        cam_pan_speed_series.append(float(np.median(np.abs(flow[...,0]))))

        # slightly favor blue motion if requested
        if args.bias_blue:
            bm = mask_blue(work)
            if bm.any():
                wblue = float(np.mean(diff[bm>0]))
                mag = 0.7*mag + 0.3*wblue

        # blue centroids (players)
        blue = mask_blue(work)
        blue_pts = contour_centroids(blue, min_area=50)  # tune if far camera
        # map to full-res coords
        blue_pts_full = [ (x*2, y*2) for (x,y) in blue_pts ]
        blue_pts_series.append(blue_pts_full)

        motion_thr = max(10, int(diff.mean() + 1.5*diff.std()))
        motion_mask = diff > motion_thr
        moving_count = 0
        for (x, y) in blue_pts:
            x = int(np.clip(x, 0, motion_mask.shape[1] - 1))
            y = int(np.clip(y, 0, motion_mask.shape[0] - 1))
            x0 = max(0, x - 1)
            y0 = max(0, y - 1)
            x1 = min(motion_mask.shape[1], x + 2)
            y1 = min(motion_mask.shape[0], y + 2)
            if motion_mask[y0:y1, x0:x1].any():
                moving_count += 1
        moving_players_series.append(moving_count)

        # ball detection: white + moving
        wb = mask_white_ball(work)
        moving = motion_mask.astype(np.uint8) * 255
        ball_cand = cv2.bitwise_and(wb, moving)
        cnts, _ = cv2.findContours(ball_cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bpos = None
        bestd = 1e9
        bpos_half = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 5 or a > 200:  # size gate at half-res
                continue
            x,y,wc,hc = cv2.boundingRect(c)
            cx_half = x + wc/2.0
            cy_half = y + hc/2.0
            cx = cx_half * 2
            cy = cy_half * 2
            if ball_pos is not None:
                d = ((cx-ball_pos[0])**2 + (cy-ball_pos[1])**2)**0.5
            else:
                d = 0
            if d < bestd:
                bestd = d
                bpos = (cx, cy)
                bpos_half = (cx_half, cy_half)

        if bpos is not None:
            ball_pos = bpos
        ball_x_series.append(ball_pos[0] if ball_pos else np.nan)
        ball_y_series.append(ball_pos[1] if ball_pos else np.nan)

        if bpos_half is not None:
            bx_half = int(round(bpos_half[0]))
            by_half = int(round(bpos_half[1]))
            if 0 <= bx_half < gmask.shape[1] and 0 <= by_half < gmask.shape[0]:
                on_pitch = bool(gmask[by_half, bx_half] > 0)
            else:
                on_pitch = False
        else:
            on_pitch = True
        ball_on_pitch_series.append(on_pitch)

        # speed (pixels/sec at full-res)
        if len(ball_speed_series) == 0 or ball_pos is None or ball_pos_prev is None:
            ball_speed_series.append(0.0)
        else:
            prev_pos = ball_pos_prev
            dt = (stride / fps)
            v = (((ball_pos[0]-prev_pos[0])**2 + (ball_pos[1]-prev_pos[1])**2)**0.5) / max(1e-6, dt)
            ball_speed_series.append(float(v))
        ball_pos_prev = tuple(ball_pos) if ball_pos else None

        # possession owner = nearest blue cluster
        if ball_pos and blue_pts_full:
            j, d = nearest(ball_pos, blue_pts_full)
            owner = j if d < (0.1*W) else -1
        else:
            owner = -1
        owners.append(owner)

        f += 1

    cap.release()

    times = np.array(times, dtype=np.float32)
    if times.size == 0:
        raise SystemExit("No frames analyzed; check sample-fps/stride.")

    flow_mag = np.array(flow_mag_series, dtype=np.float32)
    flow_mag = (flow_mag - flow_mag.min())/(flow_mag.max()-flow_mag.min()+1e-8)

    blue_pts_series = blue_pts_series[:len(times)]

    ball_speed = np.array(ball_speed_series[:len(times)], dtype=np.float32)
    bs_norm = ball_speed / (np.nanmax(ball_speed)+1e-8)

    bx = np.array(ball_x_series[:len(times)], dtype=np.float32)
    moving_players = (
        np.array(moving_players_series[:len(times)], dtype=np.int32)
        if moving_players_series
        else np.zeros(len(times), dtype=np.int32)
    )
    cam_pan_speed = (
        np.array(cam_pan_speed_series[:len(times)], dtype=np.float32)
        if cam_pan_speed_series
        else np.zeros(len(times), dtype=np.float32)
    )
    if ball_on_pitch_series:
        ball_on_pitch = np.array(ball_on_pitch_series[:len(times)], dtype=bool)
    else:
        ball_on_pitch = np.ones(len(times), dtype=bool)

    by = np.array(ball_y_series[:len(times)], dtype=np.float32)

    # audio align
    if at is not None and aenv is not None and aenv.size:
        audio_s = np.interp(times, at, aenv).astype(np.float32)
    else:
        audio_s = np.zeros_like(times, dtype=np.float32)

    owners_arr = np.array(owners[:len(times)], dtype=np.int32)

    # ---------- TACKLES / DUELS (POSSESSION FLIPS) ----------
    tackle_events = []
    if len(times) > 1:
        quick_frames = max(1, int(round(1.6 * samp_fps)))
        quick_secs = quick_frames / samp_fps
        flow_thr = 0.55
        prox_radius = 0.12 * W
        prox_needed = 2

        for i in range(1, len(times)):
            new_owner = owners_arr[i]
            if new_owner < 0:
                continue
            if np.isnan(bx[i]) or np.isnan(by[i]):
                continue

            window_start = max(0, i - quick_frames)
            last_non_blue = None
            for k in range(i - 1, window_start - 1, -1):
                if owners_arr[k] < 0:
                    last_non_blue = k
                    break
            if last_non_blue is None:
                continue

            dt = times[i] - times[last_non_blue]
            if dt > quick_secs:
                continue

            sl_start = last_non_blue
            sl = slice(sl_start, i + 1)
            flow_window = flow_mag[sl]
            if flow_window.size == 0:
                continue
            flow_peak = float(np.max(flow_window))
            if flow_peak < flow_thr:
                continue

            max_prox = 0
            for m in range(sl_start, i + 1):
                if np.isnan(bx[m]) or np.isnan(by[m]):
                    continue
                pts = blue_pts_series[m]
                close = 0
                for (px, py) in pts:
                    if ((px - bx[m])**2 + (py - by[m])**2) ** 0.5 <= prox_radius:
                        close += 1
                if close > max_prox:
                    max_prox = close
                if max_prox >= prox_needed:
                    break
            if max_prox < prox_needed:
                continue

            peak_offset = int(np.argmax(flow_window))
            event_idx = sl_start + peak_offset
            event_time = times[event_idx]

            quickness = 1.0 - clamp(dt / quick_secs, 0.0, 1.0)
            crowd = clamp(max_prox / 3.0, 0.0, 1.0)
            score = clamp(0.6 * flow_peak + 0.25 * quickness + 0.15 * crowd, 0.0, 1.0)
            tackle_events.append((event_time, score, 'tackle'))

    # ---------- PASS CHAIN DETECTION ----------
    win = int(round(args.pass_window * samp_fps))
    pass_events = []

    for i in range(len(times)):
        j0 = max(0, i-win)
        seg = owners_arr[j0:i+1]
        seg = seg[seg>=0]
        if len(seg) < args.passes_needed: continue
        # count blue->blue completions: number of *changes* between distinct owners
        # e.g., [4,4,7,7,2] has changes 4->7 and 7->2 = 2 passes
        changes = 0
        last = None
        for o in seg:
            if last is None:
                last = o
                continue
            if o != last:
                changes += 1
                last = o
        if changes >= args.passes_needed-1:
            # score: more changes + some intensity + some audio
            score = 0.55*(changes/(args.passes_needed+1)) + 0.25*flow_mag[i] + 0.20*audio_s[i]
            pass_events.append( (times[i], score, 'passes') )

    # ---------- SHOT/GOAL HEURISTIC ----------
    goal_events = []
    # define edge corridors in normalized x
    left_edge = 0.12
    right_edge = 0.88
    low_motion = cv2.blur(flow_mag, (1, max(1,int(round(2*samp_fps)))))  # ~2s blur
    for i in range(len(times)):
        v = bs_norm[i]
        x = bx[i]
        if np.isnan(x) or v < 0.25:
            continue
        xn = x / W
        toward_goal = (xn < left_edge) or (xn > right_edge)
        if not toward_goal: continue

        # look ahead a few seconds for a lull (stoppage ~ kickoff after goal)
        k1 = i
        k2 = min(len(times)-1, i + int(round(6*samp_fps)))
        post_lull = (low_motion[k1:k2+1].mean() < 0.25)

        score = 0.6*v + 0.25*flow_mag[i] + 0.15*audio_s[i]
        if post_lull:
            score += 0.25  # “goal likely” bonus

        if score > 0.5:
            goal_events.append( (times[i], min(score,1.0), 'shot_or_goal') )

    # ---------- INTENSITY PEAKS (fallback) ----------
    peaks = []
    K = int(max(5, round(len(times)/300)))  # sprinkle a few peaks across game
    if K > 0:
        idxs = np.argpartition(-(flow_mag + 0.2*audio_s), K)[:K]
        for i in idxs:
            peaks.append( (times[i], float(flow_mag[i]), 'intensity') )

    # ---------- MERGE + NMS ----------
    all_events = pass_events + goal_events + tackle_events + peaks
    all_events.sort(key=lambda x: x[0])

    merged = []
    merge_tol = 6.0  # seconds
    for t, s, tag in all_events:
        if not merged:
            merged.append([t, s, tag])
            continue
        if t - merged[-1][0] <= merge_tol:
            # keep the stronger and prefer non-'intensity'
            if s > merged[-1][1] or (merged[-1][2]=='intensity' and tag!='intensity'):
                merged[-1] = [t, s, tag]
        else:
            merged.append([t, s, tag])

    # ---------- Window each event (pre/post) + write CSV ----------
    rows = []
    for (t, s, tag) in merged:

        start = clamp(t - args.pre, 0, dur-0.1)
        end   = clamp(t + args.post, 0, dur)
        idx0 = int(np.searchsorted(times, start, side='left'))
        idx1 = int(np.searchsorted(times, end, side='right'))
        idx0 = max(0, min(idx0, len(times)))
        idx1 = max(idx0 + 1, min(idx1, len(times))) if len(times) else 0
        window_len = idx1 - idx0
        if window_len <= 0:
            continue
        flow_slice = flow_mag[idx0:idx1]
        if flow_slice.size and float(np.median(flow_slice)) < 0.4:
            continue
        moving_slice = moving_players[idx0:idx1]
        if moving_slice.size:
            active_ratio = float(np.sum(moving_slice >= 4)) / window_len
            if active_ratio < 0.6:
                continue
        pan_slice = cam_pan_speed[idx0:idx1]
        ball_slice = ball_on_pitch[idx0:idx1]
        if pan_slice.size and ball_slice.size:
            median_pan = float(np.median(pan_slice))
            off_pitch_ratio = float(np.mean(~ball_slice))
            if median_pan < 0.05 and off_pitch_ratio > 0.5:
                continue

        if tag == 'tackle':
            pre = min(args.pre, 1.8)
            post = min(args.post, 3.5)
        else:
            pre = args.pre
            post = args.post
        start = clamp(t - pre, 0, dur-0.1)
        end   = clamp(t + post, 0, dur)

        rows.append((start, end, min(1.0, s), tag))

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['start','end','score','event'])
        for a,b,s,tag in rows:
            w.writerow([f'{a:.2f}', f'{b:.2f}', f'{s:.3f}', tag])

    print(f'Wrote {len(rows)} events -> {args.out}')

if __name__ == '__main__':
    main()
