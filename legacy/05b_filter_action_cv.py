# 05b_filter_action_cv.py
import argparse, csv, math, os, sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2, numpy as np, pandas as pd

def _to_num(x):
    if pd.isna(x): return None
    s = str(x).strip().replace(',', '.')
    try: return float(s)
    except: return None

def read_candidates(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    scol = cols.get('start') or [c for c in df.columns if 'start' in c.lower()][0]
    ecol = cols.get('end')   or [c for c in df.columns if 'end'   in c.lower()][0]
    score_col = None
    for key in ('action_score','score','rank'):
        if key in cols: score_col = cols[key]; break
    rows = []
    for _,r in df.iterrows():
        s = _to_num(r[scol]); e = _to_num(r[ecol])
        if s is None or e is None or e <= s: continue
        sc = _to_num(r[score_col]) if score_col else 0.0
        rows.append(dict(start=float(s), end=float(e), score=float(sc)))
    return rows

@dataclass
class HSVRange:
    h_low:int; s_low:int; v_low:int
    h_high:int; s_high:int; v_high:int
    def low(self):  return np.array([self.h_low, self.s_low, self.v_low], dtype=np.uint8)
    def high(self): return np.array([self.h_high, self.s_high, self.v_high], dtype=np.uint8)

def parse_hsv(arg):
    a = [int(x) for x in arg.split(',')]
    if len(a)!=6: raise ValueError("HSV must be 6 ints: hL,sL,vL,hH,sH,vH")
    return HSVRange(*a)


class KitClassifier:
    """Simple HSV-based two-class jersey classifier updated online."""

    def __init__(self, navy_hsv: HSVRange, opponent_seed: Optional[HSVRange] = None) -> None:
        self.tol = np.array([15.0, 70.0, 80.0], dtype=np.float32)
        self.navy_center = np.array(
            [
                0.5 * (navy_hsv.h_low + navy_hsv.h_high),
                0.5 * (navy_hsv.s_low + navy_hsv.s_high),
                0.5 * (navy_hsv.v_low + navy_hsv.v_high),
            ],
            dtype=np.float32,
        )
        if opponent_seed is not None:
            self.opp_center = np.array(
                [
                    0.5 * (opponent_seed.h_low + opponent_seed.h_high),
                    0.5 * (opponent_seed.s_low + opponent_seed.s_high),
                    0.5 * (opponent_seed.v_low + opponent_seed.v_high),
                ],
                dtype=np.float32,
            )
        else:
            self.opp_center = np.array([30.0, 45.0, 200.0], dtype=np.float32)
        self.navy_weight = 1.0
        self.opp_weight = 1.0

    @staticmethod
    def _clip_center(center: np.ndarray) -> np.ndarray:
        center[0] = float(np.clip(center[0], 0.0, 180.0))
        center[1] = float(np.clip(center[1], 0.0, 255.0))
        center[2] = float(np.clip(center[2], 0.0, 255.0))
        return center

    def _valid_pixels(self, roi: np.ndarray) -> np.ndarray:
        if roi.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        sat_mask = roi[..., 1] > 45
        val_mask = roi[..., 2] > 60
        green = cv2.inRange(roi, (30, 30, 30), (90, 255, 255)) > 0
        fg = sat_mask & val_mask & (~green)
        if not np.any(fg):
            return np.zeros((0, 3), dtype=np.float32)
        pixels = roi[fg]
        return pixels.astype(np.float32)

    def _range(self, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        low = np.clip(center - self.tol, [0.0, 0.0, 0.0], [180.0, 255.0, 255.0]).astype(np.uint8)
        high = np.clip(center + self.tol, [0.0, 0.0, 0.0], [180.0, 255.0, 255.0]).astype(np.uint8)
        return low, high

    def _ratio(self, roi: np.ndarray, center: np.ndarray) -> float:
        low, high = self._range(center)
        mask = cv2.inRange(roi, low, high)
        return float(mask.mean() / 255.0)

    def _update(self, label: str, sample: np.ndarray, weight: float) -> None:
        sample = self._clip_center(sample.copy())
        weight = float(max(0.0, weight))
        if weight <= 0:
            return
        if label == "navy":
            total = self.navy_weight + weight
            self.navy_center = self._clip_center(
                (self.navy_center * self.navy_weight + sample * weight) / total
            )
            self.navy_weight = min(total, 1000.0)
        else:
            total = self.opp_weight + weight
            self.opp_center = self._clip_center(
                (self.opp_center * self.opp_weight + sample * weight) / total
            )
            self.opp_weight = min(total, 1000.0)

    def classify_patch(
        self, hsv_frame: np.ndarray, cx: int, cy: int, patch: int = 28
    ) -> Tuple[Optional[str], float, float, float]:
        h, w = hsv_frame.shape[:2]
        x1 = max(0, cx - patch)
        y1 = max(0, cy - patch)
        x2 = min(w, cx + patch)
        y2 = min(h, cy + patch)
        roi = hsv_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0, 0.0, 0.0
        pixels = self._valid_pixels(roi)
        if pixels.size == 0:
            return None, 0.0, 0.0, 0.0
        sample = np.median(pixels, axis=0)
        navy_ratio = self._ratio(roi, self.navy_center)
        opp_ratio = self._ratio(roi, self.opp_center)
        total_ratio = navy_ratio + opp_ratio
        if total_ratio < 1e-4:
            return None, 0.0, navy_ratio, opp_ratio
        label = "navy" if navy_ratio >= opp_ratio else "opponent"
        confidence = abs(navy_ratio - opp_ratio)
        if confidence >= 0.02:
            # weight updates by fraction of foreground pixels to remain stable
            fg_weight = min(pixels.shape[0] / 200.0, 1.0)
            self._update(label, sample, max(confidence, 0.05) * fg_weight)
        return label, float(confidence), float(navy_ratio), float(opp_ratio)

def optical_flow_metrics(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 25, 3, 5, 1.1, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    # Global (camera) direction ~ vector median of flow
    vx = np.median(flow[...,0]); vy = np.median(flow[...,1])
    vmag = math.hypot(float(vx), float(vy))
    # Pixels that deviate from global direction by > ~35°
    denom = np.maximum(1e-6, np.sqrt(flow[...,0]**2+flow[...,1]**2))
    base = max(vmag, 1e-6)
    dot = (flow[...,0]*vx + flow[...,1]*vy) / (base * denom)
    dev_mask = (dot < math.cos(math.radians(35)))
    # Residual action magnitude (not camera pan)
    residual = mag[dev_mask]
    residual_mag = float(np.median(residual)) if residual.size else 0.0
    median_flow = float(np.median(mag)) if mag.size else 0.0
    return dict(
        residual=residual_mag,
        camera_mag=float(vmag),
        camera_dir=float(math.atan2(float(vy), float(vx))) if vmag > 1e-6 else 0.0,
        median_flow=median_flow,
    )

def green_ratio(hsv):
    # wide green band for pitches
    mask = cv2.inRange(hsv, (30, 25, 25), (90, 255, 255))
    return float(np.mean(mask>0))

def find_ball_centroid(bgr):
    # white-ish but not bright lines cluster; small area = ball
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0,0,200), (180,60,255))
    # suppress pitch lines (long, thin): erode a bit
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k, iterations=1)
    cnts,_ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = white.shape
    best=None; best_score=1e9
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 6 or a > 180:  # tiny to small blobs
            continue
        (x,y), r = cv2.minEnclosingCircle(c)
        circ = 0 if r<=0 else (a/(math.pi*r*r))
        # prefer near-circular small blobs
        score = abs(1.0-circ) + 0.002*a
        if score < best_score:
            best_score = score; best = (int(x),int(y))
    return best

def team_presence_near(hsv, cx, cy, team_hsv:HSVRange, patch=22):
    h,w,_ = hsv.shape
    x1 = max(0, cx-patch); y1 = max(0, cy-patch)
    x2 = min(w, cx+patch); y2 = min(h, cy+patch)
    roi = hsv[y1:y2, x1:x2]
    if roi.size==0: return 0.0
    mask = cv2.inRange(roi, team_hsv.low(), team_hsv.high())
    return float(np.mean(mask>0))

def analyze_window(cap, start, end, fps_sample, team_hsv, kit: KitClassifier, att_third_cut=0.18):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0,start)*1000.0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    step = max(1,int(round(cap.get(cv2.CAP_PROP_FPS)/fps_sample))) if cap.get(cv2.CAP_PROP_FPS)>0 else 4

    idx=0; prev_gray=None; prev_ball=None
    green_rates=[]; flow_resid=[]; ball_speeds=[]; team_near=[]; att_pos=[]
    navy_touch=0.0; opp_touch=0.0; last_touch=None; touch_events=0; touch_conf_sum=0.0

    idx=0; prev_gray=None; prev_ball=None; prev_ball_time=None; ball_miss=0
    green_rates=[]; flow_resid=[]; pan_mags=[]; pan_dirs=[]; flow_medians=[]
    ball_speeds=[]; ball_speed_track=[]; ball_positions=[]; team_near=[]; att_pos=[]
    ball_visible=0

    while True:
        t = start + (idx/fps_sample)
        if t>=end: break
        ok = cap.grab()
        for _ in range(step-1):
            cap.grab()
        ok, frame = cap.retrieve()
        if not ok: break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        g = green_ratio(hsv); green_rates.append(g)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow_m = optical_flow_metrics(prev_gray, gray)
            flow_resid.append(flow_m['residual'])
            pan_mags.append(flow_m['camera_mag'])
            pan_dirs.append(flow_m['camera_dir'])
            flow_medians.append(flow_m['median_flow'])
        prev_gray = gray
        ball = find_ball_centroid(frame)
        t_rel = (idx / fps_sample) if fps_sample else 0.0
        if ball:
            ball_visible += 1
            cx,cy = ball
            ball_positions.append((t_rel, float(cx), float(cy)))
            if prev_ball:
                dt_frames = max(1, idx - prev_ball_time) if prev_ball_time is not None else 1
                dx = (cx-prev_ball[0]); dy = (cy-prev_ball[1])
                speed = math.hypot(dx,dy)
                if dt_frames > 1:
                    speed /= dt_frames
                ball_speeds.append(speed)
                ball_speed_track.append(speed)
            else:
                ball_speed_track.append(0.0)
            prev_ball = ball
            prev_ball_time = idx
            ball_miss = 0
            team_near.append(team_presence_near(hsv,cx,cy,team_hsv))
            # attacking thirds in X (left/right edges)
            att_pos.append( 1.0 if (cx < att_third_cut*w or cx > (1.0-att_third_cut)*w) else 0.0 )

            label, conf, navy_ratio, opp_ratio = kit.classify_patch(hsv, cx, cy)
            navy_touch += navy_ratio
            opp_touch += opp_ratio
            if label is not None:
                last_touch = label
                touch_events += 1
                touch_conf_sum += conf
                if label == "navy":
                    navy_touch += max(0.02, conf * 0.25)
                else:
                    opp_touch += max(0.02, conf * 0.25)
            elif last_touch == "navy":
                navy_touch += 0.01
            elif last_touch == "opponent":
                opp_touch += 0.01

        else:
            ball_speed_track.append(0.0)
            ball_miss += 1
            if ball_miss > 3:
                prev_ball = None
                prev_ball_time = None

        idx += 1
    # Aggregate
    green_ok = np.mean(green_rates) if green_rates else 0
    flow   = np.median(flow_resid) if flow_resid else 0
    residual_peak = float(np.max(flow_resid)) if flow_resid else 0.0
    flow_mean = float(np.mean(flow_resid)) if flow_resid else 0.0
    pan_peak = float(np.max(pan_mags)) if pan_mags else 0.0
    pan_mean = float(np.mean(pan_mags)) if pan_mags else 0.0
    if ball_speeds:
        sp = np.array(ball_speeds)
        # robust stats
        speed_med = float(np.median(sp))
        contig = int(np.max(np.convolve((sp>3.5).astype(int), np.ones(5,dtype=int), 'same')))  # >= ~5 frames
        hits = int(np.sum((sp[1:]-sp[:-1])>2.5))  # acceleration spikes
    else:
        speed_med=0.0; contig=0; hits=0
        sp = np.array([], dtype=float)
    team_pres = float(np.mean(team_near)) if team_near else 0.0
    att_frac  = float(np.mean(att_pos))  if att_pos  else 0.0

    total_touch = navy_touch + opp_touch
    if total_touch > 1e-6:
        navy_poss = float(navy_touch / total_touch)
        opp_poss = float(opp_touch / total_touch)
    elif last_touch == "navy":
        navy_poss, opp_poss = 1.0, 0.0
    elif last_touch == "opponent":
        navy_poss, opp_poss = 0.0, 1.0
    else:
        navy_poss = opp_poss = 0.0
    if last_touch == "navy":
        last_touch_flag = 1.0
    elif last_touch == "opponent":
        last_touch_flag = 0.0
    else:
        last_touch_flag = 0.5
    avg_touch_conf = (touch_conf_sum / touch_events) if touch_events else 0.0
    poss_conf = max(abs(navy_poss - opp_poss), avg_touch_conf)

    ball_visible_ratio = float(ball_visible / max(1, idx))
    speed_track_max = float(max(ball_speed_track)) if ball_speed_track else 0.0
    contig_frames = 0
    if ball_speed_track:
        run = 0
        for v in ball_speed_track:
            if v > 3.5:
                run += 1
                if run > contig_frames:
                    contig_frames = run
            else:
                run = 0

    return dict(
        green_ok=green_ok, flow=flow,
        speed_med=speed_med, contig=contig, hits=hits,
        team_pres=team_pres, att_frac=att_frac,

        navy_possession=navy_poss, opp_possession=opp_poss,
        last_touch_navy=last_touch_flag, possession_conf=poss_conf

        residual_series=flow_resid,
        residual_peak=residual_peak,
        residual_mean=flow_mean,
        pan_series=pan_mags,
        pan_dir_series=pan_dirs,
        pan_peak=pan_peak,
        pan_mean=pan_mean,
        flow_median_series=flow_medians,
        ball_speeds=ball_speeds,
        ball_speed_track=ball_speed_track,
        ball_positions=ball_positions,
        ball_visible_ratio=ball_visible_ratio,
        speed_max=float(np.max(sp)) if sp.size else 0.0,
        speed_track_max=speed_track_max,
        contig_frames=contig_frames,
        att_flags=att_pos,
        team_series=team_near,
        green_series=green_rates,
        duration=float(max(0.0, end-start)),
        sample_dt=(1.0/fps_sample) if fps_sample else 0.0,
        frame_width=float(w),
        frame_height=float(h),

    )


def _to_array(values):
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.array(values, dtype=np.float32)


def _ball_arrays(win):
    pts = win.get('ball_positions') or []
    if not pts:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    arr = np.array(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty
    times = arr[:, 0]
    xs = arr[:, 1]
    ys = arr[:, 2]
    return times, xs, ys


def _camera_pan_alignment(win, ball_dir):
    if abs(ball_dir) < 1e-3:
        return 0.0
    dirs = _to_array(win.get('pan_dir_series'))
    if dirs.size == 0:
        return 0.0
    mags = _to_array(win.get('pan_series'))
    cos_vals = np.cos(dirs)
    aligned = cos_vals * float(ball_dir)
    if mags.size and mags.size == aligned.size:
        val = float(np.average(aligned, weights=np.maximum(mags, 1e-6)))
    else:
        val = float(aligned.mean())
    return max(0.0, val)


def is_shot(win):
    times, xs, _ = _ball_arrays(win)
    if xs.size < 2:
        return 0.0
    width = max(float(win.get('frame_width', 1.0)), 1.0)
    start_x = xs[0] / width
    end_x = xs[-1] / width
    progress = end_x - start_x
    direction = math.copysign(1.0, progress) if abs(progress) > 1e-3 else 0.0
    toward_goal = abs(end_x - 0.5) > 0.30
    speed_peak = max(float(win.get('speed_max', 0.0)), float(win.get('speed_track_max', 0.0)))
    spike = (speed_peak > 6.0) or (speed_peak > 5.0 and int(win.get('hits', 0)) >= 2)
    alignment = _camera_pan_alignment(win, direction)
    pan_peak = float(win.get('pan_peak', 0.0))
    residual_peak = float(win.get('residual_peak', 0.0))
    final_presence = final_third_presence(win)
    if spike and toward_goal and alignment > 0.1 and pan_peak > 1.0 and residual_peak > 0.9 and final_presence > 0.25:
        return 1.0
    att_flags = _to_array(win.get('att_flags'))
    enters_box = bool(att_flags.size) and float(att_flags.mean()) > 0.25
    if enters_box and residual_peak > 1.1 and speed_peak > 4.0:
        return 1.0
    return 0.0


def is_shot_attempt(win, shot_conf=None):
    shot_conf = shot_conf if shot_conf is not None else is_shot(win)
    if shot_conf >= 1.0:
        return 1.0
    speed_peak = max(float(win.get('speed_max', 0.0)), float(win.get('speed_track_max', 0.0)))
    if speed_peak < 4.0:
        return 0.0
    if final_third_presence(win) < 0.2:
        return 0.0
    residual_peak = float(win.get('residual_peak', 0.0))
    pan_peak = float(win.get('pan_peak', 0.0))
    if residual_peak > 0.75 or pan_peak > 0.8:
        return 1.0
    return 0.0


def has_pass_chain(win, min_len=3, window_s=8):
    times, xs, _ = _ball_arrays(win)
    if xs.size < min_len:
        return 0.0
    width = max(float(win.get('frame_width', 1.0)), 1.0)
    xs_norm = xs / width
    team_series = _to_array(win.get('team_series'))
    team_avg = float(team_series.mean()) if team_series.size else 0.0
    if team_avg < 0.08 or float(win.get('ball_visible_ratio', 0.0)) < 0.4:
        return 0.0
    for i in range(0, len(xs_norm) - min_len + 1):
        j = i + min_len - 1
        dt = float(times[j] - times[i])
        if dt > window_s:
            continue
        seg = xs_norm[i:j+1]
        progress = float(seg[-1] - seg[0])
        direction = math.copysign(1.0, progress) if abs(progress) > 1e-3 else 0.0
        if abs(progress) < 0.18 or direction == 0.0:
            continue
        diffs = np.diff(seg)
        touches = int(np.sum(np.abs(diffs) > 0.015)) + 1
        if touches < min_len:
            continue
        if np.any(diffs * direction < -0.02):
            continue
        return 1.0
    return 0.0


def has_switch_of_play(win):
    times, xs, _ = _ball_arrays(win)
    if xs.size < 2:
        return 0.0
    width = max(float(win.get('frame_width', 1.0)), 1.0)
    xs_norm = xs / width
    span = float(xs_norm.max() - xs_norm.min()) if xs_norm.size else 0.0
    if span < 0.45:
        return 0.0
    idx_min = int(np.argmin(xs_norm))
    idx_max = int(np.argmax(xs_norm))
    dt = abs(float(times[idx_max] - times[idx_min]))
    crosses_mid = ((xs_norm[idx_min] < 0.4 and xs_norm[idx_max] > 0.6) or
                   (xs_norm[idx_max] < 0.4 and xs_norm[idx_min] > 0.6))
    edge_flip = ((xs_norm[idx_min] < 0.25 and xs_norm[idx_max] > 0.75) or
                 (xs_norm[idx_max] < 0.25 and xs_norm[idx_min] > 0.75))
    if dt <= 4.0 and (crosses_mid or edge_flip):
        return 1.0
    return 0.0


def has_tackle_or_press(win):
    speeds = _to_array(win.get('ball_speed_track'))
    if speeds.size < 2:
        return 0.0
    residual_peak = float(win.get('residual_peak', 0.0))
    if residual_peak < 0.75:
        return 0.0
    team_series = _to_array(win.get('team_series'))
    team_peak = float(team_series.max()) if team_series.size else 0.0
    dt = float(win.get('sample_dt', 0.0))
    if dt <= 0:
        dt = 1.0 / 6.0
    look = max(1, int(round(1.0 / max(dt, 1e-3))))
    press = False
    for i in range(len(speeds)):
        if speeds[i] < 3.0:
            continue
        j = min(len(speeds) - 1, i + look)
        if float(np.min(speeds[i:j+1])) < 0.8:
            press = True
            break
    if not press:
        for i in range(len(speeds)):
            if speeds[i] > 0.9:
                continue
            j = min(len(speeds) - 1, i + look)
            if float(np.max(speeds[i:j+1])) > 3.2:
                press = True
                break
    if press and team_peak > 0.12:
        return 1.0
    return 0.0


def final_third_presence(win):
    val = win.get('att_frac', 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def is_restart_setup(win):
    flow_med = float(win.get('flow', 0.0))
    pan_mean = float(win.get('pan_mean', 0.0))
    speed_peak = float(win.get('speed_track_max', 0.0))
    ball_vis = float(win.get('ball_visible_ratio', 0.0))
    team_avg = float(_to_array(win.get('team_series')).mean()) if win.get('team_series') else 0.0
    green_mean = float(np.mean(win.get('green_series'))) if win.get('green_series') else 0.0
    duration = float(win.get('duration', 0.0))
    if duration < 2.0:
        return 0.0
    if flow_med < 0.18 and pan_mean < 0.35 and speed_peak < 1.5 and ball_vis < 0.25 and team_avg < 0.08:
        return 1.0
    if flow_med < 0.22 and speed_peak < 2.0 and green_mean > 0.55 and ball_vis < 0.35:
        return 1.0
    return 0.0


def is_stationary_block(win):
    residual = float(win.get('flow', 0.0))
    speed_peak = float(win.get('speed_track_max', 0.0))
    pan_peak = float(win.get('pan_peak', 0.0))
    if residual < 0.2 and speed_peak < 1.0 and pan_peak < 0.6:
        return 1.0
    if residual < 0.28 and speed_peak < 1.4 and float(win.get('ball_visible_ratio', 0.0)) < 0.45:
        return 1.0
    return 0.0


def action_score(win):
    shot = is_shot(win)
    attempt = is_shot_attempt(win, shot)
    score  = 3.0 * shot + 2.0 * attempt
    score += 2.0 * has_pass_chain(win, min_len=3, window_s=8)
    score += 1.5 * has_switch_of_play(win)
    score += 1.2 * has_tackle_or_press(win)
    score += 0.8 * final_third_presence(win)
    score -= 2.0 * is_restart_setup(win)
    score -= 1.0 * is_stationary_block(win)
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--csv',   required=True)
    ap.add_argument('--out',   required=True)
    ap.add_argument('--fps-sample', type=float, default=6.0)
    ap.add_argument('--min-green', type=float, default=0.30)
    ap.add_argument('--min-flow',  type=float, default=0.20)     # motion not due to camera pan
    ap.add_argument('--min-ball-speed', type=float, default=3.8) # px/frame median
    ap.add_argument('--min-contig-frames', type=int, default=6)
    ap.add_argument('--min-ball-hits', type=int, default=1)
    ap.add_argument('--team-hsv', default='105,70,20,130,255,160')
    ap.add_argument('--min-team-pres', type=float, default=0.10)
    ap.add_argument('--team-bias', type=float, default=0.25)
    ap.add_argument('--opp-hsv', default=None, help="optional HSV seed for opponent kit (hL,sL,vL,hH,sH,vH)")
    ap.add_argument('--att-third-cut', type=float, default=0.18)
    args = ap.parse_args()

    team_hsv = parse_hsv(args.team_hsv)
    opp_seed = parse_hsv(args.opp_hsv) if args.opp_hsv else None
    kit = KitClassifier(team_hsv, opp_seed)
    rows = read_candidates(args.csv)
    if not rows:
        print(f"No valid rows in {args.csv}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    out_rows=[]
    for r in rows:
        m = analyze_window(cap, r['start'], r['end'], args.fps_sample, team_hsv, kit, args.att_third_cut)
        if m['green_ok'] < args.min_green:  # off-field/bench
            continue
        action = action_score(m)
        if m['flow'] < args.min_flow and action < 1.0:
            continue
        low_ball = (
            m['speed_med'] < args.min_ball_speed
            or m.get('contig_frames', 0) < args.min_contig_frames
            or m['hits'] < args.min_ball_hits
        )
        if low_ball and action < 1.5:
            continue

        mean_flow = float(m['flow'])
        g_cont = (m.get('contig_frames', 0) >= args.min_contig_frames)
        g_ball = (m['speed_med'] >= args.min_ball_speed)
        g_hits = (m['hits'] >= args.min_ball_hits)
        g_team = (m['team_pres'] >= args.min_team_pres)

        ok = g_cont and (g_ball or g_hits or mean_flow >= args.min_flow * 1.15)
        if not ok and g_team and mean_flow >= args.min_flow:
            ok = True

        action_override = False
        if not ok and action >= 1.5:
            ok = True
            action_override = True

        why = []
        if not g_cont:
            why.append("no_continuity")
        if not g_ball:
            why.append("no_ball_speed")
        if not g_hits:
            why.append("no_ball_hits")
        if mean_flow < args.min_flow:
            why.append("low_flow")
        if not g_team:
            why.append("low_team")

        if not ok:
            continue

        label = "keep_action" if action_override else "keep"
        out_rows.append(dict(
            start=r['start'], end=r['end'], action_score=round(float(action),4),

            flow=m['flow'], speed_med=m['speed_med'], contig=m['contig'], hits=m['hits'],
            team_pres=m['team_pres'], att_frac=m['att_frac'],
            navy_possession=m['navy_possession'], opp_possession=m['opp_possession'],
            last_touch_navy=m['last_touch_navy'], possession_conf=m['possession_conf']

            flow=m['flow'], speed_med=m['speed_med'], contig=m['contig'], contig_frames=m.get('contig_frames', 0), hits=m['hits'],
            team_pres=m['team_pres'], att_frac=m['att_frac'],
            why=label if not why else f"{label}({'/'.join(why)})"

        ))
    cap.release()

    if not out_rows:
        print("No clips passed the action filter; try lowering --min-flow to 0.15 or --min-ball-speed to ~3.2", file=sys.stderr)
    with open(args.out,'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ['start','end','action_score'])
        w.writeheader()
        for r in out_rows: w.writerow(r)

if __name__=="__main__":
    main()
