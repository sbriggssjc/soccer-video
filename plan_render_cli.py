# -*- coding: utf-8 -*-
import argparse, os, math, subprocess, re
import numpy as np
import pandas as pd
import cv2


def clamp(v, a, b):
    return max(a, min(b, v))


def ball_interval(ball_x, eff_w, W, margin):
    # feasible left positions that keep the ball inside with margin
    left_min = ball_x - (eff_w - margin)
    left_max = ball_x - margin
    # intersect with image bounds
    a = max(0.0, left_min)
    b = min(W - eff_w, left_max)
    return a, b  # may be a > b -> infeasible for this eff_w


def ensure_feasible_zoom(ball_x, eff_w, W, margin, zoom_min_w):
    # If infeasible, widen (increase eff_w) until feasible or we hit zoom_min_w
    for _ in range(6):
        a, b = ball_interval(ball_x, eff_w, W, margin)
        if a <= b:
            return eff_w
        eff_w = min(zoom_min_w, eff_w * 1.15)  # zoom out 15% steps
    return min(zoom_min_w, eff_w)


def ease(N):
    if N <= 1:
        return np.ones(N, float)
    t = np.linspace(0, 1, N)
    return t * t * (3 - 2 * t)  # smoothstep


def fill_segment(arr, i0, i1, v0, v1, eased=False):
    i0 = int(max(0, i0))
    i1 = int(min(len(arr), i1))
    if i1 <= i0:
        return
    N = i1 - i0
    w = ease(N) if eased else np.linspace(0, 1, N)
    seg = v0 + (v1 - v0) * w
    arr[i0:i1] = seg


# helper interpolation utilities
def _fill_segment(arr, i0, i1, v0, v1):
    """Write a linear segment v0->v1 into arr[i0:i1] with exact length."""
    i0 = int(max(0, i0))
    i1 = int(min(len(arr), i1))
    if i1 <= i0:
        return
    N = i1 - i0
    seg = np.linspace(float(v0), float(v1), N, endpoint=True, dtype=float)
    arr[i0:i1] = seg


def _fill_segment_ease(arr, i0, i1, v0, v1, fps, vmax, amax):
    """Fill arr[i0:i1] using jerk-limited easing between v0 and v1."""
    i0 = int(max(0, i0))
    i1 = int(min(len(arr), i1))
    if i1 <= i0:
        return
    n = i1 - i0
    fps = float(fps) if fps and fps > 0 else 24.0
    T = max(float(n) / fps, 1.0 / fps)
    seg = ease_path(v0, v1, T, fps, vmax=vmax, amax=amax)
    if len(seg) != n:
        if len(seg) <= 1:
            seg = np.full(n, float(v1), dtype=float)
        else:
            idx_src = np.linspace(0.0, len(seg) - 1, n, endpoint=True, dtype=float)
            seg = np.interp(idx_src, np.arange(len(seg), dtype=float), seg)
    arr[i0:i1] = seg

# ----------------------
# utils
# ----------------------
def parse_ppl(s):
    if not isinstance(s, str) or not s:
        return []
    out = []
    for tok in s.split("|"):
        try:
            x,y = tok.split(":")
            out.append((float(x), float(y)))
        except Exception:
            pass
    return out

def smooth_1d(arr, fps, secs=0.40):
    n = len(arr)
    if n < 5: return arr.copy()
    win = max(5, int(round(fps*secs)))
    if win % 2 == 0: win += 1
    win = min(win, n - (1 - n%2))  # keep odd, <= n
    if win < 5: return arr.copy()
    # simple SG-like via convolution with a Hann window (fast & robust)
    w = 0.5 - 0.5*np.cos(2*np.pi*np.arange(win)/win)
    w /= w.sum()
    pad = win//2
    a2 = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(a2, w, mode="valid")

def label_from_filename(path):
    base = os.path.basename(path).upper()
    tags = []
    for t in ["DRIBBLING","DRIBBLE","CROSS","SHOT","SHOTS","GOAL","BUILD_UP","DEFENSE","CORNER"]:
        if t in base:
            tags.append(t)
    return tags

# ----------------------
# context box around ball + nearest players
# ----------------------
def context_box(cx, cy, ppl, aspect, ctx_radius=420.0, k_near=4, ctx_pad=0.30):
    xs, ys = [cx], [cy]
    if ppl:
        near = []
        for px,py in ppl:
            d = math.hypot(px - cx, py - cy)
            if d <= ctx_radius:
                near.append((d, px, py))
        near.sort(key=lambda t: t[0])
        for _,px,py in near[:k_near]:
            xs.append(px); ys.append(py)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw = max(40.0, maxx - minx)
    bh = max(40.0, maxy - miny)

    # pad horizontally and proportionally vertically
    padw = bw * ctx_pad
    box_w = bw + 2*padw
    box_h = bh + 2*padw/ aspect

    cx_box = 0.7*cx + 0.3*((minx+maxx)/2.0)
    cy_box = 0.7*cy + 0.3*((miny+maxy)/2.0)

    # fit to aspect
    if box_w / box_h > aspect:
        box_h = box_w / aspect
    else:
        box_w = box_h * aspect

    left  = cx_box - 0.5*box_w
    top   = cy_box - 0.5*box_h
    return left, top, box_w, box_h

# ----------------------
# jerk-limited easing (position only), with vel/accel caps
# ----------------------
def ease_path(p0, p1, T, fps, vmax, amax):
    """
    Plan 1D motion from p0->p1 over T seconds with bounded vel/accel.
    Returns array of positions length N=T*fps (rounded).
    If T is short, it will still respect bounds as best as possible.
    """
    N = max(1, int(round(T*fps)))
    if N == 1: return np.array([p1], dtype=float)
    dt = 1.0/fps
    x = p0
    v = 0.0
    out = np.zeros(N, dtype=float)
    for i in range(N):
        # naive time-symmetric profile: steer v towards needed velocity
        # for remaining distance, using bang-bang accel but clipped
        rem = p1 - x
        # “required v” heuristic to stop exactly at p1:
        # v_req^2 ≈ 2*amax*|rem|
        v_req = math.copysign(math.sqrt(max(0.0, 2*amax*abs(rem))), rem)
        # move current v toward v_req within accel limits
        dv = clamp(v_req - v, -amax*dt, amax*dt)
        v  = clamp(v + dv, -vmax, vmax)
        # integrate
        x  = x + v*dt
        out[i] = x
    # final adjust to hit p1 exactly
    out[-1] = p1
    return out

# ----------------------
# plan keyframes from tags + ball speed profile
# ----------------------
def plan_keyframes(times, cx, cy, ppl_series, W, H, aspect, tags,
                   left_frac=0.44,
                   ctx_radius=420.0, k_near=4, ctx_pad=0.30,
                   zoom_min=1.0, zoom_max=1.8,
                   speed_tight=60, speed_wide=260, hyst=35):
    """
    Returns list of keyframes:
      [(t, left, top, width, zoom), ...]
    left/top/width define the context box mapping (height from aspect).
    zoom multiplies base crop width (1.0 = base width, 1.8 = tighter)
    """
    T = times[-1] - times[0]
    t0 = times[0]; t1 = times[-1]
    cx_s = smooth_1d(cx, fps=1.0/(times[1]-times[0]) if len(times)>1 else 24.0, secs=0.30)
    cy_s = smooth_1d(cy, fps=1.0/(times[1]-times[0]) if len(times)>1 else 24.0, secs=0.30)
    vx = np.gradient(cx_s, times)
    spd = np.abs(vx)

    # simple “events” from speed:
    # - dribble window: sustained medium speed near beginning third
    # - cross bump: peak speed mid-clip
    # - shot settle: last third, speed drop
    n = len(times)
    i_peak = int(np.argmax(spd)) if n else 0
    t_peak = times[i_peak] if n else t0
    i_drib = int(max(0, min(n-1, round(0.15*n))))
    t_drib = times[i_drib] if n else t0
    i_shot = int(max(0, min(n-1, round(0.85*n))))
    t_shot = times[i_shot] if n else t1

    # tag-based nudges
    if any(t in tags for t in ["DRIBBLING","DRIBBLE"]):
        t_drib = times[int(0.10*n)]
    if "CROSS" in tags:
        t_peak = times[int(0.55*n)]
    if any(t in tags for t in ["SHOT","SHOTS","GOAL"]):
        t_shot = times[int(0.80*n)]

    # helper: build context at a time index
    def box_at(i, lead_sec=0.20):
        j = min(n-1, max(0, i + int(round(lead_sec / (times[1]-times[0] if n>1 else 1/24)))) )
        cxi = cx_s[j]; cyi = cy_s[j]
        ppl = ppl_series[j] if j < len(ppl_series) else []
        L,T,Wb,Hb = context_box(cxi, cyi, ppl, aspect, ctx_radius, k_near, ctx_pad)
        # slide box so ball is left_frac into it
        left  = cxi - left_frac*Wb
        top   = cyi - 0.50*Hb
        return left, top, Wb

    KF = []

    # KF0: start wide (establish)
    idx0 = 0
    L0, T0, W0 = box_at(idx0, lead_sec=0.0)
    KF.append((times[idx0], L0, T0, W0, zoom_min))

    # KF1: dribble tighter
    i1 = i_drib
    L1, T1, W1 = box_at(i1, lead_sec=0.0)
    KF.append((t_drib, L1, T1, max(W1*0.8, W0*0.7), min(1.25, zoom_min*1.25)))

    # KF2: pre-cross anticipate (aim where ball will be at peak speed)
    i2 = i_peak
    L2, T2, W2 = box_at(i2, lead_sec=0.25)
    KF.append((t_peak-0.20, L2, T2, max(W2, W1*1.1), 1.10))

    # KF3: shot framing (tight)
    i3 = i_shot
    L3, T3, W3 = box_at(i3, lead_sec=0.10)
    KF.append((t_shot-0.10, L3, T3, max(W3*0.75, W0*0.55), min(zoom_max, 1.55)))

    # KF4: aftermath settle (wider & centered around last context)
    KF.append((t1, L3, T3, max(W3*1.30, W0), 1.00))

    # Clamp to image and ensure even-ish widths later when cropping
    return KF

# ----------------------
# main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)

    # output size
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    # camera bounds (per-second)
    ap.add_argument("--slew",   type=float, default=180.0)  # px/s pan/tilt
    ap.add_argument("--accel",  type=float, default=600.0)  # px/s^2 pan/tilt
    ap.add_argument("--zoom_rate",  type=float, default=0.35) # zoom units/s
    ap.add_argument("--zoom_accel", type=float, default=0.90) # zoom units/s^2

    # context box & zoom ranges
    ap.add_argument("--left_frac", type=float, default=0.44)
    ap.add_argument("--ctx_radius", type=float, default=420.0)
    ap.add_argument("--k_near",     type=int,   default=4)
    ap.add_argument("--ctx_pad",    type=float, default=0.30)
    ap.add_argument("--zoom_min",   type=float, default=1.00)
    ap.add_argument("--zoom_max",   type=float, default=1.80)

    # speed-to-zoom heuristics (used in planner)
    ap.add_argument("--speed_tight", type=float, default=60)
    ap.add_argument("--speed_wide",  type=float, default=260)
    ap.add_argument("--hyst",        type=float, default=35)

    args = ap.parse_args()

    # video info
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps_safe = fps if fps and fps > 0 else 24.0

    # base crop width to preserve aspect (before resize)
    aspect   = args.W_out / args.H_out
    crop_w_base = int(np.floor(H * aspect / 2) * 2)
    crop_w_base = min(crop_w_base, W - (W % 2))
    crop_h_base = int(crop_w_base / aspect)

    # load tracking
    df = pd.read_csv(args.track_csv)
    cx = pd.to_numeric(df["cx"], errors="coerce").to_numpy()
    cy = pd.to_numeric(df["cy"], errors="coerce").to_numpy()
    # fill any remaining NaNs
    for a, default in ((cx, W/2), (cy, H/2)):
        if np.isnan(a).any():
            idx = np.where(~np.isnan(a))[0]
            if len(idx) >= 2:
                a[:] = np.interp(np.arange(len(a)), idx, a[idx])
            else:
                a[:] = np.nan_to_num(a, nan=default)
    ppl_series = [parse_ppl(s) for s in df.get("ppl", [])]

    # time array
    times = (df["frame"].to_numpy(dtype=float) / (fps if fps>0 else 24.0))
    if len(times) == 0:
        raise SystemExit("Empty track CSV.")

    # tags from filename
    tags = label_from_filename(args.clip)

    # -------- plan keyframes (intent) --------
    KF = plan_keyframes(times, cx, cy, ppl_series, W, H, aspect, tags,
                        left_frac=args.left_frac,
                        ctx_radius=args.ctx_radius, k_near=args.k_near, ctx_pad=args.ctx_pad,
                        zoom_min=args.zoom_min, zoom_max=args.zoom_max,
                        speed_tight=args.speed_tight, speed_wide=args.speed_wide, hyst=args.hyst)

    # derive desired zoom curve from keyframes
    N = len(cx)
    zoom_des = np.ones(N, dtype=float)
    if len(KF) >= 1 and N:
        zoom_des.fill(float(KF[0][4]))
    if len(KF) >= 2 and N:
        t_base = times[0]
        for seg in range(len(KF) - 1):
            t0, _, _, _, Z0 = KF[seg]
            t1, _, _, _, Z1 = KF[seg + 1]
            i0 = int(np.floor((t0 - t_base) * fps_safe))
            i1 = int(np.floor((t1 - t_base) * fps_safe))
            if seg == len(KF) - 2:
                i1 = N
            i0 = max(0, min(i0, N - 1))
            i1 = max(i0 + 1, min(i1, N))
            fill_segment(zoom_des, i0, i1, Z0, Z1, eased=True)
    zoom_des[np.isnan(zoom_des)] = float(args.zoom_min)

    # -------- projected, slew-limited camera path --------
    zoom = np.clip(np.copy(zoom_des), float(args.zoom_min), float(args.zoom_max))
    zoom[np.isnan(zoom)] = float(args.zoom_min)

    zr = float(args.zoom_rate)
    za = float(args.zoom_accel)
    vz = 0.0
    for i in range(1, N):
        zt = zoom[i]
        dz = np.clip(zt - zoom[i - 1], -zr / fps_safe, zr / fps_safe)
        dv = np.clip(dz - vz, -za / (fps_safe ** 2), za / (fps_safe ** 2))
        vz += dv
        zoom[i] = zoom[i - 1] + vz

    zoom = np.minimum(zoom, float(args.zoom_max))
    zoom[~np.isfinite(zoom)] = float(args.zoom_min)

    left = np.zeros(N, float)
    v = 0.0  # px/s
    margin = 16.0
    center_k = float(args.left_frac)

    zoom_min_w = H * (args.W_out / args.H_out)
    zoom_min_w = float(zoom_min_w)

    if N:
        eff_w = zoom_min_w / max(zoom[0], 1e-6)
        eff_w = ensure_feasible_zoom(cx[0], eff_w, W, margin, zoom_min_w)
        zoom[0] = min(zoom[0], zoom_min_w / max(eff_w, 1e-6))
        eff_w = zoom_min_w / max(zoom[0], 1e-6)
        left0_pref = cx[0] - center_k * eff_w
        a, b = ball_interval(cx[0], eff_w, W, margin)
        if a > b:
            a, b = 0.0, max(0.0, W - eff_w)
        left[0] = clamp(left0_pref, a, b)

    slew = float(args.slew)
    accel = float(args.accel)
    dt = 1.0 / fps_safe
    for i in range(1, N):
        eff_w = zoom_min_w / max(zoom[i], 1e-6)
        eff_w = ensure_feasible_zoom(cx[i], eff_w, W, margin, zoom_min_w)
        zoom[i] = min(zoom[i], zoom_min_w / max(eff_w, 1e-6))
        eff_w = zoom_min_w / max(zoom[i], 1e-6)

        left_pref = cx[i] - center_k * eff_w

        a, b = ball_interval(cx[i], eff_w, W, margin)
        if a > b:
            a, b = 0.0, max(0.0, W - eff_w)

        err = left_pref - left[i - 1]
        v_des = np.clip(err / dt, -slew, slew)
        dv = np.clip(v_des - v, -accel * dt, accel * dt)
        v += dv
        v = np.clip(v, -slew, slew)

        x_next = left[i - 1] + v * dt
        x_next = clamp(x_next, a, b)

        if x_next == a and v < 0:
            v = 0.0
        if x_next == b and v > 0:
            v = 0.0

        left[i] = x_next

    for _ in range(1):
        left_s = left.copy()
        k = 0.12
        for i in range(1, N - 1):
            left_s[i] = (1 - k) * left[i] + k * 0.5 * (left[i - 1] + left[i + 1])

        for i in range(N):
            eff_w = zoom_min_w / max(zoom[i], 1e-6)
            eff_w = ensure_feasible_zoom(cx[i], eff_w, W, margin, zoom_min_w)
            a, b = ball_interval(cx[i], eff_w, W, margin)
            if a > b:
                a, b = 0.0, max(0.0, W - eff_w)
            left_s[i] = clamp(left_s[i], a, b)

        left = left_s

    # -------- render --------
    tmp_dir = os.path.join(os.path.dirname(args.out_mp4) or ".", "_temp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # derive crop from projected left/zoom
        if len(zoom):
            Z = zoom[i] if i < len(zoom) else zoom[-1]
        else:
            Z = float(args.zoom_min)
        Z = min(Z, float(args.zoom_max))

        eff_w = int(np.floor((H * (args.W_out / args.H_out)) / max(Z, 1e-6) / 2) * 2)
        eff_w = max(2, min(W - (W % 2), eff_w))

        if len(left):
            left_val = left[i] if i < len(left) else left[-1]
        else:
            left_val = 0.0
        xi = int(round(clamp(left_val, 0.0, max(0.0, W - eff_w))))

        crop = frame[:, xi:xi+eff_w]
        if crop.shape[1] != eff_w:
            pad_w = max(0, eff_w - crop.shape[1])
            crop = cv2.copyMakeBorder(crop, 0, 0, 0, pad_w, cv2.BORDER_REPLICATE)

        crop = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp_dir, f"f_{i:06d}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i += 1

    cap.release()
    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    subprocess.run([
        "ffmpeg","-y",
        "-framerate", str(int(round(fps))),
        "-i", os.path.join(tmp_dir, "f_%06d.jpg"),
        "-i", args.clip,
        "-map","0:v","-map","1:a:0?",
        "-c:v","libx264","-preset","veryfast","-crf","19",
        "-x264-params","keyint=120:min-keyint=120:scenecut=0",
        "-pix_fmt","yuv420p","-profile:v","high","-level","4.0",
        "-colorspace","bt709","-color_primaries","bt709","-color_trc","bt709",
        "-shortest","-movflags","+faststart",
        "-c:a","aac","-b:a","128k",
        args.out_mp4
    ], check=True)
    print("Wrote", args.out_mp4)

if __name__ == "__main__":
    main()
