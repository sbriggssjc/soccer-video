# -*- coding: utf-8 -*-
import argparse, os, math, subprocess, re
import numpy as np
import pandas as pd
import cv2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def interval_for_point(p, win, bound, margin=0):
    # feasible left/top for a window 'win' to include 'p' with 'margin' inside bounds [0, bound]
    a = p - (win - margin)
    b = p - margin
    return max(0.0, a), min(bound - win, b)  # (may be infeasible if a>b)


def intersect_intervals(ax, bx, ay, by):
    a = max(ax, ay)
    b = min(bx, by)
    return a, b, (a <= b)


def smoothstep_vec(N):
    if N <= 1: return np.ones(N, dtype=float)
    t = np.linspace(0,1,N)
    return t*t*(3-2*t)


def ease(N):
    if N <= 1:
        return np.ones(N, float)
    t = np.linspace(0, 1, N)
    return t * t * (3 - 2 * t)  # smoothstep


def fill_track(arr, default):
    if arr.size == 0:
        return arr
    default_arr = np.full_like(arr, float(default), dtype=float) if np.isscalar(default) else np.asarray(default, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() >= 2:
        idx = np.where(mask)[0]
        arr[:] = np.interp(np.arange(len(arr), dtype=float), idx.astype(float), arr[idx].astype(float))
    elif mask.sum() == 1:
        arr[:] = float(arr[mask][0])
    else:
        arr[:] = default_arr
    arr[~np.isfinite(arr)] = default_arr[~np.isfinite(arr)]
    return arr


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

    W_out, H_out = int(args.W_out), int(args.H_out)
    aspect = (W_out / H_out) if H_out else 1.0

    base_h = float(H)
    base_w = float(np.floor((base_h * aspect) / 2.0) * 2.0)
    if base_w > W:
        base_w = float(np.floor(W / 2.0) * 2.0)
        if base_w <= 0.0:
            base_w = float(max(2, W))
        base_h = base_w / aspect if aspect > 0 else float(H)
        base_h = float(np.floor(base_h / 2.0) * 2.0)
        if base_h <= 0.0 or base_h > float(H):
            base_h = float(H)

    base_w = max(16.0, min(float(W), base_w))
    base_h = max(16.0, min(float(H), base_h))

    # load tracking
    df = pd.read_csv(args.track_csv)
    cx = pd.to_numeric(df["cx"], errors="coerce").to_numpy(dtype=float)
    cy = pd.to_numeric(df["cy"], errors="coerce").to_numpy(dtype=float)
    px = pd.to_numeric(df.get("px", pd.Series(index=df.index)), errors="coerce").to_numpy(dtype=float)
    py = pd.to_numeric(df.get("py", pd.Series(index=df.index)), errors="coerce").to_numpy(dtype=float)

    fill_track(cx, float(W) / 2.0)
    fill_track(cy, float(H) / 2.0)
    fill_track(px, cx)
    fill_track(py, cy)

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
    dt = 1.0 / fps_safe if fps_safe else 1.0 / 24.0
    for i in range(1, N):
        zt = zoom[i]
        dz = np.clip(zt - zoom[i - 1], -zr * dt, zr * dt)
        dv = np.clip(dz - vz, -za * dt, za * dt)
        vz = clamp(vz + dv, -zr, zr)
        zoom[i] = zoom[i - 1] + vz

    zoom = np.clip(zoom, float(args.zoom_min), float(args.zoom_max))
    zoom[~np.isfinite(zoom)] = float(args.zoom_min)

    vx = np.gradient(cx) * fps_safe if N else np.zeros(N, dtype=float)
    vy = np.gradient(cy) * fps_safe if N else np.zeros(N, dtype=float)
    speed = np.hypot(vx, vy)

    qx = np.copy(px)
    qy = np.copy(py)
    fast = speed > 240.0
    for i in range(N):
        if not np.isfinite(cx[i]) or not np.isfinite(cy[i]):
            continue
        if not np.isfinite(qx[i]) or not np.isfinite(qy[i]):
            qx[i], qy[i] = cx[i], cy[i]
        if fast[i] and np.isfinite(px[i]) and np.isfinite(py[i]):
            look = 0.20
            rx = cx[i] + vx[i] * look
            ry = cy[i] + vy[i] * look
            axp = 0.6 * rx + 0.4 * qx[i]
            ayp = 0.6 * ry + 0.4 * qy[i]
            qx[i], qy[i] = axp, ayp

    margin_x = 18.0
    margin_y = 24.0
    center_kx = float(args.left_frac)
    center_ky = 0.50

    left = np.zeros(N, float)
    top = np.zeros(N, float)
    eff_w = np.zeros(N, float)
    eff_h = np.zeros(N, float)
    vxpan = 0.0
    vypan = 0.0
    slew = float(args.slew)
    accel = float(args.accel)
    vslew = 0.7 * slew
    vacc = 0.7 * accel

    for i in range(N):
        zoom_cur = max(zoom[i], 1e-6)
        scale = min(1.0, 1.0 / zoom_cur)
        okx = False
        oky = False
        ax = 0.0
        bx = max(0.0, W - base_w * scale)
        ay = 0.0
        by = max(0.0, H - base_h * scale)

        for _ in range(6):
            w = base_w * scale
            h = base_h * scale
            ax1, bx1 = interval_for_point(cx[i], w, W, margin_x)
            ax2, bx2 = interval_for_point(qx[i], w, W, margin_x)
            ax, bx, okx = intersect_intervals(ax1, bx1, ax2, bx2)

            ay1, by1 = interval_for_point(cy[i], h, H, margin_y)
            ay2, by2 = interval_for_point(qy[i], h, H, margin_y)
            ay, by, oky = intersect_intervals(ay1, by1, ay2, by2)

            if okx and oky:
                break
            scale = min(1.0, scale * 1.12)

        w = base_w * scale
        h = base_h * scale

        if not okx:
            ax, bx = interval_for_point(cx[i], w, W, margin_x)
        if not oky:
            ay, by = interval_for_point(cy[i], h, H, margin_y)

        if ax > bx:
            ax, bx = 0.0, max(0.0, W - w)
        if ay > by:
            ay, by = 0.0, max(0.0, H - h)

        left_pref = cx[i] - center_kx * w
        top_pref = cy[i] - center_ky * h

        if i == 0:
            left[i] = clamp(left_pref, ax, bx)
            top[i] = clamp(top_pref, ay, by)
        else:
            errx = left_pref - left[i - 1]
            vdx = np.clip(errx / dt, -slew, slew)
            dvx = np.clip(vdx - vxpan, -accel * dt, accel * dt)
            vxpan = np.clip(vxpan + dvx, -slew, slew)
            nx = clamp(left[i - 1] + vxpan * dt, ax, bx)
            if (nx == ax and vxpan < 0) or (nx == bx and vxpan > 0):
                vxpan = 0.0
            left[i] = nx

            erry = top_pref - top[i - 1]
            vdy = np.clip(erry / dt, -vslew, vslew)
            dvy = np.clip(vdy - vypan, -vacc * dt, vacc * dt)
            vypan = np.clip(vypan + dvy, -vslew, vslew)
            ny = clamp(top[i - 1] + vypan * dt, ay, by)
            if (ny == ay and vypan < 0) or (ny == by and vypan > 0):
                vypan = 0.0
            top[i] = ny

        w_final = max(16.0, np.floor(w / 2.0) * 2.0)
        h_final = max(16.0, np.floor(h / 2.0) * 2.0)
        w_final = min(float(W), w_final)
        h_final = min(float(H), h_final)

        eff_w[i] = w_final
        eff_h[i] = h_final
        zoom[i] = base_w / max(w_final, 1e-6)

    # -------- render --------
    tmp_dir = os.path.join(os.path.dirname(args.out_mp4) or ".", "_temp_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        if len(eff_w):
            w = int(round(eff_w[i] if i < len(eff_w) else eff_w[-1]))
        else:
            w = int(round(base_w))
        if len(eff_h):
            h = int(round(eff_h[i] if i < len(eff_h) else eff_h[-1]))
        else:
            h = int(round(base_h))
        w = max(2, min(W, w))
        h = max(2, min(H, h))

        left_val = left[i] if i < len(left) else (left[-1] if len(left) else 0.0)
        top_val = top[i] if i < len(top) else (top[-1] if len(top) else 0.0)

        x = int(round(clamp(left_val, 0.0, max(0.0, W - w))))
        y = int(round(clamp(top_val, 0.0, max(0.0, H - h))))

        crop = frame[y:y+h, x:x+w]
        if crop.shape[0] != h or crop.shape[1] != w:
            crop = cv2.copyMakeBorder(crop,
                                      0, max(0, h - crop.shape[0]),
                                      0, max(0, w - crop.shape[1]),
                                      cv2.BORDER_REPLICATE)

        crop = cv2.resize(crop, (W_out, H_out), interpolation=cv2.INTER_LANCZOS4)
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
