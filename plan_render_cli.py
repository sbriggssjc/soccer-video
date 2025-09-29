# plan_render_cli.py
import argparse, os, json
import numpy as np
import pandas as pd
import cv2, subprocess
from scipy.signal import savgol_filter

def odd(n): 
    n=int(n); 
    return n if n%2==1 else n+1

def smooth(arr, fps, seconds=0.40):
    n=len(arr)
    if n<5: return arr
    win = odd(max(5, int(round(fps*seconds))))
    win = min(win, n-(1-n%2))  # <= n and odd
    if win<5: return arr
    return savgol_filter(arr, window_length=win, polyorder=2)

def lowpass_iir(prev, target, alpha):
    return prev + alpha*(target-prev)

def clamp(v, lo, hi):
    return lo if v<lo else hi if v>hi else v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--track_csv", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--W_out", type=int, default=608)
    ap.add_argument("--H_out", type=int, default=1080)

    # camera dynamics
    ap.add_argument("--slew", type=float, default=160)      # px/s
    ap.add_argument("--accel", type=float, default=480)     # px/s^2
    ap.add_argument("--zoom_rate", type=float, default=0.22)  # 1/s
    ap.add_argument("--zoom_accel", type=float, default=0.7)  # 1/s^2
    ap.add_argument("--left_frac", type=float, default=0.44)

    # composition / look-ahead
    ap.add_argument("--zoom_min", type=float, default=1.00)
    ap.add_argument("--zoom_max", type=float, default=1.70)
    ap.add_argument("--ctx_radius", type=int, default=420)
    ap.add_argument("--k_near", type=int, default=4)
    ap.add_argument("--ctx_pad", type=float, default=0.30)

    # “fit both ball + player” zoom policy
    ap.add_argument("--player_fit_margin", type=int, default=80)   # px margin around union
    ap.add_argument("--keep_margin", type=int, default=40)         # hard keep-in-frame margin

    # safety
    ap.add_argument("--speed_tight", type=float, default=60)
    ap.add_argument("--speed_wide", type=float, default=240)
    ap.add_argument("--hyst", type=float, default=35)

    args = ap.parse_args()

    # video props
    cap=cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # base crop width for output aspect (kept exact; no aspect warp)
    base_crop_w = int(np.floor(H*args.W_out/args.H_out/2)*2)
    base_crop_w = min(base_crop_w, W - (W%2))
    base_crop_h = int(np.floor(base_crop_w*args.H_out/args.W_out/2)*2)
    # NOTE: we compute width from height (portrait), then height from width to maintain exact aspect.

    df=pd.read_csv(args.track_csv)
    cx = pd.to_numeric(df['cx'], errors='coerce').to_numpy()
    cy = pd.to_numeric(df['cy'], errors='coerce').to_numpy()
    persons_raw = df.get('persons_json', pd.Series(['']*len(df))).fillna('').tolist()

    # Fill small gaps on ball track
    def fill_short_gaps(a, limit=12):
        s = pd.Series(a)
        s = s.ffill(limit=limit).bfill(limit=limit)
        # if any NaN left, nearest fill
        if s.isna().any():
            idx = np.where(~s.isna())[0]
            if len(idx)>0:
                s = pd.Series(np.interp(np.arange(len(a)), idx, s.iloc[idx].to_numpy()))
        return s.to_numpy()

    cx = fill_short_gaps(cx)
    cy = fill_short_gaps(cy)

    # Smooth a touch to avoid micro-jitter
    cx_s = smooth(cx, fps, 0.35)
    cy_s = smooth(cy, fps, 0.35)

    # Parse persons
    persons = []
    for j in range(len(persons_raw)):
        if persons_raw[j]:
            try:
                plist = json.loads(persons_raw[j])
                persons.append([(float(x),float(y),float(x1),float(y1),float(x2),float(y2)) for (x,y,x1,y1,x2,y2) in plist])
            except Exception:
                persons.append([])
        else:
            persons.append([])

    # Plan zoom to fit ball + nearest player (if any)
    # zoom=1.0 => crop width = base_crop_w
    Z = np.ones_like(cx_s, dtype=float)
    target_left = np.zeros_like(cx_s, dtype=float)

    for i in range(len(cx_s)):
        bx,by = cx_s[i], cy_s[i]
        # pick nearest player by center distance
        plist = persons[i]
        fit_w = base_crop_w
        if plist:
            d = [ ( (px-bx)**2 + (py-by)**2, (x1,y1,x2,y2) ) for (px,py,x1,y1,x2,y2) in plist ]
            d.sort(key=lambda u:u[0])
            # candidate: nearest k
            k = min(args.k_near, len(d))
            xs=[bx]; ys=[by]
            for n in range(k):
                x1,y1,x2,y2 = d[n][1]
                xs += [x1,x2]; ys += [y1,y2]
            minx, maxx = min(xs)-args.player_fit_margin, max(xs)+args.player_fit_margin
            # compute width to fit union with context pad
            union_w = maxx - minx
            union_w *= (1.0 + args.ctx_pad)
            fit_w = clamp(union_w, base_crop_w/args.zoom_max, base_crop_w/args.zoom_min)
        # desired zoom that would achieve that width
        z_des = clamp(base_crop_w / max(1.0, fit_w), args.zoom_min, args.zoom_max)
        Z[i] = z_des

        # left edge target (ball slightly left for lead room)
        L = bx - args.left_frac * (base_crop_w / z_des)
        L = clamp(L, 0, W - (base_crop_w / z_des))
        target_left[i] = L

    # Smooth desired sequences & apply dynamics + hard keep-in-frame
    Ld = smooth(target_left, fps, 0.45)
    Zd = smooth(Z, fps, 0.45)

    dt = 1.0/float(fps)
    # camera state
    L = Ld[0]
    V = 0.0
    Zc = float(Zd[0])  # current zoom
    Zrate = 0.0

    left_path = np.zeros_like(Ld)
    zoom_path = np.zeros_like(Zd)

    # speed caps by zoom (wider => faster allowed)
    def slew_cap(z):
        # interpolate between tight and wide based on effective width
        eff_w = base_crop_w / z
        t = clamp((eff_w - (base_crop_w/args.zoom_max)) / ( (base_crop_w/args.zoom_min) - (base_crop_w/args.zoom_max) + 1e-6 ), 0.0, 1.0)
        return args.speed_tight*(1-t) + args.speed_wide*t

    for i in range(len(Ld)):
        # pan dynamics towards Ld[i]
        sc = min(args.slew, slew_cap(Zc))
        err = Ld[i] - L
        v_des = clamp(err/dt, -sc, sc)
        # accel limit
        dv = clamp(v_des - V, -args.accel*dt, args.accel*dt)
        V += dv
        L += V*dt

        # zoom dynamics towards Zd[i]
        zr_des = clamp((Zd[i] - Zc)/dt, -args.zoom_rate, args.zoom_rate)
        dzr = clamp(zr_des - Zrate, -args.zoom_accel*dt, args.zoom_accel*dt)
        Zrate += dzr
        Zc += Zrate*dt
        Zc = clamp(Zc, args.zoom_min, args.zoom_max)

        # ---- HARD KEEP-IN-FRAME CONSTRAINT ----
        eff_w = base_crop_w / Zc
        bx = cx_s[i]
        # keep the ball inside crop with a small margin
        Lmin = clamp(bx - eff_w + args.keep_margin, 0, W - eff_w)
        Lmax = clamp(bx - args.keep_margin, 0, W - eff_w)
        if L < Lmin: 
            L = Lmin; V = 0.0  # kill velocity when clamped to avoid “spring out”
        elif L > Lmax:
            L = Lmax; V = 0.0

        left_path[i] = L
        zoom_path[i] = Zc

    # Final gentle smoothing pass (no warp, we re-constrain after)
    left_path = smooth(left_path, fps, 0.25)
    zoom_path = smooth(zoom_path, fps, 0.25)

    # Render frames (pure crop+scale, no aspect warp)
    out_dir = os.path.dirname(args.out_mp4)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_temp_frames")
    os.makedirs(tmp, exist_ok=True)

    cap = cv2.VideoCapture(args.clip)
    i=0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        z = float(zoom_path[min(i, len(zoom_path)-1)])
        eff_w = int(round(base_crop_w / max(z,1e-6)))
        eff_w -= eff_w % 2
        eff_h = int(round(eff_w * args.H_out / args.W_out))
        eff_h -= eff_h % 2
        # center vertical on ball Y with safety (never warp)
        by = int(round(cy_s[min(i,len(cy_s)-1)]))
        top = clamp(by - eff_h//2, 0, H - eff_h)
        left = int(round(left_path[min(i,len(left_path)-1)]))
        left = clamp(left, 0, W - eff_w)

        crop = bgr[top:top+eff_h, left:left+eff_w]
        frame = cv2.resize(crop, (args.W_out, args.H_out), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(tmp, f'f_{i:06d}.jpg'), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        i+=1
    cap.release()

    # Encode with source audio
    fr = int(round(fps))
    subprocess.run([
        'ffmpeg','-y',
        '-framerate', str(fr),
        '-i', os.path.join(tmp, 'f_%06d.jpg'),
        '-i', args.clip,
        '-map','0:v','-map','1:a:0?',
        '-c:v','libx264','-preset','veryfast','-crf','19',
        '-x264-params','keyint=120:min-keyint=120:scenecut=0',
        '-pix_fmt','yuv420p','-profile:v','high','-level','4.0',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-shortest','-movflags','+faststart',
        '-c:a','aac','-b:a','128k',
        args.out_mp4
    ], check=True)

    print("Wrote", args.out_mp4)

if __name__ == "__main__":
    main()
