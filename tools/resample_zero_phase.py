import argparse
import json
import math

import cv2
import numpy as np

def read_plan(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            t = o.get("t", None)
            bx = o.get("bx_stab", o.get("bx"))
            by = o.get("by_stab", o.get("by"))
            if bx is None or by is None:
                continue
            rows.append((float(t if t is not None else 0.0), float(bx), float(by)))
    # ensure sorted by time
    rows.sort(key=lambda r: r[0])
    return rows


def to_uniform(rows, nframes, fps):
    # map exact frame centers i/fps onto the input (t,bx,by) with linear interp
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    xs = np.array([r[1] for r in rows], dtype=np.float64)
    ys = np.array([r[2] for r in rows], dtype=np.float64)
    t0 = ts[0]
    t1 = ts[-1]
    # If plan is shorter than video, extrapolate endpoints
    frame_ts = np.arange(nframes, dtype=np.float64) / fps + t0
    x = np.interp(frame_ts, ts, xs, left=xs[0], right=xs[-1])
    y = np.interp(frame_ts, ts, ys, left=ys[0], right=ys[-1])
    return frame_ts, x, y


def zero_phase_ma(arr, win=9):
    # centred moving average (odd window) -> zero phase lag
    if win < 3:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    # reflect pad
    a = np.pad(arr, (pad, pad), mode="reflect")
    k = np.ones(win, dtype=np.float64) / win
    out = np.convolve(a, k, mode="valid")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=24.0, help="fallback if probe fails")
    ap.add_argument("--win", type=int, default=7, help="zero-phase MA window (odd)")
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    cap.release()

    rows = read_plan(args.inp)
    if len(rows) < 3:
        raise SystemExit("Plan too short")

    ts, x, y = to_uniform(rows, nframes, fps)
    x = zero_phase_ma(x, args.win)
    y = zero_phase_ma(y, args.win)

    with open(args.outp, "w", encoding="utf-8") as f:
        for i in range(nframes):
            o = {
                "t": float(i / fps),
                "bx": float(x[i]),
                "by": float(y[i]),
                "bx_stab": float(x[i]),
                "by_stab": float(y[i]),
            }
            f.write(json.dumps(o, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
