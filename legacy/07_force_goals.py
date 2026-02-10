import csv, os, math
import numpy as np
import cv2
import argparse

# Reuse similar flow metrics as filter (brief)
def flow_score(cap, s, e, step=2, ds=2):
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    sf, ef = int(s*fps), int(e*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    prev = None
    flows = []
    for i in range(max(1, ef-sf)):
        ok, f = cap.read()
        if not ok: break
        if i % step: continue
        if ds > 1:
            f = cv2.resize(f, None, fx=1.0/ds, fy=1.0/ds, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 1, 15, 2, 5, 1.1, 0)
            flows.append(np.mean(cv2.magnitude(flow[...,0], flow[...,1])))
        prev = g
    return float(np.median(flows)) if flows else 0.0

def best_window_before(cap, anchor_t, lookback=22.0, dur=6.0, stride=0.75):
    # scan [anchor - lookback, anchor - 3s] in strides, pick highest flow_score
    start = max(0.0, anchor_t - lookback)
    end   = max(0.0, anchor_t - 3.0)
    t = start
    best = (None, -1.0)
    while t + dur <= end:
        s, e = t, t + dur
        sc = flow_score(cap, s, e)
        if sc > best[1]:
            best = ((s, e), sc)
        t += stride
    return best[0] if best[0] else (max(0.0, anchor_t - 7.0), max(0.0, anchor_t - 1.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--goal-resets", required=True)  # CSV with start,end,score (restart windows)
    ap.add_argument("--out", required=True)          # CSV forced_goals.csv
    ap.add_argument("--max-goals", type=int, default=6)
    ap.add_argument("--pre", type=float, default=1.5)
    ap.add_argument("--post", type=float, default=2.5)
    args = ap.parse_args()

    # load anchors (use 'start' as restart time)
    anchors = []
    with open(args.goal_resets, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            anchors.append(float(row.get("start", 0.0)))
    anchors = sorted(anchors)
    # dedupe close anchors
    merged = []
    for a in anchors:
        if not merged or abs(a - merged[-1]) > 8.0:
            merged.append(a)

    cap = cv2.VideoCapture(args.video)
    forced = []
    for a in merged[: args.max_goals]:
        s, e = best_window_before(cap, a, lookback=22.0, dur=6.0, stride=0.75)
        s = max(0.0, s - args.pre)   # pad before the move
        e = e + args.post
        forced.append(dict(start=round(s,2), end=round(e,2), score=1.0, tag="goal_forced"))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","score","tag"])
        w.writeheader()
        for row in forced:
            w.writerow(row)
    print(f"[force_goals] wrote {len(forced)} -> {args.out}")

if __name__ == "__main__":
    main()
