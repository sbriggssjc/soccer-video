#!/usr/bin/env python
# tools/overlay_debug.py — self-contained overlay from JSONL telemetry
import argparse, json, os
from pathlib import Path
import cv2

def read_jsonl(p):
    recs = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input mp4 used for render")
    ap.add_argument("--telemetry", required=True, help="JSONL written by unified renderer")
    ap.add_argument("--out", default=None, help="output debug mp4")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--ball-radius", type=int, default=6)
    args = ap.parse_args()

    inp = os.path.abspath(args.inp)
    tel = os.path.abspath(args.telemetry)
    out = os.path.abspath(args.out) if args.out else os.path.splitext(inp)[0] + ".__DEBUG.mp4"

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    recs = read_jsonl(tel)
    if not recs:
        print("[ERROR] no telemetry records found"); return 2

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[ERROR] cannot open input video: {inp}"); return 3

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out, fourcc, fps, (W, H), True)
    if not vw.isOpened():
        print(f"[ERROR] cannot open writer: {out}"); return 4

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        rec = recs[idx] if idx < len(recs) else recs[-1]

        # draw crop box (source pixel coords)
        crop = rec.get("crop")
        if crop and len(crop) == 4:
            x0, y0, w, h = [int(round(v)) for v in crop]
            x1, y1 = x0 + w, y0 + h
            x0 = max(0, min(W-1, x0)); y0 = max(0, min(H-1, y0))
            x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), args.thickness)

        # draw ball (red)
        bx = by = None
        ball = rec.get("ball")
        if ball and len(ball) == 2:
            bx, by = float(ball[0]), float(ball[1])
        elif "bx_stab" in rec and "by_stab" in rec:
            bx, by = float(rec["bx_stab"]), float(rec["by_stab"])
        elif "bx_raw" in rec and "by_raw" in rec:
            bx, by = float(rec["bx_raw"]), float(rec["by_raw"])

        if bx is not None and by is not None:
            bx_i, by_i = int(round(bx)), int(round(by))
            if 0 <= bx_i < W and 0 <= by_i < H:
                cv2.circle(frame, (bx_i, by_i), args.ball_radius, (0,0,255), -1)

        vw.write(frame)
        idx += 1

    cap.release(); vw.release()
    print("Wrote", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
