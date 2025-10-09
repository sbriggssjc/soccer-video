#!/usr/bin/env python
# tools/overlay_debug.py — self-contained overlay from JSONL telemetry
import argparse, json, os
from pathlib import Path
import cv2


def _get_ball_xy_src(rec, src_w, src_h):
    """
    Return ball center in *source pixel space* (x,y), regardless of which fields exist in the record.
    Accepts bx/by, bx_stab/by_stab, bx_raw/by_raw, or normalized u/v.
    """
    # priority: stabilized, then plain, then raw
    for kx, ky in (("bx_stab", "by_stab"), ("bx", "by"), ("bx_raw", "by_raw")):
        if kx in rec and ky in rec:
            return float(rec[kx]), float(rec[ky])

    # normalized fallback (0..1); tolerate slight overshoot
    if "u" in rec and "v" in rec:
        u = float(rec["u"])
        v = float(rec["v"])
        return max(0.0, min(1.0, u)) * (src_w - 1), max(0.0, min(1.0, v)) * (src_h - 1)

    # last resort: not found
    return None, None


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
    ap.add_argument("--ball-radius", type=int, default=8)
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

        row = rec

        # draw ball (red)
        bx, by = _get_ball_xy_src(rec, W, H)
        if bx is None or by is None:
            for key in ("ball", "telemetry_ball"):
                ball_val = rec.get(key)
                if isinstance(ball_val, (list, tuple)) and len(ball_val) >= 2:
                    try:
                        bx = float(ball_val[0])
                        by = float(ball_val[1])
                        if W > 1 and H > 1 and 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0:
                            bx = max(0.0, min(1.0, bx)) * (W - 1)
                            by = max(0.0, min(1.0, by)) * (H - 1)
                        break
                    except (TypeError, ValueError):
                        bx = by = None

        if bx is not None and by is not None:
            cv2.circle(
                frame,
                (int(round(bx)), int(round(by))),
                args.ball_radius,
                (0, 0, 255),
                args.thickness,
            )

        if "crop_src" in row and row["crop_src"]:
            cx, cy, cw, ch = row["crop_src"]
            p1 = (int(round(cx)), int(round(cy)))
            p2 = (int(round(cx + cw)), int(round(cy + ch)))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), args.thickness)

        vw.write(frame)
        idx += 1

    cap.release(); vw.release()
    print("Wrote", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
