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
    ap.add_argument(
        "--show-plan-text",
        action="store_true",
        help="Render debug text with frame index, plan time, and coordinates",
    )
    ap.add_argument(
        "--show-states",
        action="store_true",
        help="If a record contains a 'state' field (offscreen/occluded/unknown), render a small label.",
    )
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
            if 0.0 <= bx <= 1.0 and 0.0 <= by <= 1.0:
                bx *= W
                by *= H
            cv2.circle(
                frame,
                (int(round(bx)), int(round(by))),
                args.ball_radius,
                (0, 0, 255),
                args.thickness,
            )

        if args.show_plan_text:
            plan_t = rec.get("t") if isinstance(rec, dict) else None
            if isinstance(plan_t, (int, float)):
                plan_t_val = float(plan_t)
                plan_t_str = f"{plan_t_val:.3f}s"
            else:
                plan_t_val = None
                plan_t_str = "?"

            actual_t = idx / float(fps) if fps else 0.0

            def _extract_plan_xy(record):
                if isinstance(record, dict):
                    for key_x, key_y in ("bx_stab", "by_stab"), ("bx", "by"), ("bx_raw", "by_raw"):
                        if key_x in record and key_y in record:
                            try:
                                return float(record[key_x]), float(record[key_y])
                            except (TypeError, ValueError):
                                continue
                return None, None

            plan_bx, plan_by = _extract_plan_xy(rec)
            if plan_bx is not None and plan_by is not None and 0.0 <= plan_bx <= 1.0 and 0.0 <= plan_by <= 1.0:
                plan_bx_disp = plan_bx * (W - 1)
                plan_by_disp = plan_by * (H - 1)
            else:
                plan_bx_disp = plan_bx
                plan_by_disp = plan_by

            bx_str = f"{plan_bx_disp:.1f}" if plan_bx_disp is not None else "?"
            by_str = f"{plan_by_disp:.1f}" if plan_by_disp is not None else "?"

            text = f"f={idx} t_plan={plan_t_str} t_frame={actual_t:.3f}s bx={bx_str} by={by_str}"
            cv2.putText(
                frame,
                text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                3,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        if args.show_states:
            state_val = rec.get("state") if isinstance(rec, dict) else None
            state_str = str(state_val).strip().lower() if state_val is not None else ""
            if state_str in {"offscreen", "occluded", "unknown"}:
                label = f"[{state_str}]"
                cv2.putText(
                    frame,
                    label,
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    label,
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
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
