import argparse, json, math

def load_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows

def pick2(rec, cands):
    """Return first present key from cands in rec, else None."""
    for k in cands:
        if k in rec and rec[k] is not None:
            return float(rec[k])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--crop-w", type=int, default=486)
    ap.add_argument("--crop-h", type=int, default=864)
    ap.add_argument("--margin", type=int, default=90)
    args = ap.parse_args()

    recs = load_jsonl(args.telemetry)
    if not recs:
        print("camera-coverage: 0/0 = 0.00% (no telemetry)")
        return

    # Accept multiple spellings
    cam_x_keys = ("cx","cam_cx","center_x","camx")
    cam_y_keys = ("cy","cam_cy","center_y","camy")
    ball_x_keys = ("bx_stab","bx","bx_raw","u")
    ball_y_keys = ("by_stab","by","by_raw","v")

    inside = 0
    total = 0
    misses = []

    # If telemetry uses normalized u,v for the ball, scale to pixels.
    def to_px_x(v): return max(0.0, min(1.0, float(v)))*(args.w-1)
    def to_px_y(v): return max(0.0, min(1.0, float(v)))*(args.h-1)

    for i, r in enumerate(recs):
        cx = pick2(r, cam_x_keys)
        cy = pick2(r, cam_y_keys)
        bx = pick2(r, ball_x_keys)
        by = pick2(r, ball_y_keys)

        # Support normalized u,v for ball
        if bx is None and "u" in r: bx = to_px_x(r["u"])
        if by is None and "v" in r: by = to_px_y(r["v"])

        # Skip frames without camera or ball
        if cx is None or cy is None or bx is None or by is None:
            continue

        total += 1
        s = float(r.get("zoom_out", 1.0))
        hx = 0.5 * args.crop_w * s
        hy = 0.5 * args.crop_h * s

        x0 = cx - hx
        x1 = cx + hx
        y0 = cy - hy
        y1 = cy + hy

        ok = (bx >= x0 + args.margin and bx <= x1 - args.margin and
              by >= y0 + args.margin and by <= y1 - args.margin)
        if ok:
            inside += 1
        else:
            t = r.get("t", i/30.0)
            misses.append((i, t, bx, by))

    pct = 100.0*inside/max(1,total)
    print(f"camera-coverage: {inside}/{total} = {pct:.2f}% (crop {args.crop_w}x{args.crop_h}, margin {args.margin}px)")
    if misses:
        m = misses[0]
        print(f"first miss @ frame {m[0]} t={m[1]:.2f}s  ball=({m[2]:.1f},{m[3]:.1f})")

if __name__ == "__main__":
    main()
