import argparse, json, math


# --- robust outward nudge (Ceiling) + slightly bigger slack ---
SLACK = 8          # was 6
HALF = 0.65        # widen local warp window (seconds); was ~0.50–0.60


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


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
    inside = 0
    total = 0
    misses = []

    crop_w = args.crop_w
    crop_h = args.crop_h
    margin = args.margin
    W = args.w
    H = args.h

    for i, r in enumerate(recs):
        if r is None:
            continue
        if not all(k in r for k in ("cx", "cy", "bx", "by")):
            continue
        total += 1
        cx = float(r["cx"])
        cy = float(r["cy"])
        bx = float(r["bx"])
        by = float(r["by"])
        z_out = float(r.get("zoom_out", 1.0))
        z_in = float(r.get("zoom", 1.0))
        z_eff = max(z_out, 1.0 / max(z_in, 1e-6))
        z = max(1.0, z_eff)

        eff_w = min(W, crop_w * z)
        eff_h = min(H, crop_h * z)

        x0 = max(0.0, min(W - eff_w, cx - eff_w * 0.5))
        y0 = max(0.0, min(H - eff_h, cy - eff_h * 0.5))
        x1 = x0 + eff_w
        y1 = y0 + eff_h

        left = x0 + margin
        right = x1 - margin
        top = y0 + margin
        bottom = y1 - margin

        ok = bx >= left and bx <= right and by >= top and by <= bottom
        if ok:
            inside += 1
        else:
            t = float(r.get("t", i / 30.0))
            need_dx = 0
            need_dy = 0
            if bx < left:
                need_dx = int(math.ceil((left + SLACK) - bx))
            elif bx > right:
                need_dx = int(math.ceil((right - SLACK) - bx))

            if by < top:
                need_dy = int(math.ceil((top + SLACK) - by))
            elif by > bottom:
                need_dy = int(math.ceil((bottom - SLACK) - by))

            misses.append((i, t, bx, by, need_dx, need_dy))

    pct = 100.0 * inside / max(1, total)
    print(
        f"camera-coverage: {inside}/{total} = {pct:.2f}% (crop {args.crop_w}x{args.crop_h}, margin {args.margin}px; zoom-aware)"
    )
    if misses:
        i, t, bx, by, need_dx, need_dy = misses[0]
        print(
            "first miss @ frame"
            f" {i} t={t:.2f}s  ball=({bx:.1f},{by:.1f})"
            f" need_dx={need_dx} need_dy={need_dy}"
        )


if __name__ == "__main__":
    main()
