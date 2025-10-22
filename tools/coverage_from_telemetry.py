import argparse, json, math


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
        s = float(r.get("zoom_out", 1.0))

        # effective crop in SOURCE pixels during this frame
        eff_w = args.crop_w * s
        eff_h = args.crop_h * s
        hx = 0.5 * eff_w
        hy = 0.5 * eff_h

        x0 = cx - hx
        x1 = cx + hx
        y0 = cy - hy
        y1 = cy + hy

        ok = (
            bx >= x0 + args.margin
            and bx <= x1 - args.margin
            and by >= y0 + args.margin
            and by <= y1 - args.margin
        )
        if ok:
            inside += 1
        else:
            t = float(r.get("t", i / 30.0))
            misses.append((i, t, bx, by))

    pct = 100.0 * inside / max(1, total)
    print(
        f"camera-coverage: {inside}/{total} = {pct:.2f}% (crop {args.crop_w}x{args.crop_h}, margin {args.margin}px)"
    )
    if misses:
        i, t, bx, by = misses[0]
        print(f"first miss @ frame {i} t={t:.2f}s  ball=({bx:.1f},{by:.1f})")


if __name__ == "__main__":
    main()
