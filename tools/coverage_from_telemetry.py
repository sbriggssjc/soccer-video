import argparse
import json
import math


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            o = json.loads(ln)
            t = float(o.get("t", 0.0))
            cx = float(o.get("cx"))
            cy = float(o.get("cy"))
            bx = float(o.get("bx"))
            by = float(o.get("by"))
            rows.append((t, cx, cy, bx, by))
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

    rows = load_jsonl(args.telemetry)
    inside = 0
    misses = []
    for i, (t, cx, cy, bx, by) in enumerate(rows):
        x0 = cx - args.crop_w / 2
        y0 = cy - args.crop_h / 2
        x1 = x0 + args.crop_w
        y1 = y0 + args.crop_h
        ok = (
            bx >= x0 + args.margin
            and bx <= x1 - args.margin
            and by >= y0 + args.margin
            and by <= y1 - args.margin
        )
        if ok:
            inside += 1
        else:
            misses.append((i, t, bx, by))
    total = max(1, len(rows))
    pct = 100.0 * inside / total
    print(
        f"camera-coverage: {inside}/{total} = {pct:.2f}% (crop {args.crop_w}x{args.crop_h}, margin {args.margin}px)"
    )
    if misses:
        i, t, bx, by = misses[0]
        print(f"first miss @ frame {i} t={t:.2f}s  ball=({bx:.1f},{by:.1f})")


if __name__ == "__main__":
    main()
