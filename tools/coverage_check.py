import argparse
import json


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = obj.get("t", 0.0)
            bx = obj.get("bx_stab", obj.get("bx"))
            by = obj.get("by_stab", obj.get("by"))
            rows.append((t, float(bx), float(by)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ball", required=True, help="ball plan jsonl (bx/by or bx_stab/by_stab)")
    parser.add_argument("--w", type=int, default=1920, help="atomic width")
    parser.add_argument("--h", type=int, default=1080, help="atomic height")
    parser.add_argument(
        "--crop-w",
        type=int,
        default=486,
        help="portrait crop width in source pixels (from your logs)",
    )
    parser.add_argument(
        "--crop-h",
        type=int,
        default=864,
        help="portrait crop height in source pixels",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=80,
        help="safety px around ball inside crop",
    )
    parser.add_argument("--smooth", type=float, default=0.25)
    args = parser.parse_args()

    pts = load_jsonl(args.ball)
    # simple EMA on center
    cx = []
    cy = []
    lastx = None
    lasty = None
    alpha = args.smooth
    for _, bx, by in pts:
        if lastx is None:
            lastx, lasty = bx, by
        else:
            lastx = alpha * bx + (1 - alpha) * lastx
            lasty = alpha * by + (1 - alpha) * lasty
        # clamp center so crop stays in bounds
        x0 = max(0, min(args.w - args.crop_w, lastx - args.crop_w / 2))
        y0 = max(0, min(args.h - args.crop_h, lasty - args.crop_h / 2))
        cx.append(x0 + args.crop_w / 2)
        cy.append(y0 + args.crop_h / 2)

    inside = 0
    total = len(pts)
    misses = []
    for i, (t, bx, by) in enumerate(pts):
        x0 = cx[i] - args.crop_w / 2
        x1 = cx[i] + args.crop_w / 2
        y0 = cy[i] - args.crop_h / 2
        y1 = cy[i] + args.crop_h / 2
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
    pct = 100.0 * inside / max(1, total)
    print(
        f"coverage: {inside}/{total} = {pct:.2f}% (crop {args.crop_w}x{args.crop_h}, margin {args.margin}px, smooth {args.smooth})"
    )
    if misses:
        miss = misses[0]
        print(
            f"first miss @ frame {miss[0]} t={miss[1]:.2f}s  ball=({miss[2]:.1f},{miss[3]:.1f})"
        )


if __name__ == "__main__":
    main()
