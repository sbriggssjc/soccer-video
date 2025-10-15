import argparse, json, math, sys
from statistics import median


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            t = o.get("t", None)
            bx = o.get("bx_stab", o.get("bx"))
            by = o.get("by_stab", o.get("by"))
            if bx is None or by is None:
                rows.append((t, None, None))
            else:
                if isinstance(bx, float) and math.isnan(bx):
                    bx = None
                if isinstance(by, float) and math.isnan(by):
                    by = None
                rows.append((t, bx, by))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for t, bx, by in rows:
            o = {
                "t": float(t),
                "bx": float(bx),
                "by": float(by),
                "bx_stab": float(bx),
                "by_stab": float(by),
            }
            f.write(json.dumps(o, separators=(",", ":")) + "\n")


def ema(seq, alpha):
    out = []
    last = None
    for v in seq:
        if v is None:
            out.append(last)
            continue
        last = v if last is None else (alpha * v + (1 - alpha) * last)
        out.append(last)
    return out


def clamp_spikes(xs, ys, px_thr):
    # limit per-frame jump to px_thr by linear pull-back
    xs2 = [xs[0]]
    ys2 = [ys[0]]
    for i in range(1, len(xs)):
        x0, y0 = xs2[-1], ys2[-1]
        x1, y1 = xs[i], ys[i]
        if x1 is None or y1 is None:
            xs2.append(x0)
            ys2.append(y0)
            continue
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d > px_thr and d > 0:
            s = px_thr / d
            xs2.append(x0 + dx * s)
            ys2.append(y0 + dy * s)
        else:
            xs2.append(x1)
            ys2.append(y1)
    return xs2, ys2


def forward_fill(xs):
    last = None
    out = []
    for v in xs:
        if v is None and last is not None:
            out.append(last)
        else:
            out.append(v)
            if v is not None:
                last = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--lead", type=float, default=0.10, help="predictive lead seconds")
    ap.add_argument("--spike-thr", type=float, default=22.0, help="max px/frame")
    ap.add_argument("--smooth", type=float, default=0.25, help="EMA alpha (0..1), higher = smoother")
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    rows = load_jsonl(args.inp)
    # derive uniform timebase if missing/irregular
    # assume rows are per-frame; rebuild t from fps when needed
    uniform = []
    t0 = rows[0][0] if rows[0][0] is not None else 0.0
    for i, (t, bx, by) in enumerate(rows):
        t_use = (i / args.fps) if t is None else t
        uniform.append((t_use, bx, by))

    t = [r[0] for r in uniform]
    x = [r[1] for r in uniform]
    y = [r[2] for r in uniform]

    # fill gaps then clamp spikes, then smooth
    x = forward_fill(x)
    y = forward_fill(y)
    x, y = clamp_spikes(x, y, args.spike_thr)
    x = ema(x, args.smooth)
    y = ema(y, args.smooth)

    # predictive lead (velocity * lead_time)
    vx = [0.0] + [x[i] - x[i - 1] for i in range(1, len(x))]
    vy = [0.0] + [y[i] - y[i - 1] for i in range(1, len(y))]
    lead_frames = max(0, int(round(args.lead * args.fps)))
    if lead_frames > 0:
        x_lead = [x[i] + vx[i] * lead_frames for i in range(len(x))]
        y_lead = [y[i] + vy[i] * lead_frames for i in range(len(y))]
    else:
        x_lead = x
        y_lead = y

    out_rows = list(zip(t, x_lead, y_lead))
    write_jsonl(args.outp, out_rows)


if __name__ == "__main__":
    main()
