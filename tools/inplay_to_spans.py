import argparse, pandas as pd, numpy as np

def to_spans(df, thr=0.5, merge_gap=0.75, min_len=2.0, reset_pad=3.0, ignore_resets=True):
    """
    df columns: time (sec), in_play (0/1 or prob), reset (0/1)
    Returns DataFrame with start,end (seconds).
    """
    t = pd.to_numeric(df["time"], errors="coerce")
    p = pd.to_numeric(df["in_play"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(df.get("reset", 0), errors="coerce").fillna(0.0)

    # Optionally carve out a small window around reset==1 (kickoffs, etc.)
    if ignore_resets and "reset" in df.columns:
        # zero out 'in_play' within +/- reset_pad seconds of a reset pulse
        reset_times = t[r > 0.5].to_numpy()
        if reset_times.size:
            mask = np.ones(len(t), dtype=bool)
            for rt in reset_times:
                mask &= ~((t >= rt - reset_pad) & (t <= rt + reset_pad))
            p = p.where(mask, 0.0)

    # Binary gate from probability/score
    g = (p >= thr).astype(int)

    # Find rising/falling edges
    dg = np.diff(g, prepend=g.iloc[:1])
    starts = t[ dg ==  1 ].to_numpy()
    ends   = t[ dg == -1 ].to_numpy()

    # If started high and never ended, close at last timestamp
    if len(starts) and (len(ends) == 0 or starts[0] > ends[0]):
        ends = np.r_[ends, t.iloc[-1]]
    if len(ends) and len(starts) and ends[-1] < starts[-1]:
        ends = np.r_[ends, t.iloc[-1]]

    spans = list(zip(starts, ends))

    # Merge small gaps
    if spans:
        merged = [list(spans[0])]
        for a,b in spans[1:]:
            if a - merged[-1][1] <= merge_gap:
                merged[-1][1] = b
            else:
                merged.append([a,b])
        spans = merged

    # Enforce minimum length
    spans = [(a,b) for a,b in spans if b - a >= min_len]

    out = pd.DataFrame(spans, columns=["start","end"])
    return out.round(3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="out\\in_play.csv (time series)")
    ap.add_argument("--out", dest="out_csv", required=True, help="path to write start,end spans csv")
    ap.add_argument("--thr", type=float, default=0.5, help="in_play threshold (default 0.5)")
    ap.add_argument("--merge-gap", type=float, default=0.75, help="merge gaps <= this (sec)")
    ap.add_argument("--min-len", type=float, default=2.0, help="drop spans shorter than this (sec)")
    ap.add_argument("--reset-pad", type=float, default=3.0, help="seconds to blank around reset pulses")
    ap.add_argument("--keep-resets", action="store_true", help="do NOT blank around resets")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    out = to_spans(
        df,
        thr=args.thr,
        merge_gap=args.merge_gap,
        min_len=args.min_len,
        reset_pad=args.reset_pad,
        ignore_resets=not args.keep_resets,
    )
    if out.empty:
        raise SystemExit("No in-play spans produced; try lowering --thr or --min-len.")
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote spans: {args.out_csv} ({len(out)} rows)")

if __name__ == "__main__":
    main()

