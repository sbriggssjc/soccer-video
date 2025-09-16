import argparse, pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtered", required=True)     # highlights_filtered.csv (has action_score/team_presence if you kept that)
    ap.add_argument("--goals", required=True)        # forced_goals.csv from 07_force_goals.py
    ap.add_argument("--out", required=True)          # highlights_top10.csv
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--merge_tol", type=float, default=2.0, help="seconds; de-dup by start time proximity")
    ap.add_argument("--prefer-team", type=int, default=1)
    args = ap.parse_args()

    f = pd.read_csv(args.filtered) if len(open(args.filtered).read().strip()) else pd.DataFrame()
    g = pd.read_csv(args.goals)    if len(open(args.goals).read().strip())    else pd.DataFrame()

    rows = []
    if len(g):
        g["priority"] = 0  # goals first
        rows.append(g[["start","end","score","priority"]])
    if len(f):
        f = f.copy()
        if "action_score" in f.columns: f["score"] = f["action_score"]
        if args.prefer_team and "team_presence" in f.columns:
            f["score"] = f["score"] + 0.25 * f["team_presence"]
        f["priority"] = 1
        rows.append(f[["start","end","score","priority"]])

    if not rows:
        raise SystemExit("No input rows; check filtered/goals CSVs")

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["priority","score"], ascending=[True, False])

    # de-dup near-identical clips by start time
    out = []
    used = []
    for _, r in df.iterrows():
        s = float(r["start"])
        if any(abs(s - u) < args.merge_tol for u in used): 
            continue
        used.append(s)
        out.append({"start": r["start"], "end": r["end"], "score": r["score"]})
        if len(out) >= args.n: break

    pd.DataFrame(out).to_csv(args.out, index=False)
    print(f"[build_top10] wrote {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
