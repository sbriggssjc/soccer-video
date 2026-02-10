import argparse, csv

def read_csv(p):
    with open(p, newline="") as f:
        return [{k: v for k,v in r.items()} for r in csv.DictReader(f)]

def iou(a, b):
    # a,b are (start,end) in seconds
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    uni   = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/uni if uni>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)   # base CSV
    ap.add_argument("--b", required=True)   # additions CSV
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-iou", type=float, default=0.5)
    args = ap.parse_args()

    A = read_csv(args.a)
    B = read_csv(args.b)
    # Convert to floats and sort by score desc
    for R in (A+B):
        R["start"] = float(R["start"]); R["end"] = float(R["end"]); R["score"] = float(R["score"])
    all_rows = sorted(A+B, key=lambda r: r["score"], reverse=True)

    kept = []
    for r in all_rows:
        interval = (r["start"], r["end"])
        if any(iou(interval, (k["start"], k["end"])) >= args.min_iou for k in kept):
            continue
        kept.append(r)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["start","end","score"])
        w.writeheader()
        for r in kept:
            w.writerow({k: f"{r[k]:.2f}" if k!="score" else f"{r[k]:.3f}" for k in ["start","end","score"]})

if __name__ == "__main__":
    main()
