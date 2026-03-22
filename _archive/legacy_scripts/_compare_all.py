import csv, os

base = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
versions = {
    "v13_portrait":  "002__portrait_v13.diag.csv",
    "v12_polished":  "002__polished_v12.diag.csv",
    "v11_cleaned":   "002__cleaned_v11.diag.csv",
    "v10_consensus": "002__consensus_v10.diag.csv",
}

for name, fname in versions.items():
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    cxs = [float(r["cam_cx"]) for r in rows]
    deltas = [abs(cxs[i] - cxs[i-1]) for i in range(1, len(cxs))]
    revs = sum(1 for i in range(2, len(cxs))
               if (cxs[i-1]-cxs[i-2])*(cxs[i]-cxs[i-1]) < 0
               and abs(cxs[i-1]-cxs[i-2]) > 5
               and abs(cxs[i]-cxs[i-1]) > 5)
    bj = sum(1 for d in deltas if d > 30)
    md = sum(deltas) / len(deltas)
    srcs = {}
    for r in rows:
        s = r.get("source", "?").strip()
        srcs[s] = srcs.get(s, 0) + 1

    print(f"\n=== {name} ===")
    print(f"  Mean delta: {md:.1f}px/f, Big jumps(>30px/f): {bj}, Reversals: {revs}")
    print(f"  CX range: {min(cxs):.0f}-{max(cxs):.0f} ({max(cxs)-min(cxs):.0f}px)")
    print(f"  Sources: {srcs}")
    print(f"  YOLO={srcs.get('yolo',0)}, Tracker={srcs.get('tracker',0)}, Interp={srcs.get('interp',0)}, Hold={srcs.get('hold',0)+srcs.get('shot_hold',0)}, Centroid={srcs.get('centroid',0)}")
