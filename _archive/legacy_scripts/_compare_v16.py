import csv, os
import numpy as np

base = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
versions = [
    ("v13", "002__portrait_v13.diag.csv"),
    ("v15", "002__curated_v15.diag.csv"),
    ("v16", "002__manual_v16.diag.csv"),
]

print(f"{'Ver':<5} {'Mean':>6} {'Max':>6} {'P95':>6} {'Rev':>4} {'>30':>5} {'>15':>5} {'YOLO':>5} {'Trkr':>5} {'Cent':>5} {'Hold':>5} {'Intp':>5}")
print("-" * 75)

for label, fname in versions:
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        print(f"{label:<5} NOT FOUND")
        continue
    rows = list(csv.DictReader(open(path)))
    cam_cxs = [float(r["cam_cx"]) for r in rows]
    deltas = np.abs(np.diff(cam_cxs))
    sources = {}
    for r in rows:
        s = r.get("source", "?")
        sources[s] = sources.get(s, 0) + 1
    reversals = 0
    prev_dir = 0
    for i in range(1, len(cam_cxs)):
        dx = cam_cxs[i] - cam_cxs[i-1]
        d = 1 if dx > 2 else (-1 if dx < -2 else 0)
        if d != 0 and prev_dir != 0 and d != prev_dir:
            reversals += 1
        if d != 0:
            prev_dir = d
    print(f"{label:<5} {deltas.mean():>6.1f} {deltas.max():>6.1f} {np.percentile(deltas, 95):>6.1f} {reversals:>4} {(deltas>30).sum():>5} {(deltas>15).sum():>5} {sources.get('yolo',0):>5} {sources.get('tracker',0):>5} {sources.get('centroid',0):>5} {sources.get('hold',0):>5} {sources.get('interpolation',0):>5}")
