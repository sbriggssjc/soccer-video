import csv, os
import numpy as np

base = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"
versions = [
    ("v11", "002__cleaned_v11.diag.csv"),
    ("v12", "002__polished_v12.diag.csv"),
    ("v13", "002__portrait_v13.diag.csv"),
    ("v15", "002__curated_v15.diag.csv"),
]

print(f"{'Ver':<5} {'Mean':>6} {'Max':>6} {'P95':>6} {'Rev':>4} {'Big>30':>6} {'YOLO':>5} {'Trkr':>5} {'Cent':>5} {'Hold':>5} {'Intp':>5}")
print("-" * 70)

for label, fname in versions:
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        print(f"{label:<5} FILE NOT FOUND: {fname}")
        continue
    rows = list(csv.DictReader(open(path)))
    deltas = []
    sources = {"yolo": 0, "tracker": 0, "centroid": 0, "hold": 0, "interpolation": 0}
    reversals = 0
    big_jumps = 0
    prev_dir = 0
    cam_cxs = [float(r["cam_cx"]) for r in rows]
    for i in range(1, len(cam_cxs)):
        d = abs(cam_cxs[i] - cam_cxs[i-1])
        deltas.append(d)
        if d > 30:
            big_jumps += 1
        dx = cam_cxs[i] - cam_cxs[i-1]
        direction = 1 if dx > 2 else (-1 if dx < -2 else 0)
        if direction != 0 and prev_dir != 0 and direction != prev_dir:
            reversals += 1
        if direction != 0:
            prev_dir = direction
    for r in rows:
        src = r.get("source", "")
        if src in sources:
            sources[src] += 1
    deltas = np.array(deltas)
    print(f"{label:<5} {deltas.mean():>6.1f} {deltas.max():>6.1f} {np.percentile(deltas, 95):>6.1f} {reversals:>4} {big_jumps:>6} {sources['yolo']:>5} {sources['tracker']:>5} {sources['centroid']:>5} {sources['hold']:>5} {sources['interpolation']:>5}")
