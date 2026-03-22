"""Compare v11 (cleaned) vs v10 (consensus) vs original pipeline."""
import csv, os

def load_diag(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

base = r"D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC"

files = {
    "v11_cleaned": os.path.join(base, "002__cleaned_v11.diag.csv"),
    "v10_consensus": os.path.join(base, "002__consensus_v10.diag.csv"),
}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"  {name}: NOT FOUND at {path}")
        continue
    rows = load_diag(path)
    n = len(rows)
    bic = sum(1 for r in rows if r.get("ball_in_crop","").strip().lower() == "true")
    sources = {}
    for r in rows:
        s = r.get("source","?").strip()
        sources[s] = sources.get(s, 0) + 1

    cxs = [float(r["cam_cx"]) for r in rows if r.get("cam_cx")]
    cx_range = max(cxs) - min(cxs) if cxs else 0

    # Count camera reversals
    reversals = 0
    for i in range(2, len(cxs)):
        d1 = cxs[i-1] - cxs[i-2]
        d2 = cxs[i] - cxs[i-1]
        if d1 * d2 < 0 and abs(d1) > 5 and abs(d2) > 5:
            reversals += 1

    # Count large camera jumps (>30px/frame)
    big_jumps = 0
    for i in range(1, len(cxs)):
        if abs(cxs[i] - cxs[i-1]) > 30:
            big_jumps += 1

    # Jitter: mean absolute frame-to-frame camera change
    deltas = [abs(cxs[i] - cxs[i-1]) for i in range(1, len(cxs))]
    mean_delta = sum(deltas) / len(deltas) if deltas else 0

    print(f"\n=== {name} ===")
    print(f"  Frames: {n}")
    print(f"  BIC: {bic}/{n} ({100*bic/n:.1f}%)")
    print(f"  Sources: {sources}")
    print(f"  CX range: {min(cxs):.0f}-{max(cxs):.0f} ({cx_range:.0f}px)")
    print(f"  Camera reversals (>5px): {reversals}")
    print(f"  Big jumps (>30px/f): {big_jumps}")
    print(f"  Mean camera delta: {mean_delta:.1f}px/f")
    print(f"  YOLO frames: {sources.get('yolo', 0)}")
    print(f"  Centroid frames: {sources.get('centroid', 0)}")
    print(f"  Interp frames: {sources.get('interp', 0)}")
    print(f"  Hold frames: {sources.get('hold', 0) + sources.get('shot_hold', 0)}")
    print(f"  Tracker frames: {sources.get('tracker', 0)}")
