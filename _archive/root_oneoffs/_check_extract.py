import os, csv, glob

csvs = sorted(glob.glob(r"C:\Users\scott\Desktop\review_*.csv"))
print(f"Desktop CSVs: {len(csvs)}")
for c in csvs:
    num = c.split("review_")[1].split(".csv")[0]
    with open(c) as f:
        rows = list(csv.DictReader(f))
    filled = sum(1 for r in rows if r.get("camera_x_pct","").strip())
    confs = [float(r["confidence"]) for r in rows if r.get("confidence","").strip()]
    avg_c = sum(confs)/len(confs) if confs else 0
    low = sum(1 for c in confs if c < 0.5)
    print(f"  {num}: {filled} anchors, avg_conf={avg_c:.3f}, {low} below 0.5")
