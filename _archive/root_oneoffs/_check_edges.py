import csv
clips = [f"{i:03d}" for i in range(1, 16)]
for c in clips:
    path = rf"C:\Users\scott\Desktop\review_{c}.csv"
    vals = [float(r["camera_x_pct"]) for r in csv.DictReader(open(path)) if r["camera_x_pct"].strip()]
    mn, mx = min(vals), max(vals)
    edge = " ** EDGE" if mn <= 1 or mx >= 99 else ""
    print(f"{c}: min={mn:.1f}  max={mx:.1f}{edge}")
