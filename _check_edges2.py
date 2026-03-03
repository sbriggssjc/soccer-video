import csv
clips = ["016","017","018","019","020","022","023","024","025","026","027"]
for c in clips:
    path = rf"C:\Users\scott\Desktop\review_{c}.csv"
    vals = [float(r["camera_x_pct"]) for r in csv.DictReader(open(path)) if r["camera_x_pct"].strip()]
    if not vals:
        print(f"{c}: EMPTY (no anchors filled)")
        continue
    mn, mx = min(vals), max(vals)
    edge = " ** EDGE (zoom=1)" if mn <= 1 or mx >= 99 else ""
    print(f"{c}: min={mn:.1f}  max={mx:.1f}{edge}")
