import os, csv, glob

# Check _tmp for CSVs with filled data
tmp = r"D:\Projects\soccer-video\_tmp"
for n in ["001","002","003","004","005","006","007","008","009","010","011","012","013","014","015"]:
    path = os.path.join(tmp, f"review_{n}.csv")
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        filled = sum(1 for r in rows if r["camera_x_pct"].strip())
        print(f"{n}: {len(rows)} rows, {filled} filled")
    else:
        # Check Desktop
        dp = rf"C:\Users\scott\Desktop\review_{n}.csv"
        if os.path.exists(dp):
            with open(dp) as f:
                rows = list(csv.DictReader(f))
            filled = sum(1 for r in rows if r["camera_x_pct"].strip())
            print(f"{n}: {len(rows)} rows, {filled} filled (Desktop)")
        else:
            print(f"{n}: NOT FOUND")
