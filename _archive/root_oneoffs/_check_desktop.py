import glob
csvs = sorted(glob.glob(r"C:\Users\scott\Desktop\review_*.csv"))
pngs = sorted(glob.glob(r"C:\Users\scott\Desktop\filmstrip_*.png"))
print(f"CSVs: {len(csvs)}")
for c in csvs: print(f"  {c.split(chr(92))[-1]}")
print(f"PNGs: {len(pngs)}")
for p in pngs: print(f"  {p.split(chr(92))[-1]}")
