import csv
rows = list(csv.reader(open(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22_fixes.diag.csv')))
print("=== v22 DIAGNOSTIC SUMMARY (every 15 frames) ===")
print(f"{'frame':>5} {'ball_x':>7} {'cam_cx':>7} {'bic':>3} {'src':>10} {'dist':>7} {'flags (first 100 chars)'}")
for r in rows[1::15]:
    print(f"{r[0]:>5} {r[1]:>7} {r[5]:>7} {r[11]:>3} {r[3]:>10} {r[12]:>7} {r[15][:100]}")
print()
# Escape stats
escapes = [r for r in rows[1:] if r[11] == '0']
print(f"Total escapes: {len(escapes)}/{len(rows)-1}")
# Camera range
cxs = [float(r[5]) for r in rows[1:]]
print(f"Camera cx range: {min(cxs):.0f} - {max(cxs):.0f} (span={max(cxs)-min(cxs):.0f})")
bxs = [float(r[1]) for r in rows[1:]]
print(f"Ball x range: {min(bxs):.0f} - {max(bxs):.0f} (span={max(bxs)-min(bxs):.0f})")
# Source breakdown
from collections import Counter
srcs = Counter(r[3] for r in rows[1:])
for s, c in srcs.most_common():
    esc = sum(1 for r in rows[1:] if r[3]==s and r[11]=='0')
    print(f"  {s}: {c} frames, {esc} escapes ({100*esc/c:.0f}%)")
