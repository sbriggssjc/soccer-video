import csv

def load_diag(path):
    rows = list(csv.DictReader(open(path)))
    return rows

v21 = load_diag(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v21.diag.csv')
v22 = load_diag(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22_fixes.diag.csv')

print("=== v21 vs v22 Camera Comparison (every 30 frames) ===")
print(f"{'frame':>5} {'v21_cx':>8} {'v22_cx':>8} {'delta':>7} {'v21_bic':>7} {'v22_bic':>7} {'ball_x':>7}")
for i in range(0, min(len(v21), len(v22)), 30):
    r1, r2 = v21[i], v22[i]
    cx1, cx2 = float(r1['cam_cx']), float(r2['cam_cx'])
    print(f"{r1['frame']:>5} {cx1:>8.1f} {cx2:>8.1f} {cx2-cx1:>7.1f} {r1['ball_in_crop']:>7} {r2['ball_in_crop']:>7} {r2['ball_x']:>7}")

v21_esc = sum(1 for r in v21 if r['ball_in_crop'] == '0')
v22_esc = sum(1 for r in v22 if r['ball_in_crop'] == '0')
print(f"\nv21 escapes: {v21_esc}/{len(v21)} ({100*v21_esc/len(v21):.1f}%)")
print(f"v22 escapes: {v22_esc}/{len(v22)} ({100*v22_esc/len(v22):.1f}%)")

v21_cxs = [float(r['cam_cx']) for r in v21]
v22_cxs = [float(r['cam_cx']) for r in v22]
print(f"v21 cam range: {min(v21_cxs):.0f}-{max(v21_cxs):.0f} (span={max(v21_cxs)-min(v21_cxs):.0f})")
print(f"v22 cam range: {min(v22_cxs):.0f}-{max(v22_cxs):.0f} (span={max(v22_cxs)-min(v22_cxs):.0f})")
