import csv

def load_diag(path):
    return list(csv.DictReader(open(path)))

v21 = load_diag(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v21.diag.csv')
v22c = load_diag(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22c_planner.diag.csv')

print("=== v21 vs v22c Camera Comparison (every 30 frames) ===")
print(f"{'frame':>5} {'v21_cx':>8} {'v22c_cx':>8} {'v21_bic':>7} {'v22c_bic':>8} {'ball_x':>7} {'src':>10}")
for i in range(0, min(len(v21), len(v22c)), 30):
    r1, r2 = v21[i], v22c[i]
    print(f"{r1['frame']:>5} {float(r1['cam_cx']):>8.1f} {float(r2['cam_cx']):>8.1f} {r1['ball_in_crop']:>7} {r2['ball_in_crop']:>8} {r2['ball_x']:>7} {r2['source']:>10}")

v21_esc = sum(1 for r in v21 if r['ball_in_crop'] == '0')
v22c_esc = sum(1 for r in v22c if r['ball_in_crop'] == '0')
print(f"\n{'='*60}")
print(f"v21  escapes: {v21_esc}/{len(v21)} ({100*v21_esc/len(v21):.1f}%)")
print(f"v22c escapes: {v22c_esc}/{len(v22c)} ({100*v22c_esc/len(v22c):.1f}%)")
print(f"Improvement: {v21_esc - v22c_esc} fewer escape frames ({100*(v21_esc-v22c_esc)/v21_esc:.0f}% reduction)")

# Show remaining escapes in v22c
print(f"\n--- Remaining v22c escape frames ---")
for r in v22c:
    if r['ball_in_crop'] == '0':
        print(f"  f{r['frame']}: ball_x={r['ball_x']}, cam_cx={r['cam_cx']}, src={r['source']}, dist={r['dist_to_edge']}, flags={r['clamp_flags'][:80]}")

# Camera dynamics comparison
v21_cxs = [float(r['cam_cx']) for r in v21]
v22c_cxs = [float(r['cam_cx']) for r in v22c]
print(f"\nCamera dynamics:")
print(f"  v21  range: {min(v21_cxs):.0f}-{max(v21_cxs):.0f} (span={max(v21_cxs)-min(v21_cxs):.0f})")
print(f"  v22c range: {min(v22c_cxs):.0f}-{max(v22c_cxs):.0f} (span={max(v22c_cxs)-min(v22c_cxs):.0f})")

# v21 escape zones for comparison
print(f"\n--- v21 escape frames (for reference) ---")
esc_zones = []
in_zone = False
for r in v21:
    if r['ball_in_crop'] == '0':
        if not in_zone:
            esc_zones.append([r])
            in_zone = True
        else:
            esc_zones[-1].append(r)
    else:
        in_zone = False
for zone in esc_zones:
    f0, f1 = zone[0]['frame'], zone[-1]['frame']
    max_dist = max(abs(float(r['dist_to_edge'])) for r in zone)
    print(f"  frames {f0}-{f1} ({len(zone)}f): max escape={max_dist:.0f}px, src={zone[0]['source']}")
