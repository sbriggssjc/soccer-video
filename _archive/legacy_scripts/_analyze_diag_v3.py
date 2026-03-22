import csv

for ver, path in [('v2', 'out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__optflow_v2.diag.csv'),
                   ('v3', 'out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__optflow_v3.diag.csv')]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    bic = sum(1 for r in rows if r['ball_in_crop'] == '1')
    sl = sum(1 for r in rows if r['speed_limited'] == '1')
    escapes = [(int(r['frame']), float(r['ball_x']), float(r['cam_cx'])) for r in rows if r['ball_in_crop'] == '0']
    worst = max(escapes, key=lambda e: abs(e[1]-e[2])) if escapes else (0, 0, 0)
    print(f"\n=== {ver} ===")
    print(f"  BIC: {bic}/{len(rows)} ({100*bic/len(rows):.1f}%)")
    print(f"  Speed limited: {sl}/{len(rows)} ({100*sl/len(rows):.1f}%)")
    print(f"  Escapes: {len(escapes)} frames")
    print(f"  Worst: f{worst[0]} delta={abs(worst[1]-worst[2]):.0f}px")

# v3 escape details
print("\n=== v3 Escape frames ===")
with open('out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__optflow_v3.diag.csv') as f:
    for r in csv.DictReader(f):
        if r['ball_in_crop'] == '0':
            fi = int(r['frame'])
            bx = float(r['ball_x'])
            cx = float(r['cam_cx'])
            print(f"  f{fi}: ball_x={bx:.0f} cam_cx={cx:.0f} delta={abs(bx-cx):.0f}")
