import csv
rows = list(csv.DictReader(open(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__v22c_planner.diag.csv')))

print("=== FRAME-BY-FRAME EVENT ANALYSIS (every 5 frames) ===")
print(f"{'f':>4} {'t':>5} {'bx':>6} {'cx':>6} {'gap':>5} {'bic':>3} {'src':>10} {'crop_w':>6} {'edge%':>6}")
for r in rows[::5]:
    f = int(r['frame'])
    bx = float(r['ball_x'])
    cx = float(r['cam_cx'])
    gap = bx - cx
    cw = float(r['crop_w'])
    half = cw / 2.0
    # How close to edge: 0% = centered, 100% = at edge
    edge_pct = abs(gap) / half * 100 if half > 0 else 0
    t = f / 30.0
    print(f"{f:>4} {t:>5.1f} {bx:>6.0f} {cx:>6.0f} {gap:>+5.0f} {r['ball_in_crop']:>3} {r['source']:>10} {cw:>6.0f} {edge_pct:>5.0f}%")

print("\n=== KEY MOMENTS (camera-ball gap > 40% of half-crop) ===")
for r in rows:
    f = int(r['frame'])
    bx = float(r['ball_x'])
    cx = float(r['cam_cx'])
    cw = float(r['crop_w'])
    half = cw / 2.0
    gap = abs(bx - cx)
    edge_pct = gap / half * 100 if half > 0 else 0
    if edge_pct > 40:
        t = f / 30.0
        print(f"  f{f:>3} t={t:.1f}s: ball={bx:.0f} cam={cx:.0f} gap={bx-cx:+.0f}px ({edge_pct:.0f}% to edge) src={r['source']}")
