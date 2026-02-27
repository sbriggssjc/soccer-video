import csv, os
os.chdir(r'D:\Projects\soccer-video')

diag = 'out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__optflow_v1.diag.csv'

with open(diag) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("=== SHOT/SAVE ZONE (f310-f430) ===")
print(f"{'frame':>5} {'ball_x':>8} {'cam_cx':>8} {'source':>10} {'BIC':>4} {'dist':>6}")
for r in rows:
    fi = int(r['frame'])
    if 310 <= fi <= 430 and fi % 5 == 0:
        bx = float(r['ball_x'])
        cx = float(r['cam_cx'])
        bic = r['ball_in_crop']
        dist = float(r['dist_to_edge'])
        src = r['source']
        print(f"{fi:>5} {bx:>8.0f} {cx:>8.0f} {src:>10} {bic:>4} {dist:>6.0f}")

# Compare optflow vs camera at key points
print("\n=== TIMELINE COMPARISON ===")
print(f"{'frame':>5} {'ball_x':>8} {'cam_cx':>8} {'delta':>8} {'BIC':>4} {'source':>10}")
for r in rows:
    fi = int(r['frame'])
    if fi % 30 == 0:
        bx = float(r['ball_x'])
        cx = float(r['cam_cx'])
        delta = abs(bx - cx)
        bic = r['ball_in_crop']
        src = r['source']
        print(f"{fi:>5} {bx:>8.0f} {cx:>8.0f} {delta:>8.0f} {bic:>4} {src:>10}")

# Count BIC
total = len(rows)
in_crop = sum(1 for r in rows if r['ball_in_crop'] == '1')
print(f"\nBIC: {in_crop}/{total} ({100*in_crop/total:.1f}%)")

# Worst escapes
escapes = [(int(r['frame']), float(r['ball_x']), float(r['cam_cx']), float(r['dist_to_edge']), r['source']) 
           for r in rows if r['ball_in_crop'] == '0']
escapes.sort(key=lambda x: -abs(x[3]))
print(f"\nWorst 10 escapes:")
for fi, bx, cx, dist, src in escapes[:10]:
    print(f"  f{fi}: ball_x={bx:.0f}, cam_cx={cx:.0f}, dist={dist:.0f}, src={src}")
