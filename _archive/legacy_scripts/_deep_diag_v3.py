"""Deep diagnostic: compare ball position, camera position, crop geometry, 
and identify where the camera is NOT centered on ball."""
import csv, json, os

os.chdir(r'D:\Projects\soccer-video')
CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'

# Load v3 diag
rows = []
with open('out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__optflow_v3.diag.csv') as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Load v3 render log camera + ball timeline
print("=== v3 Render: Camera vs Ball (every 10 frames) ===")
print(f"{'frame':>5} {'ball_x':>7} {'cam_cx':>7} {'delta':>6} {'crop_w':>7} {'crop_x0':>7} {'crop_x1':>7} {'BIC':>4} {'dist_edge':>9} {'SL':>3}")
for r in rows:
    fi = int(r['frame'])
    if fi % 10 != 0: continue
    bx = float(r['ball_x'])
    cx = float(r['cam_cx'])
    cw = float(r['crop_w'])
    cx0 = float(r['crop_x0'])
    cx1 = cx0 + cw
    bic = r['ball_in_crop']
    dist = float(r['dist_to_edge'])
    sl = 'SL' if r['speed_limited'] == '1' else ''
    delta = bx - cx
    print(f"  f{fi:>3}: {bx:>7.0f} {cx:>7.0f} {delta:>+6.0f} {cw:>7.0f} {cx0:>7.0f} {cx1:>7.0f}  {bic:>3}  {dist:>8.0f}  {sl}")

# Summary stats
print(f"\n=== Crop width stats ===")
widths = [float(r['crop_w']) for r in rows]
print(f"  Min crop_w: {min(widths):.0f}")
print(f"  Max crop_w: {max(widths):.0f}")
print(f"  Mean crop_w: {sum(widths)/len(widths):.0f}")

# How many frames is camera within 100px of ball?
within100 = sum(1 for r in rows if abs(float(r['ball_x']) - float(r['cam_cx'])) < 100)
within200 = sum(1 for r in rows if abs(float(r['ball_x']) - float(r['cam_cx'])) < 200)
within50 = sum(1 for r in rows if abs(float(r['ball_x']) - float(r['cam_cx'])) < 50)
print(f"\n=== Camera centering quality ===")
print(f"  Within 50px of ball: {within50}/{len(rows)} ({100*within50/len(rows):.0f}%)")
print(f"  Within 100px of ball: {within100}/{len(rows)} ({100*within100/len(rows):.0f}%)")
print(f"  Within 200px of ball: {within200}/{len(rows)} ({100*within200/len(rows):.0f}%)")

# Where is camera stuck/lagging?
print(f"\n=== Camera stuck periods (same cx for 5+ frames) ===")
streak_start = 0
for i in range(1, len(rows)):
    cx_prev = float(rows[i-1]['cam_cx'])
    cx_cur = float(rows[i]['cam_cx'])
    if abs(cx_cur - cx_prev) < 2:  # stuck
        continue
    else:
        streak_len = i - streak_start
        if streak_len >= 5:
            cx_val = float(rows[streak_start]['cam_cx'])
            ball_range = [float(rows[j]['ball_x']) for j in range(streak_start, i)]
            print(f"  f{int(rows[streak_start]['frame'])}-f{int(rows[i-1]['frame'])}: cam stuck at {cx_val:.0f} for {streak_len}f, ball range=[{min(ball_range):.0f}, {max(ball_range):.0f}]")
        streak_start = i
