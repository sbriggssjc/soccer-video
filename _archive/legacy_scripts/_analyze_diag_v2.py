import csv

diag = 'out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__optflow_v2.diag.csv'
rows = []
with open(diag) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

total = len(rows)
bic = sum(1 for r in rows if r['ball_in_crop'] == '1')
sl = sum(1 for r in rows if r['speed_limited'] == '1')
print(f'Total frames: {total}')
print(f'Ball in crop: {bic}/{total} ({100*bic/total:.1f}%)')
print(f'Speed limited: {sl}/{total} ({100*sl/total:.1f}%)')

# Key zones
print('\n=== Shot zone (f290-f340) ===')
for r in rows:
    fi = int(r['frame'])
    if 290 <= fi <= 340 and fi % 5 == 0:
        bx = float(r['ball_x'])
        cx = float(r['cam_cx'])
        bic_flag = r['ball_in_crop']
        sl_flag = 'SL' if r['speed_limited'] == '1' else ''
        dist = float(r['dist_to_edge']) if r['dist_to_edge'] else 0
        print(f'  f{fi}: ball_x={bx:.0f} cam_cx={cx:.0f} delta={abs(bx-cx):.0f} BIC={bic_flag} {sl_flag} dist_edge={dist:.0f}')

print('\n=== Restart transition (f410-f460) ===')
for r in rows:
    fi = int(r['frame'])
    if 410 <= fi <= 460 and fi % 5 == 0:
        bx = float(r['ball_x'])
        cx = float(r['cam_cx'])
        bic_flag = r['ball_in_crop']
        sl_flag = 'SL' if r['speed_limited'] == '1' else ''
        dist = float(r['dist_to_edge']) if r['dist_to_edge'] else 0
        print(f'  f{fi}: ball_x={bx:.0f} cam_cx={cx:.0f} delta={abs(bx-cx):.0f} BIC={bic_flag} {sl_flag} dist_edge={dist:.0f}')

# Worst escapes (ball_in_crop=0)
print('\n=== Frames where ball NOT in crop ===')
escapes = [(int(r['frame']), float(r['ball_x']), float(r['cam_cx'])) for r in rows if r['ball_in_crop'] == '0']
for fi, bx, cx in escapes[:30]:
    print(f'  f{fi}: ball_x={bx:.0f} cam_cx={cx:.0f} delta={abs(bx-cx):.0f}')
print(f'  Total: {len(escapes)} frames out of crop')
