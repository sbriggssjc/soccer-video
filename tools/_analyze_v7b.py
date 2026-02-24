"""Temporary analysis script for v7b diagnostic CSV."""
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\001__fix_v7b_confweight.diag.csv')
n = len(df)
bx = df['ball_x'].values
by = df['ball_y'].values
cx = df['cam_cx'].values
conf = df['confidence'].values
src = df['source'].values
crop_x0 = df['crop_x0'].values
crop_w = df['crop_w'].values

left_margin = bx - crop_x0
right_margin = (crop_x0 + crop_w) - bx
min_margin = np.minimum(left_margin, right_margin)

print('=== WORST 30 FRAMES (ball nearest to/outside crop edge) ===')
worst = np.argsort(min_margin)[:30]
for fi in sorted(worst):
    delta = (cx[fi] - cx[fi-1]) if fi > 0 else 0
    vel = np.sqrt((bx[fi]-bx[fi-1])**2 + (by[fi]-by[fi-1])**2) if fi > 0 else 0
    side = 'LEFT' if (bx[fi] - crop_x0[fi]) < (crop_x0[fi] + crop_w[fi] - bx[fi]) else 'RIGHT'
    print(f'  f{fi:3d}: ball_x={bx[fi]:7.1f} cam_cx={cx[fi]:7.1f} offset={bx[fi]-cx[fi]:+7.1f} '
          f'margin={min_margin[fi]:+6.1f}({side}) cam_d={delta:+6.1f} src={src[fi]:10s} conf={conf[fi]:.2f}')

print()
print('=== FRAME-BY-FRAME: SHOT/SAVE ZONE (f350-f480) ===')
for fi in range(350, min(480, n)):
    delta = cx[fi] - cx[fi-1] if fi > 0 else 0
    marker = ''
    if min_margin[fi] < 80:
        side = 'LEFT' if (bx[fi] - crop_x0[fi]) < (crop_x0[fi] + crop_w[fi] - bx[fi]) else 'RIGHT'
        marker = f' *** NEAR {side} EDGE'
    if min_margin[fi] < 0:
        marker = f' !!! OUTSIDE CROP'
    print(f'  f{fi:3d}: ball_x={bx[fi]:7.1f} cam_cx={cx[fi]:7.1f} offset={bx[fi]-cx[fi]:+7.1f} '
          f'margin={min_margin[fi]:+6.1f} cam_d={delta:+6.1f} src={src[fi]:10s} conf={conf[fi]:.2f}{marker}')

print()
print('=== SOURCE DISTRIBUTION (f350-480) ===')
late = df.iloc[350:min(480, n)]
print(late['source'].value_counts().to_string())
print(f'\nMean confidence: {late["confidence"].mean():.2f}')
print(f'Frames with margin < 80px: {(min_margin[350:480] < 80).sum()}')
print(f'Frames outside crop: {(min_margin[350:480] < 0).sum()}')
print(f'\nCrop width (f350): {crop_w[350]:.0f}px')
print(f'Half crop: {crop_w[350]/2:.0f}px')
