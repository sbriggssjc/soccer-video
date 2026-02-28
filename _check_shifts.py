import numpy as np
d = np.load(r'D:\Projects\soccer-video\out\telemetry\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.ball.cam_shifts.npy')
print(f'Shape: {d.shape}, dtype: {d.dtype}')
print(f'X shifts: min={d[:,0].min():.1f}, max={d[:,0].max():.1f}, mean={d[:,0].mean():.1f}')
print(f'Y shifts: min={d[:,1].min():.1f}, max={d[:,1].max():.1f}, mean={d[:,1].mean():.1f}')
print('Frames 10-30:')
for i in range(10, 31):
    print(f'  f{i}: dx={d[i,0]:+.1f}, dy={d[i,1]:+.1f}')
