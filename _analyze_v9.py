import csv
from collections import Counter

with open(r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__pipeline_v9.diag.csv') as f:
    rows = list(csv.DictReader(f))

sources = Counter(r['source'] for r in rows)
bic = sum(1 for r in rows if r['ball_in_crop'] == '1')
total = len(rows)
cxs = [float(r['cam_cx']) for r in rows]

print(f'Total: {total}, BIC: {bic}/{total} ({100*bic/total:.1f}%)')
for s, c in sources.most_common():
    out_count = sum(1 for r in rows if r['source'] == s and r['ball_in_crop'] == '0')
    print(f'  {s}: {c} ({100*c/total:.1f}%) - {out_count} out of crop')
print(f'CX range: {min(cxs):.0f}-{max(cxs):.0f} ({max(cxs)-min(cxs):.0f}px)')
escapes = [(int(r['frame']), float(r['dist_to_edge']), r['source']) for r in rows if r['ball_in_crop'] == '0']
print(f'Escapes ({len(escapes)}): {escapes}')
sl = sum(1 for r in rows if r.get('speed_limited') == '1')
print(f'Speed limited: {sl}/{total}')

# Compare with original pipeline render
import os
orig_diag = r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC\002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00__portrait.diag.csv'
if os.path.exists(orig_diag):
    with open(orig_diag) as f2:
        orig_rows = list(csv.DictReader(f2))
    orig_sources = Counter(r['source'] for r in orig_rows)
    orig_bic = sum(1 for r in orig_rows if r['ball_in_crop'] == '1')
    orig_cxs = [float(r['cam_cx']) for r in orig_rows]
    print(f'\n--- ORIGINAL (2/25) ---')
    print(f'BIC: {orig_bic}/{len(orig_rows)} ({100*orig_bic/len(orig_rows):.1f}%)')
    for s, c in orig_sources.most_common():
        print(f'  {s}: {c} ({100*c/len(orig_rows):.1f}%)')
    print(f'CX range: {min(orig_cxs):.0f}-{max(orig_cxs):.0f} ({max(orig_cxs)-min(orig_cxs):.0f}px)')
