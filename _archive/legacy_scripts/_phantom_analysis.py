"""Analyze phantom ball positions in v22h at the three problem zones."""
import csv, os

BASE = r'D:\Projects\soccer-video\out\portrait_reels\2026-02-23__TSC_vs_NEOFC'
rows = []
with open(os.path.join(BASE, '002__v22h_gaussian6.diag.csv')) as f:
    for r in csv.DictReader(f):
        rows.append(r)

# Source label mapping (handle both int and string)
SRC_MAP = {1:'YOLO', 2:'CENTROID', 3:'BLENDED', 4:'INTERP', 5:'HOLD', 6:'SHOT_HOLD', 7:'TRACKER'}
def src_name(val):
    try:
        return SRC_MAP.get(int(val), val)
    except ValueError:
        return str(val)

print("=" * 90)
print("ZONE 1: Defender receives ball (f20-f55) — phantom at midfield?")
print("=" * 90)
print(f"{'f':<5} {'ball_x':<9} {'cam_cx':<9} {'gap':<8} {'source':<12} {'conf':<6} {'in_crop':<8}")
print("-" * 60)
for i in range(20, min(56, len(rows))):
    r = rows[i]
    bx = float(r['ball_x']); cx = float(r['cam_cx'])
    src = r['source']; conf = float(r['confidence'])
    bic = r['ball_in_crop']
    gap = abs(bx - cx)
    sn = src_name(src)
    flag = " <<<" if sn in ('CENTROID','centroid','INTERP','interp','HOLD','hold','SHOT_HOLD','shot_hold') and gap > 100 else ""
    print(f"f{i:<4} {bx:>7.0f}  {cx:>7.0f}  {gap:>5.0f}px  {sn:<12} {conf:.2f}  {bic}{flag}")

print()
print("=" * 90)
print("ZONE 2: Wing receives ball (f160-f210) — phantom at midfield?")
print("=" * 90)
print(f"{'f':<5} {'ball_x':<9} {'cam_cx':<9} {'gap':<8} {'source':<12} {'conf':<6} {'in_crop':<8}")
print("-" * 60)
for i in range(160, min(211, len(rows))):
    r = rows[i]
    bx = float(r['ball_x']); cx = float(r['cam_cx'])
    src = r['source']; conf = float(r['confidence'])
    bic = r['ball_in_crop']
    gap = abs(bx - cx)
    sn = src_name(src)
    flag = " <<<" if gap > 150 else ""
    print(f"f{i:<4} {bx:>7.0f}  {cx:>7.0f}  {gap:>5.0f}px  {sn:<12} {conf:.2f}  {bic}{flag}")

print()
print("=" * 90)  
print("ZONE 3: Shot/cross to corner flag (f290-f345) — jerks to corner flag?")
print("=" * 90)
print(f"{'f':<5} {'ball_x':<9} {'cam_cx':<9} {'gap':<8} {'source':<12} {'conf':<6} {'in_crop':<8}")
print("-" * 60)
for i in range(290, min(346, len(rows))):
    r = rows[i]
    bx = float(r['ball_x']); cx = float(r['cam_cx'])
    src = r['source']; conf = float(r['confidence'])
    bic = r['ball_in_crop']
    gap = abs(bx - cx)
    sn = src_name(src)
    flag = " <<<" if gap > 150 else ""
    print(f"f{i:<4} {bx:>7.0f}  {cx:>7.0f}  {gap:>5.0f}px  {sn:<12} {conf:.2f}  {bic}{flag}")

print()
print("=" * 90)
print("ALL large ball_x jumps (>100px between frames)")
print("=" * 90)
for i in range(1, len(rows)):
    bx_prev = float(rows[i-1]['ball_x'])
    bx = float(rows[i]['ball_x'])
    jump = bx - bx_prev
    if abs(jump) > 100:
        sp = src_name(rows[i-1]['source'])
        sn = src_name(rows[i]['source'])
        print(f"  f{i-1}->f{i}: {bx_prev:.0f}->{bx:.0f} ({jump:+.0f}px) src: {sp}->{sn}")
