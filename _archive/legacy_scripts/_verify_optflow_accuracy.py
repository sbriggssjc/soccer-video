"""
Compare optical flow v3 trajectory against ORIGINAL YOLO and centroid data
to see where optical flow diverges from ground truth.
"""
import json, os

os.chdir(r'D:\Projects\soccer-video')

CLIP = '002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00'

# Load optical flow v3
optflow = {}
with open('_optflow_track/optflow_ball_path_v3.jsonl') as f:
    for line in f:
        d = json.loads(line)
        optflow[d['frame']] = d

# Load ORIGINAL YOLO (ground truth for ball position)
yolo = {}
with open(f'out/telemetry/{CLIP}.yolo_ball.jsonl.orig_backup') as f:
    for line in f:
        d = json.loads(line)
        if d.get('_meta'): continue
        if 'frame' in d:
            yolo[d['frame']] = d

# Load ORIGINAL centroid 
centroid = {}
with open(f'out/telemetry/{CLIP}.ball.jsonl.orig_backup') as f:
    for line in f:
        d = json.loads(line)
        if d.get('_meta'): continue
        if 'frame' in d:
            centroid[d['frame']] = d

print(f"Optical flow: {len(optflow)} frames")
print(f"Original YOLO: {len(yolo)} frames")  
print(f"Original centroid: {len(centroid)} frames")

# Compare at every YOLO frame
print(f"\n=== YOLO vs Optical Flow (at YOLO frames) ===")
big_errors = 0
for fi in sorted(yolo.keys()):
    y = yolo[fi]
    ycx, ycy = y['cx'], y['cy']
    yconf = y.get('conf', 0)
    
    if fi in optflow:
        o = optflow[fi]
        ocx, ocy = o['cx'], o['cy']
        dist = ((ycx - ocx)**2 + (ycy - ocy)**2)**0.5
        marker = '  *** BAD ***' if dist > 100 else ('  * drift *' if dist > 50 else '')
        if dist > 50:
            big_errors += 1
        print(f"  f{fi}: YOLO=({ycx:.0f},{ycy:.0f}) conf={yconf:.2f}  OptFlow=({ocx:.0f},{ocy:.0f})  dist={dist:.0f}px{marker}")

print(f"\n  Frames with >50px error: {big_errors}/{len(yolo)}")

# Check centroid vs optflow at regular intervals
print(f"\n=== Centroid vs Optical Flow (every 10 frames) ===")
for fi in range(0, 496, 10):
    if fi in centroid and fi in optflow:
        c = centroid[fi]
        o = optflow[fi]
        ccx, ccy = c['cx'], c['cy']
        ocx, ocy = o['cx'], o['cy']
        dist = ((ccx - ocx)**2 + (ccy - ocy)**2)**0.5
        ymark = f" YOLO=({yolo[fi]['cx']:.0f},{yolo[fi]['cy']:.0f})" if fi in yolo else ""
        marker = ' *** DIVERGED ***' if dist > 200 else (' * drift *' if dist > 100 else '')
        print(f"  f{fi}: Centroid=({ccx:.0f},{ccy:.0f}) OptFlow=({ocx:.0f},{ocy:.0f}) dist={dist:.0f}px{ymark}{marker}")

# Check where optical flow is tracking — show the x trajectory
print(f"\n=== OptFlow X trajectory (every 5 frames) ===")
for fi in range(0, 496, 5):
    if fi in optflow:
        o = optflow[fi]
        src = o.get('source', '?')
        yinfo = f" YOLO={yolo[fi]['cx']:.0f}" if fi in yolo else ""
        cinfo = f" Cent={centroid[fi]['cx']:.0f}" if fi in centroid else ""
        print(f"  f{fi}: optflow_x={o['cx']:.0f} optflow_y={o['cy']:.0f} src={src}{yinfo}{cinfo}")
