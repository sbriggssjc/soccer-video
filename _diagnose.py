"""Diagnose batch render issues - check CSVs for sudden jumps."""
import os, csv, glob

tmp_dir = r"D:\Projects\soccer-video\_tmp"

for clip_num in ["006","007","008","009","010","011","012","013","014","015"]:
    csv_path = os.path.join(tmp_dir, f"review_{clip_num}.csv")
    if not os.path.exists(csv_path):
        print(f"{clip_num}: CSV not found in _tmp")
        continue
    
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    
    anchors = []
    empty_count = 0
    for r in rows:
        val = r["camera_x_pct"].strip()
        if val:
            anchors.append((int(r["frame"]), float(val)))
        else:
            empty_count += 1
    
    # Check for big jumps (>15% between consecutive anchors)
    big_jumps = []
    for i in range(1, len(anchors)):
        prev_frame, prev_pct = anchors[i-1]
        curr_frame, curr_pct = anchors[i]
        delta = abs(curr_pct - prev_pct)
        if delta > 15:
            big_jumps.append(f"  f{prev_frame}({prev_pct}%) -> f{curr_frame}({curr_pct}%) delta={delta}")
    
    pct_range = f"{min(a[1] for a in anchors)}-{max(a[1] for a in anchors)}%"
    status = "OK" if not big_jumps else f"JUMPS: {len(big_jumps)}"
    print(f"{clip_num}: {len(anchors)} anchors, {empty_count} empty, range {pct_range}, {status}")
    for j in big_jumps:
        print(j)
