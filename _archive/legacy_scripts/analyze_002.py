#!/usr/bin/env python3
import csv, json
from collections import defaultdict

DIAG_CSV = "out/portrait_reels/2026-02-23__TSC_vs_NEOFC/002__v21f.diag.csv"
BALL_JSONL = "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.yolov8x.jsonl"
PERSON_JSONL = "out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_person.jsonl"
FPS = 30
CENTROID_GATE = 400

diag_rows = []
with open(DIAG_CSV, newline="") as f:
    for row in csv.DictReader(f):
        diag_rows.append(row)

ball_dets = []
with open(BALL_JSONL) as f:
    for line in f:
        ball_dets.append(json.loads(line))

person_map = {}
with open(PERSON_JSONL) as f:
    for line in f:
        obj = json.loads(line)
        person_map[obj["frame"]] = [p["cx"] for p in obj.get("persons", [])]

total_frames = len(diag_rows)

# === A ===
print("=" * 100)
print("A) CAMERA CX POSITION AT EVERY 10TH FRAME")
print("=" * 100)
print(f'{"frame":>6}  {"cam_cx":>8}  {"ball_x":>8}  {"source":>10}  {"ball_in_crop":>12}')
print("-" * 60)
for row in diag_rows:
    fr = int(row["frame"])
    if fr % 10 == 0:
        print(f'{fr:>6}  {float(row["cam_cx"]):>8.1f}  {float(row["ball_x"]):>8.1f}  {row["source"]:>10}  {int(row["ball_in_crop"]):>12}')

# === B ===
print()
print("=" * 100)
print("B) YOLO DETECTIONS (conf >= 0.30) WITH CENTROID DEVIATION ANALYSIS")
print("=" * 100)

diag_yolo_frames = set()
diag_source_map = {}
for row in diag_rows:
    fr = int(row["frame"])
    diag_source_map[fr] = row["source"]
    if row["source"] == "yolo":
        diag_yolo_frames.add(fr)

kept_dets = []
all_analyzed = []

print(f'{"frame":>6}  {"ball_x":>8}  {"conf":>6}  {"pers_cx":>9}  {"deviat":>8}  {"status":>14}  {"diag_src":>10}  notes')
print("-" * 110)

for det in ball_dets:
    fr = det["frame"]
    bx = det["cx"]
    conf = det["conf"]
    if conf < 0.30:
        continue
    persons = person_map.get(fr, [])
    person_cx = sum(persons) / len(persons) if persons else None
    deviation = abs(bx - person_cx) if person_cx is not None else None
    notes = []
    if deviation is not None and deviation > CENTROID_GATE:
        status = "CENTROID_REJ"
    else:
        status = "KEPT"
    in_diag = fr in diag_yolo_frames
    if status == "KEPT" and not in_diag:
        notes.append("rejected_by_other_filter")
        status = "OTHER_REJ"
    if in_diag and status == "KEPT":
        notes.append("in_diag_as_yolo")
    dev_str = f'{deviation:>8.1f}' if deviation is not None else "     N/A"
    pcx_str = f'{person_cx:>9.1f}' if person_cx is not None else "      N/A"
    dsrc = diag_source_map.get(fr, "N/A")
    nstr = "; ".join(notes)
    print(f'{fr:>6}  {bx:>8.1f}  {conf:>6.3f}  {pcx_str}  {dev_str}  {status:>14}  {dsrc:>10}  {nstr}')
    entry = dict(frame=fr, ball_x=bx, conf=conf, person_cx=person_cx, deviation=deviation, status=status, in_diag=in_diag)
    all_analyzed.append(entry)
    if status == "KEPT":
        kept_dets.append(entry)

print()
total_ge30 = len(all_analyzed)
n_kept = sum(1 for d in all_analyzed if d["status"] == "KEPT")
n_centroid_rej = sum(1 for d in all_analyzed if d["status"] == "CENTROID_REJ")
n_other_rej = sum(1 for d in all_analyzed if d["status"] == "OTHER_REJ")
print(f"  Total detections conf>=0.30: {total_ge30}")
print(f"  KEPT (centroid pass + in diag): {n_kept}")
print(f"  CENTROID_REJ (deviation > {CENTROID_GATE}): {n_centroid_rej}")
print(f"  OTHER_REJ (passed centroid but not in diag): {n_other_rej}")

# === C ===
print()
print("=" * 100)
print("C) GAP ANALYSIS: GAPS > 20 FRAMES BETWEEN CONSECUTIVE KEPT DETECTIONS")
print("=" * 100)

if len(kept_dets) < 2:
    print("  Fewer than 2 KEPT detections.")
else:
    kept_frames = sorted(d["frame"] for d in kept_dets)
    gaps = []
    if kept_frames[0] > 20:
        gaps.append((0, kept_frames[0], "START_TO_FIRST"))
    for i in range(len(kept_frames) - 1):
        gap = kept_frames[i + 1] - kept_frames[i]
        if gap > 20:
            gaps.append((kept_frames[i], kept_frames[i + 1], "BETWEEN"))
    if (total_frames - 1) - kept_frames[-1] > 20:
        gaps.append((kept_frames[-1], total_frames - 1, "LAST_TO_END"))
    if not gaps:
        print("  No gaps > 20 frames found.")
    else:
        print(f'  {"gap_start":>10}  {"gap_end":>10}  {"length":>8}  {"time_s":>7}  {"cam_cx_start":>13}  {"cam_cx_end":>11}  {"cx_delta":>9}  camera_behavior')
        print("  " + "-" * 100)
        for g_start, g_end, g_type in gaps:
            length = g_end - g_start
            time_s = length / FPS
            cx_start = float(diag_rows[g_start]["cam_cx"])
            cx_end = float(diag_rows[g_end]["cam_cx"])
            cx_delta = cx_end - cx_start
            cx_vals = [float(diag_rows[ff]["cam_cx"]) for ff in range(g_start, min(g_end + 1, total_frames))]
            cx_range = max(cx_vals) - min(cx_vals)
            sources_in_gap = [diag_rows[ff]["source"] for ff in range(g_start, min(g_end + 1, total_frames))]
            sc = defaultdict(int)
            for s in sources_in_gap:
                sc[s] += 1
            ss = ", ".join(f"{s}={c}" for s, c in sorted(sc.items(), key=lambda x: -x[1]))
            if cx_range < 20:
                beh = f"HOLDS (range={cx_range:.0f}px)"
            elif abs(cx_delta) > 50:
                d2 = "RIGHT" if cx_delta > 0 else "LEFT"
                beh = f"DRIFTS {d2} ({cx_delta:+.0f}px, range={cx_range:.0f}px)"
            else:
                beh = f"OSCILLATES (range={cx_range:.0f}px, net={cx_delta:+.0f}px)"
            print(f'  {g_start:>10}  {g_end:>10}  {length:>8}  {time_s:>6.1f}s  {cx_start:>13.1f}  {cx_end:>11.1f}  {cx_delta:>+9.1f}  {beh}')
            print(f'  {"":>10}  sources: {ss}')

# === D ===
print()
print("=" * 100)
print("D) TIME-ANNOTATED SUMMARY: 2-SECOND WINDOWS")
print("=" * 100)

window_size = 2 * FPS
n_windows = (total_frames + window_size - 1) // window_size
kept_frame_set = set(d["frame"] for d in kept_dets)
all_det_by_frame = defaultdict(list)
for d in all_analyzed:
    all_det_by_frame[d["frame"]].append(d)

print(f'{"window":>7}  {"time_range":>14}  {"frames":>12}  {"kept_dets":>10}  {"all_det30":>10}  {"cam_cx_min":>11}  {"cam_cx_max":>11}  {"cx_range":>9}  source_breakdown')
print("-" * 135)

for ww in range(n_windows):
    f_start = ww * window_size
    f_end = min((ww + 1) * window_size - 1, total_frames - 1)
    t_start = f_start / FPS
    t_end = (f_end + 1) / FPS
    n_kept_w = sum(1 for ff in range(f_start, f_end + 1) if ff in kept_frame_set)
    n_all30 = sum(len(all_det_by_frame.get(ff, [])) for ff in range(f_start, f_end + 1))
    cx_vals = [float(diag_rows[ff]["cam_cx"]) for ff in range(f_start, f_end + 1)]
    cx_min, cx_max = min(cx_vals), max(cx_vals)
    cx_range = cx_max - cx_min
    sc2 = defaultdict(int)
    for ff in range(f_start, f_end + 1):
        sc2[diag_rows[ff]["source"]] += 1
    ss2 = ", ".join(f"{s}={c}" for s, c in sorted(sc2.items(), key=lambda x: -x[1]))
    print(f'{ww:>7}  {t_start:>5.1f}s-{t_end:>5.1f}s  {f_start:>5}-{f_end:<5}  {n_kept_w:>10}  {n_all30:>10}  {cx_min:>11.1f}  {cx_max:>11.1f}  {cx_range:>9.1f}  {ss2}')

# === Summary ===
print()
print("=" * 100)
print("OVERALL SUMMARY")
print("=" * 100)
print(f"  Total frames: {total_frames} ({total_frames/FPS:.1f}s at {FPS}fps)")
print(f"  Raw YOLO ball detections: {len(ball_dets)}")
print(f"  Detections with conf >= 0.30: {total_ge30}")
print(f"  Detections KEPT (pass centroid + in diag): {n_kept}")
print(f"  Detections CENTROID_REJ: {n_centroid_rej}")
print(f"  Detections OTHER_REJ: {n_other_rej}")
src_all = defaultdict(int)
for row in diag_rows:
    src_all[row["source"]] += 1
print("  Diag source breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(src_all.items(), key=lambda x: -x[1])))
all_cx = [float(r["cam_cx"]) for r in diag_rows]
print(f"  Camera cx range: {min(all_cx):.1f} - {max(all_cx):.1f} (total sweep: {max(all_cx)-min(all_cx):.1f}px)")
bic_count = sum(1 for r in diag_rows if int(r["ball_in_crop"]) == 1)
print(f"  Ball-in-crop frames: {bic_count}/{total_frames} ({100*bic_count/total_frames:.1f}%)")
