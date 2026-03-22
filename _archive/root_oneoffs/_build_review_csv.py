"""Build review CSV for a NEOFC clip with pre-populated camera positions.
Samples at 2fps (every 15 frames) to match filmstrip.
User edits the 'camera_x_pct' column and returns the CSV.
"""
import os, csv, traceback

CLIP_NUM = "010"
GAME = "2026-02-23__TSC_vs_NEOFC"
RESULT = rf"D:\Projects\soccer-video\_tmp\build_review_csv_result.txt"
os.makedirs(r"D:\Projects\soccer-video\_tmp", exist_ok=True)

FRAME_W = 1920
FPS = 30
SAMPLE_INTERVAL = 15  # every 15 frames = 2fps

# Find diagnostic CSV
reels_dir = rf"D:\Projects\soccer-video\out\portrait_reels\{GAME}"
diag_file = None
for f in os.listdir(reels_dir):
    if f.startswith(f"{CLIP_NUM}__") and f.endswith("__portrait.diag.csv"):
        diag_file = os.path.join(reels_dir, f)
        break

try:
    if not diag_file:
        raise FileNotFoundError(f"No diag CSV for clip {CLIP_NUM}")

    # Read diagnostic
    with open(diag_file, "r") as f:
        reader = csv.DictReader(f)
        diag_rows = list(reader)

    total_frames = len(diag_rows)

    # Sample every SAMPLE_INTERVAL frames
    review_rows = []
    for i in range(0, total_frames, SAMPLE_INTERVAL):
        row = diag_rows[i]
        frame = int(row["frame"])
        ball_x = float(row["ball_x"])
        cam_cx = float(row["cam_cx"])
        time_s = round(frame / FPS, 1)

        ball_pct = round(ball_x / FRAME_W * 100, 0)
        cam_pct = round(cam_cx / FRAME_W * 100, 0)

        review_rows.append({
            "frame": frame,
            "time_s": time_s,
            "ball_x_pct": int(ball_pct),
            "camera_x_pct": int(cam_pct),
            "notes": ""
        })

    # Write review CSV
    output_csv = rf"D:\Projects\soccer-video\_tmp\review_{CLIP_NUM}.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "time_s", "ball_x_pct", "camera_x_pct", "notes"])
        writer.writeheader()
        writer.writerows(review_rows)

    # Copy to Desktop
    import shutil
    desktop_path = rf"C:\Users\scott\Desktop\review_{CLIP_NUM}.csv"
    shutil.copy2(output_csv, desktop_path)

    msg = f"SUCCESS\n"
    msg += f"Clip: {CLIP_NUM} ({total_frames} total frames)\n"
    msg += f"Sampled: {len(review_rows)} rows (every {SAMPLE_INTERVAL} frames)\n"
    msg += f"Output: {output_csv}\n"
    msg += f"Desktop: {desktop_path}\n"
    msg += f"\nColumns:\n"
    msg += f"  frame - source frame number\n"
    msg += f"  time_s - timestamp in seconds\n"
    msg += f"  ball_x_pct - current ball x position (0-100%)\n"
    msg += f"  camera_x_pct - current camera center (0-100%) <-- EDIT THIS\n"
    msg += f"  notes - optional notes\n"
    with open(RESULT, "w") as f:
        f.write(msg)

except Exception as e:
    with open(RESULT, "w") as f:
        f.write(f"EXCEPTION: {e}\n{traceback.format_exc()}")
