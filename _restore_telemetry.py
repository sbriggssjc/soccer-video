"""Restore original telemetry from backups"""
import shutil
from pathlib import Path

base = Path(r"D:\Projects\soccer-video\out\telemetry")
prefix = "002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00"

for suffix in ["ball.jsonl", "tracker_ball.jsonl", "yolo_ball.jsonl"]:
    backup = base / f"{prefix}.{suffix}.orig_backup"
    target = base / f"{prefix}.{suffix}"
    if backup.exists():
        shutil.copy2(backup, target)
        print(f"Restored {suffix} ({backup.stat().st_size} bytes)")
    else:
        print(f"WARNING: backup not found for {suffix}")

print("Done - original telemetry restored")
