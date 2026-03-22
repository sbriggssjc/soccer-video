"""Revert v22l: remove the confidence floor filter from ball_telemetry.py."""
import sys

target = r"D:\Projects\soccer-video\tools\ball_telemetry.py"

with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the start marker
start_marker = "    # --- v22l: Confidence floor filter ---"
end_marker = "    # Index centroid samples by frame"

start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if start_marker in line and start_idx is None:
        start_idx = i
    if end_marker in line and start_idx is not None:
        end_idx = i
        break

if start_idx is None or end_idx is None:
    print(f"ERROR: Could not find markers (start={start_idx}, end={end_idx})")
    sys.exit(1)

# Remove lines from start_idx to end_idx (exclusive)
removed = end_idx - start_idx
new_lines = lines[:start_idx] + lines[end_idx:]

with open(target, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"SUCCESS: Removed {removed} lines ({start_idx+1} to {end_idx})")
print(f"Total lines: {len(new_lines)}")
