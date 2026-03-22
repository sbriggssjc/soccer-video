"""v22l: Add confidence floor filter to fuse_yolo_and_centroid.

Drops YOLO detections with conf < 0.25 from yolo_by_frame so they
never become interpolation anchors, hold endpoints, or direct positions.
This eliminates phantom ball positions (conf 0.20-0.21 at wrong locations)
that pull the camera away from the action.

Inserts the filter right after the multi-ball spatial consistency filter
and before the centroid indexing.
"""
import sys

target = r"D:\Projects\soccer-video\tools\ball_telemetry.py"

with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the marker: "# Index centroid samples by frame"
marker = "# Index centroid samples by frame"
insert_idx = None
for i, line in enumerate(lines):
    if marker in line:
        insert_idx = i
        break

if insert_idx is None:
    print(f"ERROR: Could not find '{marker}'")
    sys.exit(1)

print(f"Found marker at line {insert_idx + 1}")

# The confidence floor filter to insert BEFORE the centroid indexing
filter_code = [
    "    # --- v22l: Confidence floor filter ---\n",
    "    # Drop YOLO detections below a minimum confidence threshold.\n",
    "    # Phantom ball detections (conf 0.20-0.21) at wrong positions\n",
    "    # pull the camera away from the action when they become interp\n",
    "    # anchors, hold endpoints, or direct FUSE_YOLO positions.\n",
    "    # By filtering here (before the fusion loop and interpolation),\n",
    "    # phantoms never enter the pipeline at all.\n",
    "    _CONF_FLOOR = 0.25\n",
    "    _conf_filtered = 0\n",
    "    if yolo_by_frame:\n",
    "        _pre_count = len(yolo_by_frame)\n",
    "        yolo_by_frame = {\n",
    "            fi: s for fi, s in yolo_by_frame.items()\n",
    "            if s.conf >= _CONF_FLOOR\n",
    "        }\n",
    "        _conf_filtered = _pre_count - len(yolo_by_frame)\n",
    "        if _conf_filtered > 0:\n",
    "            print(\n",
    "                f\"[FUSION] Confidence floor filter: dropped {_conf_filtered}/{_pre_count} \"\n",
    "                f\"YOLO detections with conf < {_CONF_FLOOR:.2f}\"\n",
    "            )\n",
    "\n",
]

new_lines = lines[:insert_idx] + filter_code + lines[insert_idx:]

with open(target, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"SUCCESS: Inserted {len(filter_code)} lines at line {insert_idx + 1}")
print(f"Total lines: {len(new_lines)}")
