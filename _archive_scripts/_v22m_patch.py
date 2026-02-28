"""v22m: Replace low-conf false YOLO with centroid positions.

Instead of dropping false YOLO detections (which leaves gaps), replace
their (x,y) with the centroid position at that frame. This maintains
anchor density while pointing the camera at the player cluster (near
the action) instead of shadows, cleats, or watermarks.

Inserts right before "# Index centroid samples by frame" in
ball_telemetry.py's fuse_yolo_and_centroid().
"""
import sys

target = r"D:\Projects\soccer-video\tools\ball_telemetry.py"

with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()

marker = "    # Index centroid samples by frame"
insert_idx = None
for i, line in enumerate(lines):
    if marker in line:
        insert_idx = i
        break

if insert_idx is None:
    print(f"ERROR: Could not find marker")
    sys.exit(1)

print(f"Found marker at line {insert_idx + 1}")

filter_code = [
    "    # --- v22m: Replace low-conf false YOLO with centroid positions ---\n",
    "    # False YOLO detections (shadows, cleats, watermarks) have conf <= 0.25.\n",
    "    # Instead of dropping them (which leaves interpolation gaps), replace\n",
    "    # their position with the centroid at that frame. The centroid tracks\n",
    "    # the player cluster, which is near the real action even if not exactly\n",
    "    # on the ball. This preserves anchor density while correcting positions.\n",
    "    # _centroid_lookup was built above for the multi-ball filter.\n",
    "    _YOLO_CONF_FLOOR = 0.25\n",
    "    _replaced_count = 0\n",
    "    _dropped_count = 0\n",
    "    if yolo_by_frame:\n",
    "        _to_replace = []\n",
    "        _to_drop = []\n",
    "        for fi, s in yolo_by_frame.items():\n",
    "            if s.conf <= _YOLO_CONF_FLOOR:\n",
    "                if fi in _centroid_lookup:\n",
    "                    _to_replace.append(fi)\n",
    "                else:\n",
    "                    _to_drop.append(fi)\n",
    "        for fi in _to_replace:\n",
    "            cx, cy = _centroid_lookup[fi]\n",
    "            old_s = yolo_by_frame[fi]\n",
    "            # Create a replacement sample with centroid position but\n",
    "            # reduced confidence (0.26) so FUSION treats it as marginal\n",
    "            yolo_by_frame[fi] = BallSample(\n",
    "                frame=old_s.frame, x=cx, y=cy,\n",
    "                conf=0.26, w=0.0, h=0.0,\n",
    "            )\n",
    "            _replaced_count += 1\n",
    "        for fi in _to_drop:\n",
    "            del yolo_by_frame[fi]\n",
    "            _dropped_count += 1\n",
    "        if _replaced_count > 0 or _dropped_count > 0:\n",
    "            print(\n",
    "                f\"[FUSION] Low-conf YOLO filter: replaced {_replaced_count} \"\n",
    "                f\"with centroid, dropped {_dropped_count} \"\n",
    "                f\"(conf <= {_YOLO_CONF_FLOOR:.2f})\"\n",
    "            )\n",
    "\n",
]

new_lines = lines[:insert_idx] + filter_code + lines[insert_idx:]

with open(target, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"SUCCESS: Inserted {len(filter_code)} lines at line {insert_idx + 1}")
print(f"Total lines: {len(new_lines)}")
