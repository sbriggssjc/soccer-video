"""v22n: Replace centroid-swap with spatial consistency filter.

For each low-conf YOLO (conf <= 0.25):
  1. Find nearest REAL (conf > 0.25) YOLO before and after
  2. Compute expected ball position by linear interpolation
  3. If false YOLO is within threshold of expected: KEEP (it's near the ball)
  4. If false YOLO is far from expected: DROP (it's a spatial outlier)

This keeps useful low-conf detections (e.g. player cleat near the ball)
while removing harmful ones (shadows in midfield, watermarks).
"""
import sys

target = r"D:\Projects\soccer-video\tools\ball_telemetry.py"

with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find and remove the v22m block
start_marker = "    # --- v22m: Replace low-conf false YOLO with centroid positions ---"
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
    print(f"ERROR: v22m markers not found (start={start_idx}, end={end_idx})")
    sys.exit(1)

removed = end_idx - start_idx
print(f"Removing v22m block: lines {start_idx+1}-{end_idx} ({removed} lines)")
lines = lines[:start_idx] + lines[end_idx:]

# Now find the insertion point (same marker, now at the new position)
insert_marker = "    # Index centroid samples by frame"
insert_idx = None
for i, line in enumerate(lines):
    if insert_marker in line:
        insert_idx = i
        break

if insert_idx is None:
    print("ERROR: insertion marker not found")
    sys.exit(1)

print(f"Inserting v22n block at line {insert_idx + 1}")

new_block = [
    "    # --- v22n: Spatial consistency filter for low-conf YOLO ---\n",
    "    # Low-conf YOLO (conf <= 0.25) are often false positives (shadows,\n",
    "    # cleats, watermarks). But some happen to be near the real ball.\n",
    "    # Use spatial consistency with neighboring REAL YOLO detections:\n",
    "    # - If the false YOLO is within SPATIAL_THRESH of the expected\n",
    "    #   position (interpolated from real neighbors), KEEP it.\n",
    "    # - If it's a spatial outlier, DROP it.\n",
    "    _YOLO_CONF_FLOOR = 0.25\n",
    "    _SPATIAL_THRESH = 200.0  # px: max deviation from expected trajectory\n",
    "    _false_kept = 0\n",
    "    _false_dropped = 0\n",
    "    if yolo_by_frame:\n",
    "        # Separate real vs false YOLO\n",
    "        _real_frames = sorted(\n",
    "            fi for fi, s in yolo_by_frame.items()\n",
    "            if s.conf > _YOLO_CONF_FLOOR\n",
    "        )\n",
    "        _false_frames = sorted(\n",
    "            fi for fi, s in yolo_by_frame.items()\n",
    "            if s.conf <= _YOLO_CONF_FLOOR\n",
    "        )\n",
]
new_block += [
    "        _to_drop = []\n",
    "        for fi in _false_frames:\n",
    "            s = yolo_by_frame[fi]\n",
    "            # Find nearest real YOLO before and after\n",
    "            prev_real = None\n",
    "            next_real = None\n",
    "            for rf in _real_frames:\n",
    "                if rf < fi:\n",
    "                    prev_real = rf\n",
    "                elif rf > fi and next_real is None:\n",
    "                    next_real = rf\n",
    "                    break\n",
    "            # Compute expected position\n",
    "            if prev_real is not None and next_real is not None:\n",
    "                # Interpolate between neighbors\n",
    "                p = yolo_by_frame[prev_real]\n",
    "                n = yolo_by_frame[next_real]\n",
    "                t = (fi - prev_real) / max(next_real - prev_real, 1)\n",
    "                exp_x = p.x + t * (n.x - p.x)\n",
    "            elif prev_real is not None:\n",
    "                # Hold at prev\n",
    "                exp_x = yolo_by_frame[prev_real].x\n",
    "            elif next_real is not None:\n",
    "                # Hold at next\n",
    "                exp_x = yolo_by_frame[next_real].x\n",
    "            else:\n",
    "                # No real YOLO at all — drop false\n",
    "                _to_drop.append(fi)\n",
    "                continue\n",
    "            deviation = abs(float(s.x) - float(exp_x))\n",
    "            if deviation > _SPATIAL_THRESH:\n",
    "                _to_drop.append(fi)\n",
    "            else:\n",
    "                _false_kept += 1\n",
]
new_block += [
    "        for fi in _to_drop:\n",
    "            del yolo_by_frame[fi]\n",
    "            _false_dropped += 1\n",
    "        if _false_kept > 0 or _false_dropped > 0:\n",
    "            print(\n",
    "                f\"[FUSION] Spatial consistency filter: \"\n",
    "                f\"kept {_false_kept} low-conf YOLO (near trajectory), \"\n",
    "                f\"dropped {_false_dropped} outliers \"\n",
    "                f\"(conf <= {_YOLO_CONF_FLOOR}, deviation > {_SPATIAL_THRESH:.0f}px)\"\n",
    "            )\n",
    "            if _to_drop:\n",
    "                for _dfi in sorted(_to_drop):\n",
    "                    print(f\"  [DROPPED] f{_dfi}\")\n",
    "\n",
]

new_lines = lines[:insert_idx] + new_block + lines[insert_idx:]

with open(target, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"SUCCESS: Removed {removed} v22m lines, inserted {len(new_block)} v22n lines")
print(f"Total lines: {len(new_lines)}")
