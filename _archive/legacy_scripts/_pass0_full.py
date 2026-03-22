import sys
target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"
with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()
pass1_idx = None
for i, line in enumerate(lines):
    if "# --- Pass 1: Centroid drift clamp ---" in line:
        pass1_idx = i
        break
if pass1_idx is None:
    print("ERROR")
    sys.exit(1)
print(f"Pass1 at {pass1_idx+1}", flush=True)
for i, line in enumerate(lines):
    if "v22i: Post-FUSION" in line:
        lines[i] = "            # == v22i/v22j: Post-FUSION phantom cleanup ====================\n"
        break
for i, line in enumerate(lines):
    if "# Pass 1 - Centroid drift clamp" in line:
        lines[i] = "            # Pass 0 - Low-conf YOLO filter (conf<0.25 -> HOLD)\n"
        lines.insert(i+1, "            # Pass 1 - Centroid drift clamp (source=2 drifting from YOLO)\n")
        pass1_idx += 1
        break
pass0 = []
pass0.append("            # --- Pass 0: Low-conf YOLO -> HOLD ---\n")
pass0.append("            _prev_trusted_x = None\n")
pass0.append("            _phantom_yolo_frames = set()\n")
pass0.append("            for _fi in range(len(positions)):\n")
pass0.append("                _src = fusion_source_labels[_fi]\n")
pass0.append("                _conf = fusion_confidence[_fi]\n")
pass0.append("                if _src == 1 and _conf >= _PHANTOM_MIN_CONF:\n")
pass0.append("                    _prev_trusted_x = float(positions[_fi][0])\n")
pass0.append("                elif _src == 1 and _conf < _PHANTOM_MIN_CONF:\n")
pass0.append("                    _phantom_yolo_frames.add(_fi)\n")
pass0.append("                    if _prev_trusted_x is not None:\n")
pass0.append("                        positions[_fi][0] = _prev_trusted_x\n")
pass0.append("                    else:\n")
pass0.append("                        _af, _ax = _nearest_yolo_anchor(_fi)\n")
pass0.append("                        if _ax is not None:\n")
pass0.append("                            positions[_fi][0] = _ax\n")
pass0.append("                    fusion_source_labels[_fi] = 5\n")
pass0.append("                    fusion_confidence[_fi] = 0.3\n")
pass0.append("                    _phantom_fixes += 1\n")
pass0.append("                elif _src in (2, 3, 7) and _conf >= _PHANTOM_MIN_CONF:\n")
pass0.append("                    _prev_trusted_x = float(positions[_fi][0])\n")
pass0.append("\n")
pass0.append("            # Fix HOLD/shot_hold runs seeded by phantom YOLO\n")
pass0.append("            for _pf in sorted(_phantom_yolo_frames):\n")
pass0.append("                _hold_x = None\n")
pass0.append("                for _bi in range(_pf-1, max(-1,_pf-60), -1):\n")
pass0.append("                    if _bi >= 0 and fusion_source_labels[_bi] == 1 and fusion_confidence[_bi] >= _PHANTOM_MIN_CONF:\n")
pass0.append("                        _hold_x = float(positions[_bi][0])\n")
pass0.append("                        break\n")
pass0.append("                    elif _bi >= 0 and fusion_source_labels[_bi] in (2,3,7) and fusion_confidence[_bi] >= _PHANTOM_MIN_CONF:\n")
pass0.append("                        _hold_x = float(positions[_bi][0])\n")
pass0.append("                        break\n")
pass0.append("                if _hold_x is None:\n")
pass0.append("                    _af, _ax = _nearest_yolo_anchor(_pf)\n")
pass0.append("                    _hold_x = _ax if _ax is not None else None\n")
pass0.append("                if _hold_x is None:\n")
pass0.append("                    continue\n")
pass0.append("                for _fi2 in range(_pf+1, min(len(positions), _pf+120)):\n")
pass0.append("                    _s2 = fusion_source_labels[_fi2]\n")
pass0.append("                    if _s2 in (5, 6):\n")
pass0.append("                        _old_x = float(positions[_fi2][0])\n")
pass0.append("                        if abs(_old_x - _hold_x) > _PHANTOM_MAX_DRIFT:\n")
pass0.append("                            positions[_fi2][0] = _hold_x\n")
pass0.append("                            fusion_confidence[_fi2] = 0.3\n")
pass0.append("                            _phantom_fixes += 1\n")
pass0.append("                    elif _s2 == 4:\n")
pass0.append("                        _old_x = float(positions[_fi2][0])\n")
pass0.append("                        if abs(_old_x - _hold_x) > _PHANTOM_MAX_DRIFT:\n")
pass0.append("                            positions[_fi2][0] = _hold_x\n")
pass0.append("                            fusion_source_labels[_fi2] = 5\n")
pass0.append("                            fusion_confidence[_fi2] = 0.3\n")
pass0.append("                            _phantom_fixes += 1\n")
pass0.append("                    else:\n")
pass0.append("                        break\n")
pass0.append("\n")
new_lines = lines[:pass1_idx] + pass0 + lines[pass1_idx:]
with open(target, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
print(f"SUCCESS: {len(pass0)} lines inserted, total={len(new_lines)}", flush=True)
