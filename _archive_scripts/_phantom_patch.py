"""Insert post-FUSION phantom cleanup into render_follow_unified.py."""
import sys

target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"

with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()

insert_idx = None
for i, line in enumerate(lines):
    if "# --- Wrong-ball sanity check ---" in line:
        insert_idx = i
        break

if insert_idx is None:
    print("ERROR: marker not found")
    sys.exit(1)

print(f"Found marker at line {insert_idx + 1}")

cleanup = r'''            # == v22i: Post-FUSION phantom cleanup ========================
            # Pass 1 - Centroid drift clamp (source=2 drifting from YOLO)
            # Pass 2 - Low-conf YOLO interp anchor filter (conf<0.25)
            # =============================================================

            _PHANTOM_MAX_DRIFT = 150.0
            _PHANTOM_MIN_CONF  = 0.25
            _phantom_fixes = 0

            _yolo_anchors = []
            for _fi in range(len(positions)):
                if (fusion_source_labels[_fi] == 1
                        and fusion_confidence[_fi] >= _PHANTOM_MIN_CONF
                        and used_mask[_fi]
                        and not np.isnan(positions[_fi][0])):
                    _yolo_anchors.append((_fi, float(positions[_fi][0])))

            def _nearest_yolo_anchor(frame_idx):
                if not _yolo_anchors:
                    return None, None
                best_dist = float("inf")
                best = None
                for af, ax in _yolo_anchors:
                    d = abs(af - frame_idx)
                    if d < best_dist:
                        best_dist = d
                        best = (af, ax)
                return best

'''
cleanup += r'''            # --- Pass 1: Centroid drift clamp ---
            for _fi in range(len(positions)):
                if fusion_source_labels[_fi] != 2:
                    continue
                if not used_mask[_fi] or np.isnan(positions[_fi][0]):
                    continue
                _anc_frame, _anc_x = _nearest_yolo_anchor(_fi)
                if _anc_x is None:
                    continue
                _cx = float(positions[_fi][0])
                if abs(_cx - _anc_x) > _PHANTOM_MAX_DRIFT:
                    positions[_fi][0] = _anc_x
                    fusion_source_labels[_fi] = 5
                    fusion_confidence[_fi] = 0.3
                    _phantom_fixes += 1

'''
cleanup += r'''            # --- Pass 2: Low-conf YOLO interp anchor filter ---
            _in_interp = False
            _interp_start = 0
            for _fi in range(len(positions) + 1):
                _is_interp = (_fi < len(positions)
                              and fusion_source_labels[_fi] == 4
                              and used_mask[_fi]
                              and not np.isnan(positions[_fi][0]))
                if _is_interp and not _in_interp:
                    _interp_start = _fi
                    _in_interp = True
                elif not _is_interp and _in_interp:
                    _interp_end = _fi - 1
                    _in_interp = False
                    _left_anchor_conf = 0.0
                    _left_anchor_x = None
                    _right_anchor_conf = 0.0
                    _right_anchor_x = None
                    for _bi in range(_interp_start - 1, max(-1, _interp_start - 10), -1):
                        if _bi >= 0 and fusion_source_labels[_bi] == 1:
                            _left_anchor_conf = float(fusion_confidence[_bi])
                            _left_anchor_x = float(positions[_bi][0])
                            break
                    for _bi in range(_interp_end + 1, min(len(positions), _interp_end + 10)):
                        if fusion_source_labels[_bi] == 1:
                            _right_anchor_conf = float(fusion_confidence[_bi])
                            _right_anchor_x = float(positions[_bi][0])
                            break
'''
cleanup += r'''                    _bad_left = _left_anchor_conf < _PHANTOM_MIN_CONF and _left_anchor_x is not None
                    _bad_right = _right_anchor_conf < _PHANTOM_MIN_CONF and _right_anchor_x is not None
                    if _bad_left or _bad_right:
                        if _bad_right and not _bad_left and _left_anchor_x is not None:
                            _hold_x = _left_anchor_x
                        elif _bad_left and not _bad_right and _right_anchor_x is not None:
                            _hold_x = _right_anchor_x
                        else:
                            _anc_f, _anc_x = _nearest_yolo_anchor((_interp_start + _interp_end) // 2)
                            _hold_x = _anc_x if _anc_x is not None else float(positions[_interp_start][0])
                        for _ri in range(_interp_start, _interp_end + 1):
                            positions[_ri][0] = _hold_x
                            fusion_source_labels[_ri] = 5
                            fusion_confidence[_ri] = 0.3
                            _phantom_fixes += 1

            if _phantom_fixes > 0:
                logger.info("[PHANTOM] Post-FUSION cleanup: fixed %d phantom frames", _phantom_fixes)

'''

cleanup_lines = [l + "\n" for l in cleanup.split("\n")]

new_lines = lines[:insert_idx] + cleanup_lines + lines[insert_idx:]

with open(target, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"SUCCESS: Inserted {len(cleanup_lines)} lines before line {insert_idx + 1}")
print(f"Total lines now: {len(new_lines)}")
