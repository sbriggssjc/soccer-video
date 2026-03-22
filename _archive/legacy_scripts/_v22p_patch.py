"""
v22p patch — three surgical changes:

1. ball_telemetry.py: Replace v22n spatial filter with simple conf floor
   (drop ALL YOLO with conf <= 0.25 — all 13 verified false positives)

2. ball_telemetry.py: Ramped shot-hold confidence (0.45 → 0.20) so early
   frames resist Gaussian pull but late frames allow smooth transition

3. render_follow_unified.py: Add SHOT_HOLD (source=6) to EMA protection
   with alpha=0.50 — moderate resistance to backward-pass erosion
   (~10 frame penetration vs ~50 at alpha=0.12)
"""
import re

# ── 1. ball_telemetry.py: Replace v22n spatial filter with simple conf floor ──
bt_path = "tools/ball_telemetry.py"
with open(bt_path, "r", encoding="utf-8") as f:
    bt = f.read()

# Find the v22n spatial filter block and replace it
old_filter = '''    # --- v22n: Spatial consistency filter for low-conf YOLO ---
    # Low-conf YOLO (conf <= 0.25) are often false positives (shadows,
    # cleats, watermarks). But some happen to be near the real ball.
    # Use spatial consistency with neighboring REAL YOLO detections:
    # - If the false YOLO is within SPATIAL_THRESH of the expected
    #   position (interpolated from real neighbors), KEEP it.
    # - If it's a spatial outlier, DROP it.
    _YOLO_CONF_FLOOR = 0.25
    _SPATIAL_THRESH = 200.0  # px: max deviation from expected trajectory
    _false_kept = 0
    _false_dropped = 0
    if yolo_by_frame:
        # Separate real vs false YOLO
        _real_frames = sorted(
            fi for fi, s in yolo_by_frame.items()
            if s.conf > _YOLO_CONF_FLOOR
        )
        _false_frames = sorted(
            fi for fi, s in yolo_by_frame.items()
            if s.conf <= _YOLO_CONF_FLOOR
        )
        _to_drop = []
        for fi in _false_frames:
            s = yolo_by_frame[fi]
            # Find nearest real YOLO before and after
            prev_real = None
            next_real = None
            for rf in _real_frames:
                if rf < fi:
                    prev_real = rf
                elif rf > fi and next_real is None:
                    next_real = rf
                    break
            # Compute expected position
            if prev_real is not None and next_real is not None:
                # Interpolate between neighbors
                p = yolo_by_frame[prev_real]
                n = yolo_by_frame[next_real]
                t = (fi - prev_real) / max(next_real - prev_real, 1)
                exp_x = p.x + t * (n.x - p.x)
            elif prev_real is not None:
                # Hold at prev
                exp_x = yolo_by_frame[prev_real].x
            elif next_real is not None:
                # Hold at next
                exp_x = yolo_by_frame[next_real].x
            else:
                # No real YOLO at all — drop false
                _to_drop.append(fi)
                continue
            deviation = abs(float(s.x) - float(exp_x))
            if deviation > _SPATIAL_THRESH:
                _to_drop.append(fi)
            else:
                _false_kept += 1
        for fi in _to_drop:
            del yolo_by_frame[fi]
            _false_dropped += 1
        if _false_kept > 0 or _false_dropped > 0:
            print(
                f"[FUSION] Spatial consistency filter: "
                f"kept {_false_kept} low-conf YOLO (near trajectory), "
                f"dropped {_false_dropped} outliers "
                f"(conf <= {_YOLO_CONF_FLOOR}, deviation > {_SPATIAL_THRESH:.0f}px)"
            )
            if _to_drop:
                for _dfi in sorted(_to_drop):
                    print(f"  [DROPPED] f{_dfi}")'''

new_filter = '''    # --- v22p: Drop ALL low-conf YOLO (verified false positives) ---
    # All 13 YOLO detections with conf <= 0.25 were visually verified as
    # false: shadows, player cleats, watermark text, sideline ball.
    # Drop them all to prevent false anchors in interpolation.
    _YOLO_CONF_FLOOR = 0.25
    _false_dropped = 0
    if yolo_by_frame:
        _to_drop = [
            fi for fi, s in yolo_by_frame.items()
            if s.conf <= _YOLO_CONF_FLOOR
        ]
        for fi in _to_drop:
            del yolo_by_frame[fi]
            _false_dropped += 1
        if _false_dropped > 0:
            print(
                f"[FUSION] v22p conf floor: dropped {_false_dropped} "
                f"false YOLO (conf <= {_YOLO_CONF_FLOOR})"
            )
            for _dfi in sorted(_to_drop):
                print(f"  [DROPPED] f{_dfi}")'''

assert old_filter in bt, "Could not find v22n spatial filter block"
bt = bt.replace(old_filter, new_filter)

# ── 2. ball_telemetry.py: Ramped shot-hold confidence ──
# Replace fixed _SH_CONF and the fill loop with ramped version
old_sh_fill = '''                    _SH_CONF = 0.25                       # low confidence'''
new_sh_fill = '''                    _SH_CONF_START = 0.45                 # v22p: high conf at start (resist Gaussian)
                    _SH_CONF_END = 0.20                   # v22p: low conf at end (smooth transition)'''

assert old_sh_fill in bt, "Could not find _SH_CONF line"
bt = bt.replace(old_sh_fill, new_sh_fill)

# Now replace the fill loop to use ramped confidence
old_fill_loop = '''                    for k in range(fi + 1, fj):
                        _sh_t = k - fi
                        if _sh_t <= _SH_EXTRAP:
                            _sh_cur_x += _sh_cur_vx
                            _sh_cur_y += _sh_cur_vy
                            _sh_cur_vx *= _SH_DECEL
                            _sh_cur_vy *= _SH_DECEL
                        else:
                            _sh_cur_x = _sh_hold_x
                            _sh_cur_y = _sh_hold_y
                        positions[k, 0] = max(0.0, min(width, _sh_cur_x))
                        positions[k, 1] = max(0.0, min(height, _sh_cur_y))
                        confidence[k] = _SH_CONF
                        source_labels[k] = FUSE_SHOT_HOLD'''

new_fill_loop = '''                    _sh_gap_len = fj - fi - 1  # total frames in this shot-hold
                    for k in range(fi + 1, fj):
                        _sh_t = k - fi
                        if _sh_t <= _SH_EXTRAP:
                            _sh_cur_x += _sh_cur_vx
                            _sh_cur_y += _sh_cur_vy
                            _sh_cur_vx *= _SH_DECEL
                            _sh_cur_vy *= _SH_DECEL
                        else:
                            _sh_cur_x = _sh_hold_x
                            _sh_cur_y = _sh_hold_y
                        positions[k, 0] = max(0.0, min(width, _sh_cur_x))
                        positions[k, 1] = max(0.0, min(height, _sh_cur_y))
                        # v22p: ramped confidence — high at start, low at end
                        _sh_frac = (_sh_t - 1) / max(_sh_gap_len - 1, 1)
                        confidence[k] = _SH_CONF_START + _sh_frac * (_SH_CONF_END - _SH_CONF_START)
                        source_labels[k] = FUSE_SHOT_HOLD'''

assert old_fill_loop in bt, "Could not find shot-hold fill loop"
bt = bt.replace(old_fill_loop, new_fill_loop)

with open(bt_path, "w", encoding="utf-8") as f:
    f.write(bt)
print(f"[OK] {bt_path}: v22n->v22p filter + ramped shot-hold conf")

# ── 3. render_follow_unified.py: Add SHOT_HOLD to EMA protection ──
ru_path = "tools/render_follow_unified.py"
with open(ru_path, "r", encoding="utf-8") as f:
    ru = f.read()

old_ema = '''            _FUSE_YOLO = np.uint8(1)
            _FUSE_INTERP = np.uint8(4)
            _FUSE_HOLD = np.uint8(5)
            _high_conf_alpha = 0.85  # trust own value — minimal neighbour pull
            for _si in range(_n_pos):
                _sl = fusion_source_labels[_si]
                if _sl == _FUSE_YOLO or _sl == _FUSE_INTERP:
                    _per_alpha[_si] = _high_conf_alpha
                elif _sl == _FUSE_HOLD:
                    # FUSE_HOLD frames are backward-filled copies of a
                    # YOLO anchor — they should NOT be pulled by the
                    # EMA's backward pass.  With alpha=0.85 the 15%
                    # leakage compounds across 10-50 hold frames and
                    # lets distant centroid positions drag the ball far
                    # from its true location, pushing the kick taker to
                    # the frame edge on free-kick clips.  Alpha=1.0
                    # locks them to the YOLO anchor position.
                    _per_alpha[_si] = 1.0'''

new_ema = '''            _FUSE_YOLO = np.uint8(1)
            _FUSE_INTERP = np.uint8(4)
            _FUSE_HOLD = np.uint8(5)
            _FUSE_SHOT_HOLD = np.uint8(6)  # v22p
            _high_conf_alpha = 0.85  # trust own value — minimal neighbour pull
            for _si in range(_n_pos):
                _sl = fusion_source_labels[_si]
                if _sl == _FUSE_YOLO or _sl == _FUSE_INTERP:
                    _per_alpha[_si] = _high_conf_alpha
                elif _sl == _FUSE_HOLD:
                    # FUSE_HOLD frames are backward-filled copies of a
                    # YOLO anchor — they should NOT be pulled by the
                    # EMA's backward pass.  With alpha=0.85 the 15%
                    # leakage compounds across 10-50 hold frames and
                    # lets distant centroid positions drag the ball far
                    # from its true location, pushing the kick taker to
                    # the frame edge on free-kick clips.  Alpha=1.0
                    # locks them to the YOLO anchor position.
                    _per_alpha[_si] = 1.0
                elif _sl == _FUSE_SHOT_HOLD:
                    # v22p: Moderate EMA protection for shot-hold.
                    # alpha=0.50 means backward-pass erosion penetrates
                    # ~10 frames (natural transition zone) instead of
                    # ~50 frames at default alpha=0.12.
                    _per_alpha[_si] = 0.50'''

assert old_ema in ru, "Could not find EMA protection block"
ru = ru.replace(old_ema, new_ema)

with open(ru_path, "w", encoding="utf-8") as f:
    f.write(ru)
print(f"[OK] {ru_path}: SHOT_HOLD EMA alpha=0.50")

print("\n=== v22p patch complete ===")
print("Changes:")
print("  1. ball_telemetry.py: v22n spatial filter → v22p conf floor (drop ALL 13 false)")
print("  2. ball_telemetry.py: _SH_CONF 0.25 → ramped 0.45→0.20")
print("  3. render_follow_unified.py: SHOT_HOLD EMA alpha=0.50")
