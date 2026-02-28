"""
v22r: Minimal change from v22h baseline.

REVERT all v22n/v22p/v22q changes, then apply ONLY:
1. Drop f332 from yolo_by_frame (false shadow in shot zone — enables shot-hold)
2. Zero-velocity shot-hold (freeze at last YOLO x=1607 instead of wrong-dir extrap)
3. Moderate shot-hold confidence (0.40 fixed) for Gaussian weight resistance

Keep ALL other 12 false YOLO as interpolation anchors.
Revert EMA shot-hold protection (let Gaussian handle the transition naturally).
"""
import re

# ── 1. ball_telemetry.py ──
bt_path = "tools/ball_telemetry.py"
with open(bt_path, "r", encoding="utf-8") as f:
    bt = f.read()

# REVERT: Replace v22p conf floor with targeted f332-only drop
old_filter = '''    # --- v22p: Drop ALL low-conf YOLO (verified false positives) ---
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

new_filter = '''    # --- v22r: Drop ONLY f332 (false shadow in shot zone) ---
    # f332 is a verified false YOLO (player shadow at x=919, real ball
    # at x=1607+).  Dropping it enables shot-hold activation for the
    # f314->f425 gap.  Keep the other 12 low-conf YOLO as interpolation
    # anchors — they're near the real trajectory and guide the camera.
    _V22R_DROP = {332}  # frames to remove from yolo_by_frame
    _v22r_dropped = []
    for _dfi in sorted(_V22R_DROP):
        if _dfi in yolo_by_frame:
            del yolo_by_frame[_dfi]
            _v22r_dropped.append(_dfi)
    if _v22r_dropped:
        print(f"[FUSION] v22r: dropped {len(_v22r_dropped)} targeted false YOLO: {_v22r_dropped}")'''

assert old_filter in bt, "Could not find v22p conf floor block"
bt = bt.replace(old_filter, new_filter)

# KEEP: zero-velocity shot-hold (v22q) — already in the code
assert "_sh_vx = 0.0" in bt, "Zero-velocity fix (v22q) should already be present"
assert "_sh_vy = 0.0" in bt, "Zero-velocity fix (v22q) should already be present"

# Change ramped confidence to fixed moderate value
old_conf = '''                    _SH_CONF_START = 0.45                 # v22p: high conf at start (resist Gaussian)
                    _SH_CONF_END = 0.20                   # v22p: low conf at end (smooth transition)'''
new_conf = '''                    _SH_CONF = 0.40                       # v22r: moderate conf (resist Gaussian erosion)'''

assert old_conf in bt, "Could not find v22p ramped conf"
bt = bt.replace(old_conf, new_conf)

# Fix the fill loop to use fixed conf instead of ramped
old_ramp = '''                        # v22p: ramped confidence — high at start, low at end
                        _sh_frac = (_sh_t - 1) / max(_sh_gap_len - 1, 1)
                        confidence[k] = _SH_CONF_START + _sh_frac * (_SH_CONF_END - _SH_CONF_START)'''
new_ramp = '''                        confidence[k] = _SH_CONF'''

assert old_ramp in bt, "Could not find v22p ramped fill"
bt = bt.replace(old_ramp, new_ramp)

# Remove the _sh_gap_len line (no longer needed)
bt = bt.replace(
    "                    _sh_gap_len = fj - fi - 1  # total frames in this shot-hold\n",
    ""
)

with open(bt_path, "w", encoding="utf-8") as f:
    f.write(bt)
print(f"[OK] {bt_path}: v22r — drop only f332, zero-vel, fixed conf=0.40")

# ── 2. render_follow_unified.py: REVERT shot-hold EMA protection ──
ru_path = "tools/render_follow_unified.py"
with open(ru_path, "r", encoding="utf-8") as f:
    ru = f.read()

# Remove the SHOT_HOLD EMA protection added in v22p
old_ema = '''            _FUSE_YOLO = np.uint8(1)
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

new_ema = '''            _FUSE_YOLO = np.uint8(1)
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

assert old_ema in ru, "Could not find v22p EMA with SHOT_HOLD"
ru = ru.replace(old_ema, new_ema)

with open(ru_path, "w", encoding="utf-8") as f:
    f.write(ru)
print(f"[OK] {ru_path}: reverted SHOT_HOLD EMA protection")

print("\n=== v22r patch complete ===")
print("Changes from v22h:")
print("  1. Drop ONLY f332 (enables shot-hold in shot zone)")
print("  2. Zero-velocity shot-hold (freeze at x=1607)")
print("  3. Shot-hold conf=0.40 (moderate Gaussian resistance)")
print("Reverted:")
print("  - v22p conf floor (was dropping ALL 13 false YOLO)")
print("  - v22p EMA shot-hold protection")
print("  - v22p ramped confidence")
