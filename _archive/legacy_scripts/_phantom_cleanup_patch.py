"""Insert post-FUSION phantom cleanup into render_follow_unified.py"""
import re

target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"

with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# Find the insertion point: after the logger.info block, before Wrong-ball sanity check
marker = '            )\n\n            # --- Wrong-ball sanity check ---'

if marker not in content:
    print("ERROR: Could not find insertion marker!")
    exit(1)

patch = '''            )

            # ── v22i: Post-FUSION phantom cleanup ─────────────────────
            # Two passes to eliminate phantom ball positions that pull
            # the camera away from the real action:
            #
            # Pass 1 — Centroid drift clamp
            #   CENTROID (source=2) tracks scene motion (player clusters),
            #   not the ball.  When centroid drifts >MAX_DRIFT px from the
            #   nearest YOLO anchor, replace with HOLD at that anchor.
            #
            # Pass 2 — Low-confidence YOLO interp anchor filter
            #   INTERP (source=4) linearly interpolates between YOLO
            #   endpoints.  If an endpoint has conf < MIN_CONF, the
            #   whole interp segment may target a phantom.  Replace
            #   with HOLD from the other (trusted) anchor.
            # ───────────────────────────────────────────────────────────

            _PHANTOM_MAX_DRIFT = 150.0   # px — max centroid drift from nearest YOLO
            _PHANTOM_MIN_CONF  = 0.25    # min YOLO conf to trust as interp anchor
            _phantom_fixes = 0

            # Build YOLO anchor index: frames where source==1 (YOLO) and conf >= MIN_CONF
            _yolo_anchors = []  # list of (frame_idx, x_position)
            for _fi in range(len(positions)):
                if (fusion_source_labels[_fi] == 1
                        and fusion_confidence[_fi] >= _PHANTOM_MIN_CONF
                        and used_mask[_fi]
                        and not np.isnan(positions[_fi][0])):
                    _yolo_anchors.append((_fi, float(positions[_fi][0])))
'''

content = content.replace(marker, patch + '''
            def _nearest_yolo_anchor(frame_idx: int) -> tuple:
                """Return (anchor_frame, anchor_x) of closest trusted YOLO frame."""
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

            # --- Pass 1: Centroid drift clamp ---
            for _fi in range(len(positions)):
                if fusion_source_labels[_fi] != 2:  # only CENTROID
                    continue
                if not used_mask[_fi] or np.isnan(positions[_fi][0]):
                    continue
                _anc_frame, _anc_x = _nearest_yolo_anchor(_fi)
                if _anc_x is None:
                    continue
                _cx = float(positions[_fi][0])
                if abs(_cx - _anc_x) > _PHANTOM_MAX_DRIFT:
                    positions[_fi][0] = _anc_x
                    fusion_source_labels[_fi] = 5  # HOLD
                    fusion_confidence[_fi] = 0.3
                    _phantom_fixes += 1
''', 1)
