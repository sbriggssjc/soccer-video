# Portrait Framing Audit — Clip 006 BUILD_UP_AND_GOAL

**Clip**: `006__2025-10-04__TSC_vs_FC_Tulsa_Black__BUILD_UP_AND_GOAL__t699.00-t720.00.mp4`
**Preset**: `auto` | **Source**: 1920×1080 @ 30fps | **Output**: 1080×1920 portrait
**Pipeline run**: `batch_pipeline.py --preset auto --force --clip <path>`

---

## Executive Summary

Two visible framing failures in the rendered portrait clip:

1. **Post-goal rightward pan to coach (≈11s mark)**: Camera continues panning right past the goal, ending up on the coach/sideline instead of staying on the celebration.
2. **Ball outside portrait frame in first ≈5 seconds**: The actual ball escapes the narrow 9:16 crop during rapid positional swings in the opening seconds.

Both issues share a common root: **YOLO detection gaps during critical moments create interpolated ball trajectories that diverge from reality**, and the downstream camera planner has no mechanism to detect or correct these phantom paths.

---

## Issue 1: Rightward Pan Past Goal (≈11s)

### What the telemetry shows

```
[FUSION] Long-flight: frames 354->429 (t=11.8s->14.3s, 75 frames),
         ball x=1486->1842 (581px) [linear]
```

A 75-frame (2.5s) YOLO dropout spans the exact moment the goal is scored. The fusion bridges this gap with **linear interpolation** from the last pre-goal YOLO (x=1486) to the first post-goal YOLO (x=1842), creating a steady rightward drift of 581px.

Ball x timeline through the goal event:
```
f300=1564, f330=1857, f354(last YOLO)=1486,
  ...75 frames interpolated rightward...
f429(next YOLO)=1842, f450=1886
```

Camera follows obediently:
```
f330=1580, f360=1382, f390=1490, f420=1577, f450=1491
```

### Root cause chain

#### 1. YOLO drops out during the goal — `ball_telemetry.py:1760-1765`

When the ball enters the net, YOLO loses detection for 75 frames (motion blur, occlusion by net mesh, players celebrating in front). This is the most critical 2.5 seconds of the clip and has zero direct observations.

#### 2. Linear interpolation creates a phantom rightward trajectory — `ball_telemetry.py:1897-1921`

The gap (75 frames) is under `LONG_INTERP_GAP` (90), so fusion interpolates. The distance (581px) exceeds `_LARGE_DIST_PX` (500), forcing `[linear]` mode instead of smoothstep. The interpolation assumes the ball traveled in a straight line from x=1486 to x=1842. In reality, the ball entered the net and stopped near x≈1486; the rightward shift to x=1842 is caused by the **broadcast camera panning right** to show the celebration, which shifts the ball's pixel position in the source frame.

```python
# ball_telemetry.py:1799-1805
_LARGE_DIST_PX = 500.0
if dist > _LARGE_DIST_PX:
    _interp_mode = "linear"  # ← forces linear for this 581px gap
```

#### 3. Goal-event detection fails — `render_follow_unified.py:5376-5417`

The scorer-memory system requires:
- `smooth_speed_pf > 6.0` for ≥5 consecutive frames (shot in flight)
- Followed by `_decel > 3.0` (rapid deceleration = ball hitting net)

Because the ball positions during frames 354–429 are **linearly interpolated**, the speed is approximately constant (~7.7 px/frame). There is no sharp deceleration event. The `goal_event` flag is **never set**.

```python
# render_follow_unified.py:5401-5404
if _high_speed_sustained >= 5 and _scorer_x is not None:
    _decel = _prev_smooth_speed - smooth_speed_pf
    if _decel > 3.0:  # ← never fires: interpolated speed is constant
        _goal_event_frame = frame_idx
```

Consequences of the missed goal event:
- **No goal-clip trim** (lines 8674-8699): The `[TRIM] Goal clip:` message is absent from the output. Only the blunt 21s max-duration trim fires, leaving ~10s of post-goal footage.
- **No celebration tracking**: The scorer snap-back (lines 5757-5785) never activates, so the camera doesn't lock onto the goal scorer.
- **No celebration zoom-out** (lines 5726-5741): The camera stays tight instead of widening to capture the celebration.

#### 4. Source-camera pan contaminates YOLO anchors

The telemetry reports pan on 396/809 frames (49%). The motion centroid compensates for pan in its **detection** (frame differencing), but both YOLO and centroid positions are in **local frame coordinates** — when the broadcast camera pans right after the goal, YOLO's next detection at frame 429 is at x=1842 not because the ball moved there, but because the source camera panned. The interpolation path from 1486→1842 is an artifact of the pan, not ball movement.

#### 5. Edge filter triple-pass confirms near-edge YOLO issues

```
[FUSION] Edge re-filter: 79/236 (33%) exceeded 20% threshold
[FUSION] Edge triple-filter: still 70/236 (30%); reduced to minimal 19px
```

33% of YOLO detections are near the frame edge. The triple-filter drops the margin to 19px to keep them. Some of these near-edge detections may be artifacts of the source camera's pan framing, contributing false anchor points for interpolation.

### Suggested fixes for Issue 1

**A. Detect goal events from the clip tag, not just ball deceleration.** The filename contains `GOAL`. When the tag is present, pre-scan the YOLO timeline for the last high-speed segment followed by a dropout gap — that gap IS the goal moment. Set `_goal_event_frame` to the start of the dropout.

**B. Treat long YOLO dropouts after high-speed segments as "ball stopped" events.** If the ball was moving >6 px/frame and then YOLO drops out for >30 frames, assume the ball has stopped (goal, save, out of play). Hold the camera at the last confident position instead of interpolating rightward.

**C. Cross-reference interpolation direction with source-camera pan direction.** If the interpolated ball movement aligns with the detected source-camera pan direction, the "movement" is likely a pan artifact, not real ball travel. Suppress or reduce the interpolation.

**D. Reduce `_POST_GOAL_TAIL_S` or make goal-trim work with inferred goals.** Currently 3.0s of tail after a detected goal event (line 8672). If the goal event can be inferred from the GOAL tag + YOLO dropout, the trim should still fire.

---

## Issue 2: Ball Outside Portrait Frame in First ≈5 Seconds

### What the telemetry shows

```
[FUSION] Backward-hold: filled 32 leading frames from first YOLO at frame 32
[FUSION] Long-flight: frames 45->66 (t=1.5s->2.2s), ball x=820->1839 (1088px) [linear]
[FUSION] Long-flight: frames 66->82 (t=2.2s->2.7s), ball x=1839->1573 (598px) [linear]
[FUSION] Long-flight: frames 82->112 (t=2.7s->3.7s), ball x=1573->917 (661px) [linear]
```

Ball x timeline:
```
f0=813, f30=813, f60=1548, f90=1398, f120=921
```

Camera cx timeline:
```
f0=825, f30=839, f60=1300, f90=1205, f120=993
```

Portrait crop width at zoom 1.0 ≈ 607px (half-width ≈ 303px). At zoom_min 0.85 ≈ 714px (half-width ≈ 357px).

### The math that shows the escape

At frame 60: ball x=1548, camera cx=1300.
- Distance from center: 248px
- At zoom=1.0: half-width=303px → ball is 55px inside right edge (tight but in)
- At zoom=1.35 (speed zoom tightens during fast motion): half-width=225px → **ball is 23px OUTSIDE**

The speed-zoom configuration at `render_presets.yaml:255-260` tightens the crop during fast ball motion:
```yaml
speed_zoom:
  v_lo: 2.0    # start tightening above 2 px/frame
  v_hi: 12.0   # maximum tightening at 12 px/frame
  zoom_lo: 1.35 # tight zoom at low speed
  zoom_hi: 0.92 # wide zoom at high speed
```

At the 1088px-in-21-frames interpolated speed (~52 px/frame), zoom_hi=0.92 applies (wide). But the **zoom slew** (line 5754) limits how fast zoom can change — the zoom may still be transitioning from tight (stationary hold at frames 0-32) to wide. During the transition, the crop is narrow enough for the ball to escape.

### Root cause chain

#### 1. Backward hold parks the camera at a stale position — `ball_telemetry.py:1925-1974`

Frames 0–31 are backward-held at the first YOLO position (x≈813). The camera settles at cx≈825. When the ball suddenly jumps to x=1839 (frame 45→66), the camera is 1014px behind.

```python
# ball_telemetry.py:1962-1963
positions[k, 0] = yf.x   # hold at first YOLO x
positions[k, 1] = yf.y
```

#### 2. Camera can't keep up with 1088px jump — `render_follow_unified.py:5181-5184, 5976`

Base speed limit: 1400 px/s → 46.7 px/frame at 30fps. Even with pan_detect_boost (2.5×) and keep-in-view boost (3×), the maximum is ~140 px/frame. The ball moves 52 px/frame over the interpolation, but the camera needs to close a 1014px gap that built up during the hold period. Closing that gap at 140 px/frame takes ~7 frames (0.23s) — during which the ball is outside the crop.

```python
# render_follow_unified.py:5181-5184
px_per_sec_x = self.speed_limit * 1.35      # 1890 px/s
pxpf_x = px_per_sec_x / render_fps          # 63 px/frame base
```

#### 3. Confidence-based speed damping slows the camera further — `render_follow_unified.py:5970-5973`

The backward-held frames have confidence 0.22. The interpolated frames have confidence 0.38.

```python
# render_follow_unified.py:5972
_conf_speed_scale = 0.70 + 0.50 * frame_conf  # 0.81 at conf=0.22, 0.89 at conf=0.38
```

This reduces effective speed to 81-89% during exactly the frames where the camera needs to move fastest.

#### 4. EMA smoothing adds further lag — `render_follow_unified.py:5844-5874`

The deadzone EMA (center_alpha=0.22 from preset) means the camera only moves 22% of the distance toward the ball per frame. On a 1014px gap: frame 1 moves 223px, frame 2 moves 174px, frame 3 moves 135px... it takes ~10 frames to close within 100px of the ball.

#### 5. Diagnostic is self-referential — `render_follow_unified.py:8750-8789`

The `[DIAG] Ball in crop: 630/630 (100.0%)` metric checks the **fused** ball position against the crop — not the real ball. The emergency keep-in-view corrections (lines 6006-6141) shift the crop to include the fused ball position. But the fused position may not be where the real ball is.

The centroid-dominance warning (line 8818) only fires when centroid frames exceed 50%. In this clip, the trimmed output has: `yolo=203, centroid=52, interp=334, hold=32`. Centroid is only 8%, but **interpolated frames are 53%** — and the warning doesn't cover interpolated frames. The diagnostic has a blind spot for interpolation-dominated clips.

```python
# render_follow_unified.py:8818 — only warns about centroid, not interpolation
if _checked > 0 and _centroid_only_total > _checked * 0.50:
    print(f"[DIAG] WARNING: ...")
```

### Suggested fixes for Issue 2

**A. Suppress speed-zoom tightening when ball position confidence is low.** During backward-hold (conf=0.22) and interpolated (conf=0.38) stretches, the zoom should default to wide (zoom_min) rather than tightening based on ball speed. The position is uncertain — keep the crop wide.

**B. Exempt keep-in-view corrections from the final speed clamp.** The final speed clamp at line 6162 can undo emergency crop shifts. Add a flag so that when emergency keep-in-view or margin corrections moved the crop, the final clamp preserves at least enough movement to keep the ball visible.

**C. Add interpolation-dominance warning to diagnostics.** Extend the centroid-dominance check at line 8818 to also warn when interpolated frames exceed 40% of total frames. These clips need manual review since the ball-in-crop metric is unreliable.

**D. Widen the backward-hold zone.** When the first YOLO detection is >30 frames in, the camera is parked at a stale position for 1+ seconds. Consider starting the camera at frame center or at a wider zoom for backward-held frames, then smoothly transitioning to the first real detection.

**E. Add a "catch-up" speed boost for gap-after-hold transitions.** When the camera has been holding for >15 frames and the ball suddenly jumps (first post-hold YOLO), temporarily lift the speed limit to allow the camera to close the gap within 3-5 frames.

---

## Systemic Issues Affecting Both Problems

### 1. YOLO coordinates are in local frame space — no pan stabilization

Both YOLO and centroid positions are in each frame's local pixel coordinates (`ball_telemetry.py:1436-1525`). When the broadcast camera pans, object positions shift in pixel space. The motion centroid compensates for pan in its **detection** step (frame differencing), but reports the centroid in local coordinates. This means:

- Interpolation between YOLO endpoints at different pan states creates phantom trajectories
- A stationary ball can appear to "move" 500+ pixels between YOLO detections simply because the source camera panned

**Fix**: Transform YOLO detections into stabilized/stitched coordinates before fusion. The motion telemetry already computes cumulative camera offsets for pan compensation — apply these to YOLO positions so interpolation between frames operates in a stable coordinate system.

### 2. The goal-event detector is speed-based, not context-based

The `_high_speed_sustained → _decel` mechanism (`render_follow_unified.py:5376-5417`) requires direct observation of the ball decelerating. When YOLO drops out at the moment of the goal (which is common — net occlusion, motion blur, celebration crowd), the detector fails silently. The clip name (`BUILD_UP_AND_GOAL`) contains the answer, but the camera planner ignores it.

**Fix**: Use clip metadata (action tags from the atomic index) as context for the camera planner. A `GOAL` tag should pre-configure:
- An expected goal frame range (from the clip's t-start/t-end and action metadata)
- A fallback goal-event detection using YOLO dropout patterns
- An automatic wide zoom in the post-goal section

### 3. The ball-in-crop diagnostic has a blind spot for interpolation

The 100% ball-in-crop metric is misleading. It verifies the fused ball position against the crop — but for 53% of frames in this clip, the fused position is interpolated (not observed). The camera follows the interpolation, so the interpolation is always "in crop" — a tautology. The real ball may be elsewhere.

**Fix**:
- Track a separate metric: `yolo_confirmed_in_crop` that only counts frames with actual YOLO detections
- Add an `interp_dominance_warning` when interpolated frames exceed 40%
- When interpolation dominance is high, report `ball_in_crop` as "unreliable"

---

## Pipeline Parameters for This Clip

| Parameter | Value | Source |
|-----------|-------|--------|
| `speed_limit` | 1400 px/s | `render_presets.yaml:238` |
| `smoothing` | 0.22 | `render_presets.yaml:236` |
| `post_smooth_sigma` | 5.0 | `render_presets.yaml:239` |
| `zoom_min` / `zoom_max` | 0.85 / 1.50 | `render_presets.yaml:240-241` |
| `keepinview_nudge` | 0.65 | `render_presets.yaml:252` |
| `keepinview_zoom_gain` | 0.40 | `render_presets.yaml:253` |
| `center_alpha` (EMA) | 0.22 | from `smoothing` |
| `EDGE_MARGIN_FRAC` | 0.01 (after triple-filter) | `ball_telemetry.py:1570` |
| `LONG_INTERP_GAP` | 90 frames | `ball_telemetry.py:1729` |
| `_SCORER_MAX_DRIFT` | 12% of width (230px) | `render_follow_unified.py:5313` |
| `_POST_GOAL_TAIL_S` | 3.0s | `render_follow_unified.py:8672` |

---

## Priority Ranking of Fixes

| # | Fix | Impact | Effort | Issues Addressed |
|---|-----|--------|--------|-----------------|
| 1 | Detect goal events from YOLO dropout after high-speed segments | High | Medium | Issue 1 |
| 2 | Stabilize YOLO coordinates to stitched frame before fusion | High | High | Issues 1 & 2 |
| 3 | Suppress speed-zoom tightening on low-confidence frames | Medium | Low | Issue 2 |
| 4 | Add catch-up speed boost after backward-hold gaps | Medium | Low | Issue 2 |
| 5 | Add interpolation-dominance warning to diagnostics | Medium | Low | Both |
| 6 | Use clip action tags as context for camera planner | High | Medium | Issue 1 |
| 7 | Exempt keep-in-view from final speed clamp | Medium | Low | Issue 2 |
