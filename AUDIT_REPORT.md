# Repository Audit Report

**Date:** 2026-02-10
**Repository:** soccer-video
**Scope:** Full codebase audit for correctness, completeness, and production-readiness

---

## Project Overview

This repository is an end-to-end toolkit for converting curated soccer match moments into polished 1080x1920 portrait reels for social media. The pipeline transforms hand-selected "atomic" 16:9 clips through AI upscaling (Real-ESRGAN), motion-aware cropping, camera follow, branding overlays, and watermarks.

**Tech stack:** Python 3.9+, FFmpeg, OpenCV, Real-ESRGAN, PowerShell, Pydantic, Loguru
**Primary OS target:** Windows 10/11

---

## BLOCKER-Level Issues (Must Fix)

These issues cause crashes (`NameError`, `TypeError`) on affected code paths.

### B1. `render_follow_unified.py:3079-3091` — Ball position lookup always returns `None`

`_pair_from_mapping()` and `_pair_from_sequence()` both lack return statements, so they implicitly return `None`. This means `_get_ball_xy_src()` always returns `(None, None)`, completely breaking ball-position-based camera follow from telemetry data.

```python
# Line 3079 - missing return
def _pair_from_mapping(mapping, key_x, key_y):
    if key_x not in mapping or key_y not in mapping:
        return None
    val_x = mapping.get(key_x)
    val_y = mapping.get(key_y)
    # BUG: no return statement

# Line 3089 - missing return
def _pair_from_sequence(seq):
    if len(seq) < 2:
        return None
    # BUG: no return statement
```

**Fix:** Add `return (val_x, val_y)` and `return (seq[0], seq[1])` respectively.

### B2. `render_follow_unified.py:1476` — `_motion_centroid` references undefined variable `flow`

```python
mag = cv2.magnitude(flow[..., 0], flow[..., 1])  # 'flow' is never defined
```

The function receives `prev_gray` and `cur_gray` but never calls `cv2.calcOpticalFlowFarneback()`. Will raise `NameError` on every invocation.

### B3. `render_follow_unified.py:3670-3730` — `plan_crop_from_ball` references multiple undefined variables

- Line 3670: `zoom_value` used before assignment
- Lines 3723-3730: `prev_w`, `prev_h`, `prev_x`, `prev_y` never defined

### B4. `render_follow_unified.py:4099-4125` — `portrait_config_from_preset._parse_size` references undefined variables

- Lines 4099-4100: `w` and `h` used instead of `width` and `height`
- Lines 4106-4107: Same issue
- Line 4125: `mbw_f` and `mbh_f` never assigned (should be derived from `mbw` and `mbh`)

### B5. `io.py:79-80` — `int(None)` crash on missing video width/height

```python
width = int(stream.get("width"))     # stream.get("width") -> None if missing
height = int(stream.get("height"))   # int(None) raises TypeError
```

Any video lacking standard metadata fields will crash here.

### B6. `io.py:82` — `ValueError` on fraction-format `avg_frame_rate`

```python
fps = float(Fraction(r_frame_rate)) if "0/0" not in r_frame_rate else float(stream.get("avg_frame_rate", 0.0))
```

When the fallback `avg_frame_rate` is a string like `"30/1"`, `float("30/1")` raises `ValueError`. Must parse through `Fraction()` first.

---

## HIGH-Severity Issues

### H1. `upscale.py:4-5` — Hardcoded absolute Windows paths

```python
REALESRGAN_EXE = r"C:\Users\scott\soccer-video\tools\realesrgan\realesrgan-ncnn-vulkan.exe"
UPSCALE_OUT_ROOT = Path(r"C:\Users\scott\soccer-video\out\upscaled")
```

Unusable on any machine other than the original developer's. The `mkdir` on line 6 executes at import time, creating nonsensical directory trees on Linux.

### H2. `upscale.py:30-31` — `shlex.split` on Windows command string

```python
cmd = f'"{exe}" -i "{src}" -o "{out}" -n {model} -s {scale}'
proc = subprocess.run(shlex.split(cmd), ...)
```

`shlex.split` is a POSIX parser. On Windows it strips quotes from paths containing spaces, causing command failures.

### H3. `top10_cheers.py:498` — No-op single-quote escape

```python
normalized = full.as_posix().replace("'", "\'")
```

In Python, `"\'"` equals `"'"`. This replace is a no-op. File paths containing single quotes produce invalid FFmpeg concat files.

### H4. `top10_cheers.py:450-453` — Unconditional deletion of all files in output directory

```python
for old_file in output_dir.glob("*"):
    if old_file.is_file():
        old_file.unlink()
```

Deletes every file in `output_dir` before rendering. If misconfigured, this causes data loss. Should restrict to `*.mp4` or use a dedicated temp directory.

### H5. `top10_cheers.py:500` — ASCII encoding for concat file

```python
concat_path.write_text("\n".join(lines), encoding="ascii")
```

Paths containing non-ASCII characters cause `UnicodeEncodeError`. Should use `encoding="utf-8"`.

### H6. `email_processor.py:6` — Broken import path

```python
from src.correction_logger import configure_default_learning_client, safe_log_learning
```

`src/` has no `__init__.py`, so this import always fails with `ModuleNotFoundError`.

### H7. OpenCV variant conflict across dependency files

| File | Variant |
|------|---------|
| `pyproject.toml` | `opencv-python-headless` |
| `setup.cfg` | `opencv-python` |
| `requirements-autoframe.txt` | `opencv-python` |
| `requirements.lock.txt` | `opencv-python` |

These packages conflict — pip will fail or overwrite installations.

### H8. `requirements.lock.txt` — UTF-8 BOM character

Line 1 starts with a UTF-8 BOM (`\ufeff`), which can break pip parsing.

---

## MEDIUM-Severity Issues

### M1. `config.py:45` — YAML fallback parser truncates values containing `#`

```python
line = raw_line.split("#", 1)[0].rstrip()
```

Config values like `title: "Game #5"` are silently truncated to `title: "Game` when PyYAML is not installed.

### M2. `detect.py:99-103` — Array index / seconds unit conflation

`_expand_window` subtracts seconds from array indices. Works incidentally because bins ≈ 1 second, but breaks if binning changes.

### M3. `goals.py:335-337` — Silent mutation of caller's `HighlightWindow` objects

`detect_goal_windows` mutates the `windows` list passed by `detect_highlights()`, changing `event` and `score` fields before the caller's merge/sort/filter logic.

### M4. `ball_tracker.py:13-14` — Hard import of cv2 without try/except

Every other module wraps `import cv2` in try/except. `ball_tracker.py` does not, crashing on import if OpenCV is missing.

### M5. `io.py:39` — loguru `{}` formatting with stdlib logger fallback

```python
logger.debug("Running command: {}", " ".join(cmd))
```

Uses loguru-style `{}` placeholders. When loguru is not installed, the `_ProxyLogger` fallback uses `%`-style formatting, printing the `{}` literally.

### M6. `io.py:78` — `KeyError` if ffprobe data lacks `"format"` key

```python
duration = float(stream.get("duration") or data["format"].get("duration") or 0.0)
```

If `"format"` is missing from `data`, raises `KeyError`.

### M7. `shrink.py:206` — `AttributeError` if cv2 is None

`cv2` is conditionally set to `None` on import failure, but `smart_shrink()` uses `cv2.VideoCapture()` without checking, causing `AttributeError`.

### M8. `shrink.py:239` — All clip frames loaded into memory

```python
frames = list(_tracked_frames(cap, start_frame, end_frame))
```

Materializes all frames into a list. A 10s/30fps 1080p clip = ~1.8 GB RAM.

### M9. Six `cv2.VideoCapture` resource leaks across the core package

| File | Function | Lines |
|------|----------|-------|
| `detect.py` | `_compute_motion_scores` | 49-85 |
| `colors.py` | `calibrate_colors` | 52-92 |
| `rank.py` | `_activity_profile` | 197-226 |
| `goals.py` | `detect_scoreboard_deltas` | 139-190 |
| `goals.py` | `detect_net_events` | 227-265 |
| `ball_tracker.py` | CSV file handle | 307-314 |

All use `cap.release()` without `try/finally`, leaking handles on exceptions.

### M10. `_pydantic_compat.py:68` — Required fields silently default to `None`

The fallback `BaseModel` sets required fields (no default, no default_factory) to `None` instead of raising a validation error. This masks missing configuration at construction time.

### M11. `catalog.py:725` — Ambiguous operator precedence

```python
if ratio is not None and ratio >= 0.9 or (start_close and end_close):
```

Due to Python operator precedence: `(A and B) or C`, not `A and (B or C)`. The intent is unclear.

### M12. `ball_path_planner_v3.py:436-440` — Requires GUI, cannot run headless

`cv2.selectROI()` requires a display window. Cannot run in CI/CD or on headless servers.

### M13. Missing `__init__.py` in multiple directories

- `tests/` — breaks pytest discovery in some configurations
- `tools/` — breaks `from tools.utils import ...` in tests
- `examples/` — breaks `from examples.generate_sample import ...` in tests
- `src/` — breaks all imports from `src/`

### M14. Undeclared dependencies

| Module | Dependency | Where used |
|--------|-----------|------------|
| `goals.py` | `pytesseract` | OCR for scoreboard reading |
| `correction_logger.py` | `supabase` | Learning log backend |

### M15. `setup.cfg` vs `pyproject.toml` dependency divergence

`setup.cfg` is missing `pandas`, `soundfile`, and `tqdm`. Having two packaging files creates confusion about which is canonical.

### M16. 25+ PowerShell scripts with hardcoded `C:\Users\scott\...` paths

All scripts use the original developer's absolute Windows paths as defaults. Should use `$PSScriptRoot` or relative paths.

### M17. `render_follow_unified.py` — Debug print statements in production

Lines 7593-7637 contain `print("[DEBUG] ...")` statements that pollute stdout.

### M18. `render_follow_unified.py` — Self-import pattern

Lines 1111, 1214: `from render_follow_unified import load_any_telemetry` — the module imports from itself, creating fragility if the file is renamed.

---

## LOW-Severity Issues

### L1. `__init__.py:6-18` — Dead lazy-loading code overwritten by eager imports on lines 20-23

### L2. `rank.py:165` — `min(0, times.size - 1)` always evaluates to 0 (probably meant `max`)

### L3. `rank.py:391-392` — `k = k or config.rank.k` silently overrides `k=0`

### L4. `_tqdm.py:27` — Fallback `tqdm` returns raw iterable, discarding kwargs

### L5. `cli.py:223-225` — Dead code unreachable due to `required=True` on argparse subparsers

### L6. `reels.py:29-43` — `_parse_concat` silently drops entries if "outpoint" is missing

### L7. `colors.py:35` — `cv2.inRange` on float32 HSV (non-standard, works but fragile)

### L8. `top10_cheers.py:460-484` — Output-seeking (`-ss` after `-i`) is slower than input-seeking

### L9. `config.yaml` `branding:` section has no corresponding `BrandingConfig` model — silently ignored

### L10. `config.yaml` references non-existent paths (`fonts/Montserrat-ExtraBold.ttf`, `models/ball_v1.pt`)

### L11. `configs/zoom.yaml` has duplicate keys in `landscape:` profile

### L12. `track_orange_ball.py:12` — Exits with code 0 on error (should be non-zero)

### L13. `soundfile` declared as a core dependency but never imported by the installable package

### L14. `propagation_utils.py:22-24` — Bare `except Exception: pass` silently swallows errors

### L15. `render_follow_unified.py` — Duplicate imports (`import os` x2, `import sys` x2)

---

## Test Coverage Gaps

Modules with **zero test coverage**:
- `clips.py` (clip export)
- `reels.py` (reel rendering)
- `detect.py` (highlight detection)
- `config.py` (configuration loading)
- `colors.py` (color calibration)
- `io.py` (video I/O)
- `clip_gating.py` (clip filtering)
- `src/correction_logger.py`
- `src/email_processor.py`
- `src/propagation_utils.py`

Test infrastructure issues:
- No `conftest.py` for shared fixtures
- Tests import from `tools` and `examples` which lack `__init__.py`
- `test_ball_tracker.py` dynamically loads a 68KB module at collection time

---

## Recommended Fix Priority

### Phase 1 — Get the pipeline working (Blockers + High)
1. Fix `render_follow_unified.py` missing return statements (B1)
2. Fix `render_follow_unified.py` undefined variables (B2, B3, B4)
3. Fix `io.py` crash on missing metadata (B5, B6)
4. Replace hardcoded paths in `upscale.py` with relative/configurable paths (H1)
5. Fix `shlex.split` usage on Windows (H2)
6. Fix single-quote escaping in `top10_cheers.py` (H3)
7. Resolve OpenCV variant conflict (H7)
8. Add `__init__.py` to `src/`, `tools/`, `tests/`, `examples/` (M13)
9. Fix broken import in `email_processor.py` (H6)

### Phase 2 — Harden reliability (Medium)
10. Add `try/finally` to all `cv2.VideoCapture` usages (M9)
11. Fix YAML fallback parser `#` truncation (M1)
12. Fix logger format string mismatch in `io.py` (M5)
13. Guard against `cv2 is None` in `shrink.py` and `ball_tracker.py` (M4, M7)
14. Declare missing optional dependencies (M14)
15. Replace hardcoded paths in PowerShell scripts (M16)
16. Remove debug prints from `render_follow_unified.py` (M17)
17. Fix `_pydantic_compat.py` required field handling (M10)

### Phase 3 — Polish and tests
18. Add tests for untested core modules
19. Consolidate `setup.cfg` into `pyproject.toml` (eliminate duplicate)
20. Fix remaining low-severity issues
21. Add `conftest.py` with shared fixtures
22. Fix `requirements.lock.txt` BOM and completeness
23. Add `BrandingConfig` model or remove dead config section
