# Quick QC Gates

These steps provide a rapid signal on whether a clip's tracking and rendering outputs look healthy. Run them at the listed points in the workflow.

## After Planning

```bash
python tools/path_qc.py
```

* Expect the 95th percentile speed (``spd p95``) to stay below roughly **120 px/frame** for 24 fps clips.
* Look for only small, infrequent speed spikes.

## Visual Path Check

Overlay the planned ball path onto a rendered clip to visually confirm tracking quality:

```bash
python tools/overlay_path_samples.py --in ".\out\atomic_clips\...\022__SHOT__t3028.10-t3059.70.mp4" --path "out\render_logs\tester_022__SHOT.ball.jsonl"
```

## After Rendering

```bash
python tools/diag_crop_guarantee.py --telemetry "out/render_logs/tester_022__SHOT.jsonl"
```

* Expect the `outside_crop` metric to remain near zero.

## Telemetry Sanity Check

Create the following helper script to verify telemetry coverage:

```powershell
@'
import json, collections
p = r"out\render_logs\tester_022__SHOT.jsonl"
have_ball = have_crop = 0
used = collections.Counter()
for l in open(p, 'r', encoding='utf-8'):
    d = json.loads(l)
    have_ball += int("ball" in d)
    have_crop += int("crop" in d)
    used[d.get("used", "?")] += 1
print("ball frames:", have_ball, "crop frames:", have_crop, "used:", dict(used))
'@ | Set-Content -Encoding UTF8 tools\telemetry_quickcheck.py
python tools/telemetry_quickcheck.py
```

If ball frames are low or the `used` counts show many `planner` or `tm_fail` entries, loosen the re-associate threshold.

