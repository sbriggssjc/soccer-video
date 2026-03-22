import json
path = 'out/telemetry/002__2026-02-23__TSC_vs_NEOFC__BUILD_AND_SHOTS__t17.00-t32.00.yolo_ball.jsonl'
with open(path) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        if d.get('_meta'):
            continue
        print(f"Entry {i}: keys={list(d.keys())}")
        print(f"  Values: {d}")
        break
