import sys, json, pathlib
p = pathlib.Path(r"out\render_logs\tester_022__SHOT.ball.jsonl")
if not p.exists():
    print("NO BALL PATH WRITTEN"); sys.exit(2)
n = sum(1 for _ in p.open("r", encoding="utf-8"))
print("ballpath lines:", n)
