import json, collections, sys
p = r"out\render_logs\tester_022__SHOT.jsonl"
xs=[]; ys=[]; used=collections.Counter()
with open(p,"r",encoding="utf-8") as f:
    for l in f:
        d=json.loads(l)
        x0,y0,_,_ = d.get("crop",[0,0,0,0])
        xs.append(x0); ys.append(y0)
        used[d.get("used","?")] += 1
print("x0 min/max:", min(xs), max(xs))
print("y0 min/max:", min(ys), max(ys))
print("used:", dict(used))
