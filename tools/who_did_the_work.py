import json, collections
p=r"out\render_logs\tester_022__SHOT.jsonl"
u=collections.Counter()
for l in open(p,'r',encoding='utf-8'):
    u[json.loads(l).get("used","?")] += 1
print(dict(u))
