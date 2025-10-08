import json, collections
p=r"out\render_logs\tester_022__SHOT.jsonl"
u=collections.Counter(); xs=[]; ys=[]
for l in open(p,'r',encoding='utf-8'):
    d=json.loads(l); u[d.get("used","?")]+=1
    x0,y0,_,_=d.get("crop",[0,0,0,0]); xs.append(x0); ys.append(y0)
print("used:", dict(u))
print("x0 min/max:", min(xs), max(xs))
print("y0 min/max:", min(ys), max(ys))
