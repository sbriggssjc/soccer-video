import sys, csv, numpy as np, matplotlib.pyplot as plt, os
csv_path = sys.argv[1]
out_png  = os.path.splitext(csv_path)[0] + "_report.png"
N=[]; cx=[]; cy=[]; conf=[]; W=H=FPS=None
with open(csv_path,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        N.append(int(r["n"])); cx.append(float(r["cx"])); cy.append(float(r["cy"])); conf.append(float(r["conf"]))
        W=int(r["w"]); H=int(r["h"]); FPS=float(r["fps"])
N=np.array(N); cx=np.array(cx); cy=np.array(cy); conf=np.array(conf)
dt = 1.0/max(FPS,1.0)
vx=np.gradient(cx,dt); vy=np.gradient(cy,dt); spd=np.hypot(vx,vy)

print(f"frames={len(N)}, W×H={W}x{H}, FPS={FPS:.2f}")
print(f"conf<=0.1: {np.mean(conf<=0.1)*100:.1f}%   conf<=0.3: {np.mean(conf<=0.3)*100:.1f}%")
print(f"median speed: {np.median(spd):.1f}px/s   95th: {np.percentile(spd,95):.1f}px/s")
print("first 30 frames conf:", conf[:30])

plt.figure(figsize=(12,6))
ax1=plt.subplot(2,1,1); ax1.plot(N,cx,label="cx"); ax1.plot(N,cy,label="cy"); ax1.set_title("Ball track (px)"); ax1.legend(); ax1.grid(True,alpha=.3)
ax2=plt.subplot(2,1,2); ax2.plot(N,conf,label="conf"); ax2.plot(N,spd,alpha=.6,label="speed px/s"); ax2.set_ylim(bottom=0); ax2.set_title("Confidence & speed"); ax2.legend(); ax2.grid(True,alpha=.3)
plt.tight_layout(); plt.savefig(out_png, dpi=150)
print("wrote", out_png)
