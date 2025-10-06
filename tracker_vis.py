import sys, os, csv, cv2, numpy as np
vid, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
cap=cv2.VideoCapture(vid)
if not cap.isOpened(): raise SystemExit("cannot open "+vid)
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
rows=[]
with open(csv_path,"r",newline="") as f:
    rd=csv.DictReader(f)
    for r in rd:
        rows.append((int(r["n"]), float(r["cx"]), float(r["cy"]), float(r["conf"])))
n=0; ok,frm=cap.read()
while ok and n<len(rows):
    _, x, y, c = rows[n]
    color=(0,0,255) if c>=0.35 else (0,255,255)
    cv2.circle(frm,(int(round(x)),int(round(y))), 10, color, 3)
    if n%2==0:
        cv2.imwrite(os.path.join(out_dir,f"{n:06d}.png"),frm)
    ok,frm=cap.read(); n+=1
cap.release()
print("wrote PNGs to", out_dir)
