import os,sys,cv2,numpy as np
vid=sys.argv[1]; outdir=sys.argv[2]; weights=sys.argv[3]
use_yolo=False
try:
  if os.path.exists(weights):
    from ultralytics import YOLO
    yolo=YOLO(weights); use_yolo=True
except: pass

cap=cv2.VideoCapture(vid)
if not cap.isOpened(): raise SystemExit("cannot open "+vid)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
RED1=((0,120,80),(8,255,255))
RED2=((165,120,80),(179,255,255))
ORNG=((8,120,80),(20,255,255))
GREEN=((35,25,40),(95,255,255))   # stronger veto

def mask_red(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv,np.array(RED1[0]),np.array(RED1[1]))|cv2.inRange(hsv,np.array(RED2[0]),np.array(RED2[1]))|cv2.inRange(hsv,np.array(ORNG[0]),np.array(ORNG[1]))
    g=cv2.inRange(hsv,np.array(GREEN[0]),np.array(GREEN[1]))
    m=cv2.bitwise_and(m, cv2.bitwise_not(g))
    m=cv2.medianBlur(m,5)
    m=cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    return m

for n in range(int(2*fps)):
    ok,frm=cap.read()
    if not ok: break
    m=mask_red(frm)
    # Hough on masked gray only
    gry=cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
    gry=cv2.bitwise_and(gry, gry, mask=m)
    gry=cv2.GaussianBlur(gry,(7,7),1.4)
    cir=cv2.HoughCircles(gry, cv2.HOUGH_GRADIENT, dp=1.3, minDist=26, param1=140, param2=28, minRadius=7, maxRadius=30)

    vis=frm.copy()
    # show mask edges lightly
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c)<30: continue
        cv2.drawContours(vis,[c],-1,(0,165,255),1)
    if cir is not None:
        for x,y,r in np.uint16(np.around(cir))[0,:]:
            cv2.circle(vis,(x,y),r,(0,255,255),2)
    if use_yolo:
        try:
            rs=yolo.predict(source=frm, conf=0.25, imgsz=640, verbose=False)
            if len(rs) and getattr(rs[0],"boxes",None) is not None:
                for b in rs[0].boxes:
                    x1,y1,x2,y2=b.xyxy[0].cpu().numpy()
                    cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        except: pass
    cv2.putText(vis,f"frame {n}",(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
    cv2.putText(vis,f"frame {n}",(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
    cv2.imwrite(os.path.join(outdir,f"{n:04d}.png"),vis)
cap.release()
print("probe PNGs ->", outdir)
