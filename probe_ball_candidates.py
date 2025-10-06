import os,sys,cv2,numpy as np
vid=sys.argv[1]; outdir=sys.argv[2]; weights=sys.argv[3]
use_yolo=False; yolo=None
if os.path.exists(weights) and os.path.getsize(weights)>0:
    try:
        from ultralytics import YOLO
        yolo=YOLO(weights); use_yolo=True
    except: pass

cap=cv2.VideoCapture(vid)
if not cap.isOpened(): raise SystemExit("cannot open "+vid)
H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
def mask_hsv(img, lo, hi):
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lo), np.array(hi))
# bands
RED1=( (0,60,60), (8,255,255) )
RED2=( (165,60,60),(179,255,255) )
ORNG=( (8,60,60), (22,255,255) )
WHITE=( (0,0,200), (179,40,255) )   # low S, high V
GREEN=( (35,30,40),(95,255,255) )

for n in range(0, min(120, int(cap.get(cv2.CAP_PROP_FPS))*2)):
    ok,frm = cap.read()
    if not ok: break
    mR1=mask_hsv(frm,*RED1); mR2=mask_hsv(frm,*RED2); mO=mask_hsv(frm,*ORNG); mW=mask_hsv(frm,*WHITE)
    # veto green field to reduce lines
    mG = mask_hsv(frm,*GREEN)
    mAll = cv2.bitwise_or(cv2.bitwise_or(mR1,mR2), cv2.bitwise_or(mO,mW))
    mAll = cv2.bitwise_and(mAll, cv2.bitwise_not(mG))
    # Hough circles on gray
    g=cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY); g=cv2.GaussianBlur(g,(7,7),1.5)
    circles=cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=120, param2=20, minRadius=5, maxRadius=60)

    vis = frm.copy()
    # draw masks outlines
    for m,color in [(mR1,(0,0,255)),(mR2,(0,0,200)),(mO,(0,128,255)),(mW,(255,255,255))]:
        cnts,_=cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c)<30: continue
            cv2.drawContours(vis,[c],-1,color,1)
    # draw circles
    if circles is not None:
        for x,y,r in np.uint16(np.around(circles))[0,:]:
            cv2.circle(vis,(x,y),r,(0,255,255),2)

    # YOLO boxes
    if use_yolo:
        try:
            rs=yolo.predict(source=frm, conf=0.25, imgsz=max(640,((max(W,H)+31)//32)*32), verbose=False)
            if len(rs) and getattr(rs[0],"boxes",None) is not None:
                b=rs[0].boxes
                for i in range(len(b)):
                    x1,y1,x2,y2 = b.xyxy[i].cpu().numpy().tolist()
                    cv2.rectangle(vis,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        except: pass

    cv2.putText(vis,f"frame {n}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),3)
    cv2.putText(vis,f"frame {n}",(12,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),1)
    cv2.imwrite(os.path.join(outdir,f"{n:04d}.png"), vis)
cap.release()
print("probe PNGs ->", outdir)
