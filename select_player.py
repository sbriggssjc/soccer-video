import argparse, json, pathlib, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="out_json", required=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened(): raise SystemExit(f"Cannot open {args.src}")
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    win = "Select Player (frames: 0..%d) – press S to select, ENTER to accept, ESC to quit" % (N-1)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    cur = 0
    def on_trackbar(v):
        nonlocal cur
        cur = int(v)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = cap.read()
        if ok:
            cv2.imshow(win, frame)

    cv2.createTrackbar("frame", win, 0, max(0,N-1), on_trackbar)
    on_trackbar(0)

    sel = None
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord('s'), ord('S')):
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            ok, frame = cap.read()
            if not ok: continue
            r = cv2.selectROI("Draw player box, ENTER to accept", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Draw player box, ENTER to accept")
            if r is not None and r[2] > 4 and r[3] > 8:
                sel = {"frame": int(cur), "x": float(r[0]), "y": float(r[1]), "w": float(r[2]), "h": float(r[3]), "fps": fps}
                cv2.putText(frame, "Selected!", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.imshow(win, frame)
        elif key in (10,13):  # ENTER
            if sel:
                p = pathlib.Path(args.out_json)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w", encoding="utf-8") as f:
                    json.dump(sel, f, indent=2)
                print(f"[OK] Wrote selection -> {p}")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
