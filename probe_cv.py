import cv2, sys
for p in [r"recordings\raw\20250913_214118000_iOS.MP4", r"recordings\raw\20250913_221300000_iOS.MP4"]:
    cap = cv2.VideoCapture(p)
    print("OPEN", p, cap.isOpened())
    ok, frame = cap.read()
    print("READ", p, ok, (None if frame is None else frame.shape))
    cap.release()
    if not (cap.isOpened() and ok and frame is not None):
        sys.exit(2)
