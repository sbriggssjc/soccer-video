"""Create smaller contact sheets - 3x2 grids, lower quality"""
import cv2, os, numpy as np

src = r'D:\Projects\soccer-video\_optflow_track\verify_frames'
dst = r'D:\Projects\soccer-video\_optflow_track'

# Pick the most important frames
key_frames = [
    # Sheet 1: Early/mid (build-up)
    ['f0000.png', 'f0090.png', 'f0140.png', 'f0200.png', 'f0250.png', 'f0270.png'],
    # Sheet 2: Shot zone (critical!)
    ['f0290.png', 'f0300.png', 'f0310.png', 'f0314.png', 'f0330.png', 'f0360.png'],
    # Sheet 3: Post-shot and restart
    ['f0380.png', 'f0400.png', 'f0417.png', 'f0425.png', 'f0450.png', 'f0480.png'],
]

thumb_w, thumb_h = 320, 180
cols, rows_per = 3, 2

for si, batch in enumerate(key_frames):
    canvas = np.zeros((rows_per * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, fn in enumerate(batch):
        path = os.path.join(src, fn)
        if not os.path.exists(path): continue
        img = cv2.imread(path)
        small = cv2.resize(img, (thumb_w, thumb_h))
        r, c = divmod(i, cols)
        y0, x0 = r * thumb_h, c * thumb_w
        canvas[y0:y0+thumb_h, x0:x0+thumb_w] = small
    
    out = os.path.join(dst, f'verify_sheet_{si+1}.jpg')
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 60])
    sz = os.path.getsize(out)
    print(f"Sheet {si+1}: {out} ({sz//1024}KB) - {batch}")
