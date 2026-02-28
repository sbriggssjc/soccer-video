"""Create contact sheets - 4x3 grids of key frames with ball markers"""
import cv2, os, numpy as np

src = r'D:\Projects\soccer-video\_optflow_track\verify_frames'
dst = r'D:\Projects\soccer-video\_optflow_track'

files = sorted([f for f in os.listdir(src) if f.endswith('.png')])

# Make contact sheets of 12 frames each, resized to 480x270 each
thumb_w, thumb_h = 480, 270
cols, rows_per = 4, 3
per_sheet = cols * rows_per

for sheet_idx in range(0, len(files), per_sheet):
    batch = files[sheet_idx:sheet_idx + per_sheet]
    canvas = np.zeros((rows_per * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    
    for i, fn in enumerate(batch):
        img = cv2.imread(os.path.join(src, fn))
        small = cv2.resize(img, (thumb_w, thumb_h))
        r, c = divmod(i, cols)
        y0, x0 = r * thumb_h, c * thumb_w
        canvas[y0:y0+thumb_h, x0:x0+thumb_w] = small
    
    out = os.path.join(dst, f'contact_sheet_{sheet_idx//per_sheet + 1}.jpg')
    cv2.imwrite(out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
    sz = os.path.getsize(out)
    print(f"Sheet {sheet_idx//per_sheet + 1}: {len(batch)} frames -> {out} ({sz//1024}KB)")
    print(f"  Frames: {', '.join(batch)}")
