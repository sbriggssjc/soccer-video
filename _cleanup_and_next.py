import os, glob

# Clean Desktop CSVs and PNGs from batch 006-015
cleaned = 0
for pat in [r"C:\Users\scott\Desktop\review_*.csv",
            r"C:\Users\scott\Desktop\filmstrip_*.png"]:
    for f in glob.glob(pat):
        os.remove(f)
        cleaned += 1
print(f"Desktop: cleaned {cleaned} files")

# Clean temp files
for pat in [r"D:\Projects\soccer-video\_tmp\batch_render_result.txt",
            r"D:\Projects\soccer-video\_tmp\filmstrip_*_frames"]:
    for f in glob.glob(pat):
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            import shutil
            shutil.rmtree(f)
print("Temp files cleaned")
