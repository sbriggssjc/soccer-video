import glob, os
for f in glob.glob(r"D:\Projects\soccer-video\_tmp\render_direct_003_result.txt"):
    os.remove(f)
print("DONE")
