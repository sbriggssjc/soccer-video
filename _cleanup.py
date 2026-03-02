import glob, os
for f in glob.glob(r"C:\Users\scott\Desktop\review_034.csv"):
    os.remove(f)
for f in glob.glob(r"C:\Users\scott\Desktop\filmstrip_034.png"):
    os.remove(f)
for f in glob.glob(r"D:\Projects\soccer-video\_tmp\render_direct_034_result.txt"):
    os.remove(f)
print("DONE")
