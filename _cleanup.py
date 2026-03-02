import glob, os
for f in glob.glob(r"C:\Users\scott\Desktop\review_031.csv"):
    os.remove(f)
for f in glob.glob(r"C:\Users\scott\Desktop\filmstrip_031.png"):
    os.remove(f)
for f in glob.glob(r"D:\Projects\soccer-video\_tmp\render_direct_031_result.txt"):
    os.remove(f)
print("DONE")
