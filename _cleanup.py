import glob, os
for f in glob.glob(r"C:\Users\scott\Desktop\review_005.csv"):
    os.remove(f)
for f in glob.glob(r"C:\Users\scott\Desktop\filmstrip_005.png"):
    os.remove(f)
print("DONE")
