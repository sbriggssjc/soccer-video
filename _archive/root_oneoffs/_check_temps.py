import os, tempfile, glob
td = tempfile.gettempdir()
for pat in ["batch*", "neofc*"]:
    for f in glob.glob(os.path.join(td, pat)):
        sz = os.path.getsize(f) / (1024*1024)
        print(f"{f}  ({sz:.1f} MB)")
        os.remove(f)
        print(f"  DELETED")
print("Temp cleanup done")
