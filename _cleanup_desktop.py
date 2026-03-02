import glob, os
desktop = r"C:\Users\scott\Desktop"
count = 0
for pat in ["filmstrip_*.png", "review_*.csv", "clip_*.mp4"]:
    for f in glob.glob(os.path.join(desktop, pat)):
        os.remove(f)
        count += 1
        print(f"Deleted: {os.path.basename(f)}")
print(f"\nTotal deleted: {count}")
remaining = glob.glob(os.path.join(desktop, "filmstrip_*")) + glob.glob(os.path.join(desktop, "review_*"))
print(f"Remaining filmstrip/review files: {len(remaining)}")
