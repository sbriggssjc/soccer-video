import subprocess, os, shutil
tiny = r"D:\Projects\soccer-video\_debug_tiny"
# Find the cowork outputs mount
for candidate in [
    os.path.expanduser(r"~\Desktop\cowork\outputs\_debug_tiny"),
    os.path.expanduser(r"~\Documents\cowork\outputs\_debug_tiny"),
]:
    pass

# Just list what we have
for fn in sorted(os.listdir(tiny)):
    fp = os.path.join(tiny, fn)
    sz = os.path.getsize(fp)
    print(f"{fn}: {sz} bytes", flush=True)
