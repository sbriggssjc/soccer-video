import subprocess, os
inp = r"D:\Projects\soccer-video\_debug_frames"
out = r"D:\Projects\soccer-video\_debug_tiny"
os.makedirs(out, exist_ok=True)
for fn in os.listdir(inp):
    if fn.endswith(".jpg"):
        inf = os.path.join(inp, fn)
        outf = os.path.join(out, fn.replace(".jpg",".png"))
        subprocess.run(["ffmpeg","-y","-i",inf,"-vf","scale=iw/5:ih/5","-q:v","8",outf], capture_output=True)
print("DONE", flush=True)
