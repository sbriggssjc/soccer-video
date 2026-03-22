"""Wait for Greenwood batch to finish, then render NEOFC clip 005."""
import time, os, subprocess, sys

BATCH_RESULT = r"D:\Projects\soccer-video\_tmp\batch_greenwood_0221_result.txt"
PYTHON = sys.executable
RENDER_SCRIPT = r"D:\Projects\soccer-video\_apply_and_render.py"
STATUS_FILE = r"D:\Projects\soccer-video\_tmp\render_005_status.txt"

def log(msg):
    with open(STATUS_FILE, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    print(msg)

with open(STATUS_FILE, "w") as f:
    f.write("Waiting for Greenwood batch to complete...\n")

# Poll batch result file for completion
while True:
    time.sleep(30)
    try:
        with open(BATCH_RESULT, "r") as f:
            content = f.read()
        if "BATCH COMPLETE" in content:
            log("Greenwood batch finished! Starting NEOFC 005 render...")
            break
        if "EXCEPTION" in content:
            log("Greenwood batch hit an exception, starting NEOFC 005 anyway...")
            break
        # Log last line for monitoring
        lines = [l for l in content.strip().split("\n") if l.strip()]
        if lines:
            log(f"Batch still running: {lines[-1].strip()}")
    except Exception as e:
        log(f"Error reading batch status: {e}")

# Now run the NEOFC 005 render
log("Launching NEOFC clip 005 render...")
os.chdir(r"D:\Projects\soccer-video")
result = subprocess.run([PYTHON, RENDER_SCRIPT], capture_output=True, text=True, timeout=1200)
if result.returncode == 0:
    log("NEOFC clip 005 render COMPLETE!")
else:
    log(f"NEOFC clip 005 render FAILED (rc={result.returncode})")
    log((result.stderr or "")[-1000:])
