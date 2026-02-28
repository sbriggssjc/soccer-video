import sys
target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"
with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()
pass1_idx = None
for i, line in enumerate(lines):
    if "# --- Pass 1: Centroid drift clamp ---" in line:
        pass1_idx = i
        break
if pass1_idx is None:
    print("ERROR")
    sys.exit(1)
print(f"Pass1 at {pass1_idx+1}", flush=True)
for i, line in enumerate(lines):
    if "v22i: Post-FUSION" in line:
        lines[i] = "            # == v22i/v22j: Post-FUSION phantom cleanup ====================\n"
        break
for i, line in enumerate(lines):
    if "# Pass 1 - Centroid drift clamp" in line:
        lines[i] = "            # Pass 0 - Low-conf YOLO filter (conf<0.25 -> HOLD)\n"
        lines.insert(i+1, "            # Pass 1 - Centroid drift clamp (source=2 drifting from YOLO)\n")
        pass1_idx += 1
        break
