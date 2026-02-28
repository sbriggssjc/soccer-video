target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"
with open(target, "r", encoding="utf-8") as f:
    lines = f.readlines()
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if "_PHANTOM_CLEANUP_ENABLED" in line and start_idx is None:
        start_idx = i
    if "return best" in line and start_idx is not None and end_idx is None:
        end_idx = i
print(f"start={start_idx} end={end_idx}")
if start_idx is not None and end_idx is not None:
    while end_idx + 1 < len(lines) and lines[end_idx + 1].strip() == "":
        end_idx += 1
    new_lines = lines[:start_idx] + lines[end_idx + 1:]
    with open(target, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"Removed {end_idx - start_idx + 1} lines, total={len(new_lines)}")
else:
    print("Nothing to remove (already clean)")
