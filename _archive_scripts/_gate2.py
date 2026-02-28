import sys
target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"
with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the Pass 0 start to add a guard that skips to the logger
old = """            # --- Pass 0: Low-conf YOLO -> HOLD ---
            _prev_trusted_x = None"""
new = """            if not _PHANTOM_CLEANUP_ENABLED:
                _phantom_fixes = 0  # skip all passes
            else:
              # --- Pass 0: Low-conf YOLO -> HOLD ---
              _prev_trusted_x = None"""

if old not in content:
    print("ERROR: Pass 0 start not found")
    sys.exit(1)
content = content.replace(old, new, 1)

# Now indent ALL the code between Pass 0 start and the phantom logger
# Actually this is too complex. Let me try a different approach.
# Just replace the 3 pass headers with early returns.
print("Trying different approach...", flush=True)

# Revert the change
content = content.replace(new, old, 1)

# SIMPLEST: Just make each pass check the flag at the top
# Pass 0: replace "# --- Pass 0:" line with "if _PHANTOM_CLEANUP_ENABLED: # --- Pass 0:"
# and indent the body... no, that's the same problem.

# ACTUALLY SIMPLEST: remove the entire block and replace with pass.
# Find from "# --- Pass 0:" to "if _phantom_fixes > 0:" logger inclusive
start_marker = "            # --- Pass 0: Low-conf YOLO -> HOLD ---"
end_marker = '            if _phantom_fixes > 0:\n                logger.info("[PHANTOM] Post-FUSION cleanup: fixed %d phantom frames", _phantom_fixes)'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)
if start_idx < 0 or end_idx < 0:
    print(f"ERROR: markers not found start={start_idx} end={end_idx}")
    sys.exit(1)

end_idx += len(end_marker)
# Also eat trailing newlines
while end_idx < len(content) and content[end_idx] == '\n':
    end_idx += 1

removed = content[start_idx:end_idx]
content = content[:start_idx] + content[end_idx:]

with open(target, "w", encoding="utf-8") as f:
    f.write(content)

lines_removed = removed.count('\n')
print(f"SUCCESS: Removed {lines_removed} lines of phantom cleanup passes", flush=True)
print(f"Total chars: {len(content)}", flush=True)
