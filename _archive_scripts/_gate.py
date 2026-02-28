import sys
target = r"D:\Projects\soccer-video\tools\render_follow_unified.py"
with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# Wrap Pass 0 block
old = "            # --- Pass 0: Low-conf YOLO -> HOLD ---\n            _prev_trusted_x"
new = "            # --- Pass 0: Low-conf YOLO -> HOLD ---\n            if _PHANTOM_CLEANUP_ENABLED:\n              _prev_trusted_x"
if old not in content:
    print("ERROR: Pass 0 marker not found")
    sys.exit(1)
content = content.replace(old, new, 1)

# Wrap Pass 1 block
old2 = "            # --- Pass 1: Centroid drift clamp ---\n            for _fi in range(len(positions)):\n                if fusion_source_labels[_fi] != 2:"
new2 = "            # --- Pass 1: Centroid drift clamp ---\n            if _PHANTOM_CLEANUP_ENABLED:\n             for _fi in range(len(positions)):\n                if fusion_source_labels[_fi] != 2:"
# Actually this approach of re-indenting is fragile. Let me just
# gate the entire section differently.
# Better: find the start of Pass 0 and end (the phantom logger),
# and wrap everything in a single if block.
print("Using simpler approach", flush=True)
