import os
desk = "C:/Users/scott/Desktop"
removed = 0
for i in range(1, 6):
    for pattern in [f"review_neofc_{i:03d}.csv", f"filmstrip_neofc_{i:03d}.jpg"]:
        p = os.path.join(desk, pattern)
        if os.path.exists(p):
            os.remove(p)
            print(f"DEL: {pattern}")
            removed += 1
print(f"Removed {removed} files")
