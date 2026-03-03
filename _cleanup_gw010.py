import os
desk = "C:/Users/scott/Desktop"
for f in ["review_gw_010.csv", "filmstrip_gw_010.jpg"]:
    p = os.path.join(desk, f)
    if os.path.exists(p):
        try:
            os.remove(p)
            print(f"DEL: {f}")
        except Exception as e:
            print(f"SKIP {f}: {e}")
print("Done")
