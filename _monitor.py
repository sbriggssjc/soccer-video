import time
for i in range(6):
    time.sleep(30)
    try:
        with open(r"D:\Projects\soccer-video\_tmp\batch_render_result.txt") as f:
            lines = f.readlines()
        print(f"=== Check {i+1} ({(i+1)*30}s) === {len(lines)} lines")
        for ln in lines[-5:]:
            print(ln.rstrip())
    except Exception as e:
        print(f"Check {i+1}: {e}")
print("MONITOR DONE")
