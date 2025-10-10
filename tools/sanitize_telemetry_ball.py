import json, math, sys, pathlib as p
from typing import Any

inp = p.Path(sys.argv[1])
outp = p.Path(sys.argv[2]) if len(sys.argv) > 2 else inp

clean = []
for line in inp.read_text(encoding="utf-8").splitlines():
    if not line.strip(): 
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue

    b = d.get("ball", None)
    bx = by = None
    if isinstance(b, dict):
        bx, by = b.get("x"), b.get("y")
    elif isinstance(b, (list, tuple)) and len(b) >= 2:
        bx, by = b[0], b[1]

    ok = (bx is not None and by is not None 
          and isinstance(bx, (int, float)) and isinstance(by, (int, float))
          and math.isfinite(bx) and math.isfinite(by))

    if not ok:
        # remove invalid ball so overlay_debug just skips drawing the red dot
        if "ball" in d:
            del d["ball"]

    clean.append(json.dumps(d))

outp.write_text("\n".join(clean), encoding="utf-8")
print(f"[sanitize] wrote {len(clean)} rows -> {outp}")
