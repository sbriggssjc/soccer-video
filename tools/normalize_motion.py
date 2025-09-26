import sys, pandas as pd

src = sys.argv[1]
dst = sys.argv[2]
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0

df = pd.read_csv(src, comment="#", sep=None, engine="python")
lc = {c.lower(): c for c in df.columns}
want = {
    "frame":  ["frame","n","idx","frame_id","f"],
    "ball_x": ["ball_x","x","cx","ball.cx","center_x","ballx"],
    "ball_y": ["ball_y","y","cy","ball.cy","center_y","bally"],
}
def pick(k):
    for a in want[k]:
        if a in lc: return lc[a]
    raise KeyError(f"Missing column for {k}. Found: {list(df.columns)}")

out = pd.DataFrame({
    "frame":  pd.to_numeric(df[pick("frame")],  errors="coerce"),
    "ball_x": pd.to_numeric(df[pick("ball_x")], errors="coerce"),
    "ball_y": pd.to_numeric(df[pick("ball_y")], errors="coerce"),
})
out = out.dropna().sort_values("frame")
out["frame"] = out["frame"].astype(int)
out["t"] = out["frame"] / fps
out.to_csv(dst, index=False)
print(f"Wrote {dst} ({len(out)} rows) at {fps} fps.")
