import sys, pandas as pd

src = sys.argv[1]
dst = sys.argv[2]
fps = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0

df = pd.read_csv(src, comment="#", sep=None, engine="python")

# Case-insensitive lookup
lc = {c.lower(): c for c in df.columns}

# Aliases (add/adjust if your CSV uses different names)
want = {
    "frame":  ["frame","n","idx","frame_id","f"],
    "ball_x": ["ball_x","x","cx","ball.cx","center_x","ballx"],
    "ball_y": ["ball_y","y","cy","ball.cy","center_y","bally"],
}

def pick(key):
    for alias in want[key]:
        if alias in lc:
            return lc[alias]
    raise KeyError(f"Missing column for {key}. Found: {list(df.columns)}")

out = pd.DataFrame({
    "frame":  pd.to_numeric(df[pick("frame")], errors="coerce"),
    "ball_x": pd.to_numeric(df[pick("ball_x")], errors="coerce"),
    "ball_y": pd.to_numeric(df[pick("ball_y")], errors="coerce"),
})

# Drop rows with missing essentials and sort
out = out.dropna(subset=["frame","ball_x","ball_y"]).sort_values("frame")
out["frame"] = out["frame"].astype(int)

# Time (seconds) from fps
out["t"] = out["frame"] / fps

out.to_csv(dst, index=False)
print(f"Wrote {dst} with columns {list(out.columns)} and {len(out)} rows at {fps} fps.")
