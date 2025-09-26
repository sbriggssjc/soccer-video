import sys, pandas as pd

src = sys.argv[1]
dst = sys.argv[2]

# Read with autodetected delimiter and ignoring commented rows
df = pd.read_csv(src, comment="#", sep=None, engine="python")

# Case-insensitive lookup
lc = {c.lower(): c for c in df.columns}

# Add/adjust aliases here if your CSV uses different names
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
    "frame":  df[pick("frame")],
    "ball_x": df[pick("ball_x")],
    "ball_y": df[pick("ball_y")],
})
out.to_csv(dst, index=False)
print(f"Wrote {dst} with columns {list(out.columns)} and {len(out)} rows.")
