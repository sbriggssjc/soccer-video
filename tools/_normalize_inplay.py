import pandas as pd, pathlib
p = pathlib.Path(r".\out\in_play.csv")
df = pd.read_csv(p)

lower = {c: c.lower() for c in df.columns}
start_aliases = {"start","t0","begin","onset","start_sec","start_s"}
end_aliases   = {"end","t1","finish","offset","end_sec","end_s"}

s = next((c for c in df.columns if lower[c] in start_aliases), None)
e = next((c for c in df.columns if lower[c] in end_aliases),   None)
if not (s and e):
    raise SystemExit(f"Could not find start/end in {p} (cols={list(df.columns)})")

df[[s,e]].rename(columns={s:"start", e:"end"}).to_csv(p, index=False)
print("normalized:", p)

