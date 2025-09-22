import io, re, pathlib, pandas as pd

p = pathlib.Path(r".\out\in_play.csv")
text = p.read_text(encoding="utf-8-sig", errors="replace")

# Try a few common delimiters
for sep in [",",";","|","\t"]:
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        if len(df.columns) > 1: break
    except Exception:
        df = None
if df is None:
    raise SystemExit("Could not read in_play.csv")

orig_cols = list(df.columns)
lower = {c:str(c).strip().lower() for c in df.columns}

start_alias = {"start","t0","begin","onset","start_sec","start_s","s","start (s)","start_seconds","from","a"}
end_alias   = {"end","t1","finish","offset","end_sec","end_s","e","end (s)","end_seconds","to","b","stop"}

def pick(colset):
    for c in df.columns:
        if lower[c] in colset:
            return c
    return None

s = pick(start_alias)
e = pick(end_alias)

# If still missing, look for ms columns
if s is None and "start_ms" in lower.values(): s = [c for c in df.columns if lower[c]=="start_ms"][0]
if e is None and "end_ms"   in lower.values(): e = [c for c in df.columns if lower[c]=="end_ms"][0]

# If a single "span"/"range" column exists like "123.4-130.0", split it
if (s is None or e is None):
    span = next((c for c in df.columns if lower[c] in {"span","segment","range"}), None)
    if span:
        parts = df[span].astype(str).str.extract(r"(?P<start>[\d:\.]+)\s*[-,;]\s*(?P<end>[\d:\.]+)")
        df["__start__"], df["__end__"] = parts["start"], parts["end"]
        s, e = "__start__", "__end__"

if s is None or e is None:
    raise SystemExit(f"Could not find start/end (cols={orig_cols})")

def to_seconds(val):
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    # hh:mm:ss(.ms) or mm:ss(.ms)
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", s):
        parts = [float(x) for x in s.split(":")]
        return parts[0]*3600 + parts[1]*60 + parts[2] if len(parts)==3 else parts[0]*60 + parts[1]
    # plain or comma decimal
    try:
        return float(s.replace(",", "."))
    except Exception:
        return float("nan")

# If inputs are ms, convert; otherwise parse to seconds
def maybe_ms(col):
    name = lower.get(col,"")
    return name.endswith("_ms") or name.endswith("msec") or name.endswith("millis")

start_vals = pd.to_numeric(df[s], errors="coerce")/1000.0 if maybe_ms(s) else df[s].map(to_seconds)
end_vals   = pd.to_numeric(df[e], errors="coerce")/1000.0 if maybe_ms(e) else df[e].map(to_seconds)

out = pd.DataFrame({"start": start_vals, "end": end_vals})
out = out.dropna()
out = out[out["end"] > out["start"]].round(3)

out.to_csv(p, index=False)
print("in_play.csv normalized to columns: start,end")
print(out.head().to_string(index=False))

