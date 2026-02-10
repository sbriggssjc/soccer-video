import csv, io, re, pathlib
import pandas as pd

p = pathlib.Path(r".\out\in_play.csv")
raw = p.read_bytes()
text = raw.decode("utf-8-sig", "replace")

# Sniff delimiter
sniffer = csv.Sniffer()
dialect = None
for delim in [",",";","\t","|"]:
    try:
        dialect = csv.get_dialect("sniffed")
        break
    except Exception: pass
try:
    dialect = sniffer.sniff(text.splitlines()[0] if "\n" in text else text, delimiters=";,|\t,")
except Exception:
    class D: delimiter=","
    dialect = D()

df = pd.read_csv(io.StringIO(text), sep=getattr(dialect,"delimiter",","), engine="python")
orig_cols = list(df.columns)

# Lower/strip cols
df.columns = [str(c).strip() for c in df.columns]
lower = {c: c.lower().strip() for c in df.columns}

start_aliases = {"start","t0","begin","onset","start_sec","start_s","s","s_sec","start (s)","start_seconds"}
end_aliases   = {"end","t1","finish","offset","end_sec","end_s","e","e_sec","end (s)","end_seconds"}
start_ms_aliases = {"start_ms","s_ms","start_msec","start_millis"}
end_ms_aliases   = {"end_ms","e_ms","end_msec","end_millis"}

def find_col(cands):
    for c in df.columns:
        if lower[c] in cands: return c
    return None

s = find_col(start_aliases)
e = find_col(end_aliases)

# Support ms-only headers
if s is None:
    s_ms = find_col(start_ms_aliases)
    if s_ms is not None:
        df["start"] = pd.to_numeric(df[s_ms], errors="coerce")/1000.0
        s = "start"
if e is None:
    e_ms = find_col(end_ms_aliases)
    if e_ms is not None:
        df["end"] = pd.to_numeric(df[e_ms], errors="coerce")/1000.0
        e = "end"

# If still missing, try a single "span" col like "123.4-130.0"
if s is None or e is None:
    span = None
    for c in df.columns:
        if re.fullmatch(r"(span|segment|range)", lower[c]):
            span = c; break
    if span:
        se = df[span].astype(str).str.extract(r"(?P<start>[\d:\.]+)\s*[-,;]\s*(?P<end>[\d:\.]+)")
        df["start"], df["end"] = se["start"], se["end"]
        s, e = "start", "end"

if s is None or e is None:
    raise SystemExit(f"Could not find start/end in {p} (cols={orig_cols})")

def to_seconds(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan": return float("nan")
    # mm:ss or hh:mm:ss(.ms)
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", s):
        parts = [float(t) for t in s.split(":")]
        if len(parts)==2:  # mm:ss
            return parts[0]*60 + parts[1]
        elif len(parts)==3:  # hh:mm:ss
            return parts[0]*3600 + parts[1]*60 + parts[2]
    # plain number
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return float("nan")

out = pd.DataFrame({
    "start": df[s].map(to_seconds),
    "end":   df[e].map(to_seconds),
})

out = out.dropna()
out = out[out["end"] > out["start"]]
# round to ms for ffconcat sanity
out = out.round({"start":3, "end":3})

# Write back normalized CSV
out.to_csv(p, index=False)
print("Normalized in_play.csv -> start,end")
print(out.head().to_string(index=False))

