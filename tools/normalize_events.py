# tools/normalize_events.py
import argparse, csv, re, sys, math, pathlib

def parse_time(val):
    """Accepts 'SS', 'SS.S', 'MM:SS', 'MM:SS.s', 'HH:MM:SS', 'HH:MM:SS.s' -> seconds (float)."""
    if val is None: return None
    s = str(val).strip()
    if s == "": return None
    # If plain number, return as float
    try:
        return float(s)
    except ValueError:
        pass
    # hh:mm[:ss[.fff]]
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = parts
            return float(m) * 60 + float(sec)
        elif len(parts) == 3:
            h, m, sec = parts
            return float(h) * 3600 + float(m) * 60 + float(sec)
    except ValueError:
        pass
    raise ValueError(f"Unrecognized time format: {val!r}")

def first_present(row, names):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input events CSV (timestamps like 02:35.5 okay)")
    ap.add_argument("--out", dest="outp", required=True, help="Output events CSV with numeric t0,t1 seconds")
    ap.add_argument("--label-col", default=None, help="Optional label/title/event column name to preserve")
    args = ap.parse_args()

    inp = pathlib.Path(args.inp)
    outp = pathlib.Path(args.outp)

    rows = list(csv.DictReader(inp.open(newline="", encoding="utf-8-sig")))
    if not rows:
        print("Input CSV is empty.", file=sys.stderr)
        sys.exit(2)

    # Try to detect columns
    # common starts: t0/start/start_sec/start_ms/ts_start
    # common ends:   t1/end/end_sec/end_ms/ts_end
    start_cand = first_present(rows[0], ["t0","start","start_sec","start_ms","ts_start","begin"])
    end_cand   = first_present(rows[0], ["t1","end","end_sec","end_ms","ts_end","finish"])
    if not start_cand or not end_cand:
        # Try windowed schema: t + pre/post
        t_cand   = first_present(rows[0], ["t","time","timestamp"])
        pre_cand = first_present(rows[0], ["pre","pre_sec","lead","prebuffer"])
        post_cand= first_present(rows[0], ["post","post_sec","tail","postbuffer"])
        if not (t_cand and pre_cand and post_cand):
            raise SystemExit("Could not detect start/end or t/pre/post columns in CSV.")
        mode = "window"
    else:
        mode = "range"

    # label column?
    label_col = args.label_col
    if label_col is None:
        label_col = first_present(rows[0], ["label","title","event","name","clip","play"])

    # Write normalized CSV with t0,t1 (seconds) + label if present + keep any extras
    fieldnames = list(rows[0].keys())
    for col in ["t0","t1"]:
        if col not in fieldnames: fieldnames.append(col)
    if label_col and label_col not in fieldnames:
        fieldnames.append(label_col)

    with outp.open("w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r2 = dict(r)
            if mode == "range":
                s_raw = r.get(start_cand, "")
                e_raw = r.get(end_cand, "")
                # milliseconds variants
                if start_cand.endswith("_ms"):
                    s_sec = float(s_raw)/1000.0 if s_raw != "" else None
                else:
                    s_sec = parse_time(s_raw) if s_raw != "" else None
                if end_cand.endswith("_ms"):
                    e_sec = float(e_raw)/1000.0 if e_raw != "" else None
                else:
                    e_sec = parse_time(e_raw) if e_raw != "" else None
            else:
                t = parse_time(r.get(t_cand, ""))
                pre = parse_time(r.get(pre_cand, "0")) or 0.0
                post = parse_time(r.get(post_cand, "0")) or 0.0
                s_sec = max(0.0, t - pre)
                e_sec = max(s_sec, t + post)
            if s_sec is None or e_sec is None or e_sec <= s_sec:
                # skip bad rows
                continue
            r2["t0"] = f"{s_sec:.3f}"
            r2["t1"] = f"{e_sec:.3f}"
            writer.writerow(r2)

if __name__ == "__main__":
    main()
