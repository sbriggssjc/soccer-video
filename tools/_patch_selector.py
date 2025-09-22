import re, pathlib, sys
p = pathlib.Path(r".\tools\selector.py")
t = p.read_text(encoding="utf-8")

# Normalize tabs → 4 spaces to avoid mix issues
t = t.replace("\t", "    ")

# Remove any stray bare guard line that might exist
t = re.sub(r'^\s*getattr\(self,\s*\'in_play_mask\',\s*None\)\s+is\s+not\s+None:\s*\r?\n', '', t, flags=re.M)

# Replace the single coverage line with a guarded block
pat = re.compile(r'^([ ]*)_inplay\s*=\s*self\.in_play_mask\.coverage\(t0,\s*t1\)\s*$', re.M)
def repl(m):
    indent = m.group(1)
    return (
        f"{indent}_inplay = 1.0\n"
        f"{indent}if getattr(self, 'in_play_mask', None) is not None:\n"
        f"{indent}    try:\n"
        f"{indent}        _inplay = self.in_play_mask.coverage(t0, t1)\n"
        f"{indent}    except Exception:\n"
        f"{indent}        _inplay = 1.0\n"
    )
t, n = pat.subn(repl, t)
if n == 0:
    sys.exit("Did not find the coverage line to patch.")

p.write_text(t, encoding="utf-8")
print("Patched selector.py (guarded in_play coverage).")

