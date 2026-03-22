"""Move legacy iteration/debug scripts from repo root to _archive_scripts/.

Keeps: tools/, config/, out/, in/, brand/, src/, tests/, docs/, scripts/,
       pipeline/, recipes/, configs/, examples/, recordings/, runs/, legacy/,
       finals/, clips/, cleanup_reports/, soccer_highlights/,
       .git*, pyproject.toml, setup.cfg, requirements*, README*, config.yaml,
       roster*, *.ps1 (PowerShell pipeline scripts), *.csv, *.json,
       AtomicClips.*, *.pt (YOLO models), and a few other infra files.
"""
import os, shutil, re

ROOT = r"D:\Projects\soccer-video"
ARCHIVE = os.path.join(ROOT, "_archive_scripts")
os.makedirs(ARCHIVE, exist_ok=True)

# Directories to never touch
KEEP_DIRS = {
    ".git", ".github", ".claude", "tools", "config", "configs", "out", "in",
    "brand", "src", "tests", "docs", "scripts", "pipeline", "recipes",
    "examples", "recordings", "runs", "legacy", "finals", "clips",
    "cleanup_reports", "soccer_highlights", "__pycache__",
    "_archive_scripts",
}

# File patterns to KEEP (anchored to basename)
KEEP_FILE_PATTERNS = [
    r"^\.git",                # .gitignore, .gitattributes, etc.
    r"^pyproject\.toml$",
    r"^setup\.cfg$",
    r"^requirements",
    r"^README",
    r"^config\.yaml$",
    r"^roster",
    r"\.ps1$",                # PowerShell pipeline scripts
    r"^AtomicClips\.",        # Manifest/status files
    r"\.pt$",                 # YOLO models
    r"^atomic_clips_index",   # Index files
    r"^05_make_social_reel",  # Shell scripts
    r"^vfCore\.ps1$",
]

# Files that are clearly iteration artifacts (move these)
MOVE_PATTERNS = [
    r"^_",                    # All underscore-prefixed files
    r"^analyze_",             # Analysis scripts
    r"^extract_",             # Extract scripts
]

moved = 0
kept = 0
skipped_dirs = 0

for entry in sorted(os.listdir(ROOT)):
    full = os.path.join(ROOT, entry)

    # Skip directories we want to keep
    if os.path.isdir(full):
        if entry in KEEP_DIRS:
            skipped_dirs += 1
            continue
        # Move debug/temp directories
        if entry.startswith("_"):
            dest = os.path.join(ARCHIVE, entry)
            shutil.move(full, dest)
            moved += 1
            print(f"  MOVE DIR: {entry}")
            continue
        skipped_dirs += 1
        continue

    # Check keep patterns first
    dominated = False
    for pat in KEEP_FILE_PATTERNS:
        if re.search(pat, entry):
            dominated = True
            break
    if dominated:
        kept += 1
        continue

    # Check move patterns
    should_move = False
    for pat in MOVE_PATTERNS:
        if re.search(pat, entry):
            should_move = True
            break

    if should_move:
        dest = os.path.join(ARCHIVE, entry)
        shutil.move(full, dest)
        moved += 1
        print(f"  MOVE: {entry}")
    else:
        kept += 1

print(f"\nDone. Kept {kept} files, skipped {skipped_dirs} dirs, moved {moved} items to _archive_scripts/")
