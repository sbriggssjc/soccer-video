#!/usr/bin/env python3
"""Repository indexing utility for soccer-video project.

This tool inspects a repository tree and emits a collection of machine-readable
artifacts that summarise files, folders and detected relationships.  The
implementation favours best-effort heuristics that are reasonably fast on large
trees while still producing rich metadata for downstream analysis.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fnmatch
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import openpyxl  # type: ignore
except ImportError:  # noqa: F401
    openpyxl = None  # type: ignore

try:
    from Levenshtein import distance as levenshtein_distance  # type: ignore
except ImportError:  # noqa: F401
    def levenshtein_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current_row = [i]
            for j, cb in enumerate(b, 1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]



LOGGER = logging.getLogger("index_repo")


DEFAULT_EXCLUDES = [
    "out/**",
    "runs/**",
    ".git/**",
    ".venv/**",
    "node_modules/**",
    "__pycache__/**",
    "**/*.mp4",
    "**/*.mov",
    "**/*.mkv",
    "**/*.avi",
    "**/*.webm",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.tif",
    "**/*.psd",
    "**/*.zip",
    "**/*.7z",
    "**/*.rar",
    "**/*.iso",
]


TEXT_EXTENSIONS = {
    ".py",
    ".ps1",
    ".psm1",
    ".psd1",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".md",
    ".txt",
    ".csv",
    ".sh",
    ".bat",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".ipynb",
}


SCRIPT_EXTENSIONS = {
    ".py": "script_py",
    ".ps1": "script_ps1",
    ".psm1": "script_ps1",
    ".bat": "script_bat",
    ".sh": "script_sh",
}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".psd",
    ".bmp",
    ".gif",
    ".svg",
}
FONT_EXTENSIONS = {".otf", ".ttf", ".woff", ".woff2"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"}
DOC_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}


BRAND_KEYWORDS = {"brand", "watermark", "badge", "overlay", "end_card"}


@dataclass
class FileRecord:
    repo_relpath: str
    abspath: str
    size_bytes: int
    ext: str
    file_type: str
    lines: Optional[int] = None
    first_line: Optional[str] = None
    sha1: Optional[str] = None
    git_last_commit: Optional[str] = None
    git_created_commit: Optional[str] = None
    last_modified_fs: Optional[str] = None
    category_guess: Optional[str] = None
    language_guess: Optional[str] = None
    declared_tools: List[str] = field(default_factory=list)
    script_role: Optional[str] = None
    references_out_paths: List[str] = field(default_factory=list)
    references_in_paths: List[str] = field(default_factory=list)
    references_brand_assets: List[str] = field(default_factory=list)
    invokes_external_cmds: List[str] = field(default_factory=list)
    imports_internal_modules: List[str] = field(default_factory=list)
    imports_external_modules: List[str] = field(default_factory=list)
    parameters_detected: List[str] = field(default_factory=list)
    used_by_signals: List[str] = field(default_factory=list)
    suspect_orphan: bool = False
    notes: List[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a repository tree")
    parser.add_argument("--root", required=True, help="Root folder to scan")
    parser.add_argument(
        "--include-outputs",
        action="store_true",
        help="Include typical output/media folders",
    )
    parser.add_argument(
        "--globs",
        default="**/*",
        help="Semicolon separated glob patterns to include",
    )
    parser.add_argument(
        "--exclude-globs",
        default="",
        help="Semicolon separated glob patterns to exclude",
    )
    parser.add_argument(
        "--max-hash-mb",
        type=float,
        default=25.0,
        help="Maximum file size (MB) to hash by default",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Worker threads for scanning",
    )
    return parser.parse_args()


def configure_logging(out_dir: Path) -> None:
    log_file = out_dir / "index_repo.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def compile_patterns(patterns: Sequence[str]) -> List[str]:
    return [p.strip() for p in patterns if p and p.strip()]


def should_exclude(path: Path, patterns: Sequence[str], root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    for pattern in patterns:
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


def is_binary_string(data: bytes) -> bool:
    if b"\0" in data:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    translated = data.translate(None, text_chars)
    return bool(translated)


def read_text_file(path: Path, max_bytes: int = 2_000_000) -> Tuple[Optional[str], Optional[str]]:
    try:
        with path.open("rb") as fh:
            chunk = fh.read(max_bytes)
            if is_binary_string(chunk):
                return None, None
            text = chunk.decode("utf-8", errors="ignore")
            remainder = fh.read(1)
            if remainder:
                text += "\n..."
            return text, text.splitlines()[0].strip() if text.splitlines() else ""
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed reading text for %s: %s", path, exc)
        return None, None


def compute_sha1(path: Path, max_bytes: int) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        hasher = hashlib.sha1()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed hashing %s: %s", path, exc)
        return None


def compute_sha1_streaming(path: Path) -> Optional[str]:
    try:
        hasher = hashlib.sha1()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed hashing %s: %s", path, exc)
        return None


def guess_file_type(ext: str, path: Path) -> str:
    if ext in SCRIPT_EXTENSIONS:
        return SCRIPT_EXTENSIONS[ext]
    if ext in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf"}:
        return "config"
    if ext in IMAGE_EXTENSIONS:
        return "asset_image"
    if ext in FONT_EXTENSIONS:
        return "asset_font"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in DOC_EXTENSIONS:
        return "doc"
    if ext == ".md":
        return "markdown"
    if ext == ".ipynb":
        return "notebook"
    return "other"


def guess_category(path: Path, file_type: str) -> str:
    rel = path.as_posix().lower()
    if any(part in rel for part in ["/out/", "/runs/", "cache", "tmp", "render"]):
        return "generated_output"
    if "brand" in rel:
        return "asset_brand"
    if "overlay" in rel or "badge" in rel:
        return "asset_overlay"
    if any(ext in rel for ext in ["_index", "clip", "clips", "playlist"]):
        return "clip_index"
    if "log" in rel or rel.endswith(".log"):
        return "log"
    if file_type in {"script_py", "script_ps1", "script_bat", "script_sh"}:
        return "source_code"
    if file_type in {"config", "markdown"}:
        return "config"
    return "other"


def guess_language(ext: str) -> str:
    mapping = {
        ".py": "python",
        ".ps1": "powershell",
        ".psm1": "powershell",
        ".bat": "batch",
        ".sh": "bash",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".md": "markdown",
        ".txt": "text",
        ".csv": "csv",
        ".ipynb": "notebook",
    }
    return mapping.get(ext, "none")


DECLARED_TOOL_PATTERNS = {
    "ffmpeg": re.compile(r"ffmpeg", re.IGNORECASE),
    "magick": re.compile(r"magick", re.IGNORECASE),
    "opencv": re.compile(r"cv2|opencv", re.IGNORECASE),
    "yolo": re.compile(r"yolo", re.IGNORECASE),
    "python": re.compile(r"python\b", re.IGNORECASE),
    "imagemagick": re.compile(r"convert\b|mogrify", re.IGNORECASE),
}


EXTERNAL_CMD_PATTERNS = [
    re.compile(r"ffmpeg", re.IGNORECASE),
    re.compile(r"magick", re.IGNORECASE),
    re.compile(r"yt-dlp", re.IGNORECASE),
    re.compile(r"subprocess\.(run|call|popen)", re.IGNORECASE),
    re.compile(r"Invoke-\w+", re.IGNORECASE),
    re.compile(r"Start-Process", re.IGNORECASE),
]


OUT_PATH_PATTERN = re.compile(r"out[\\/][^\s'\"]+")
IN_PATH_PATTERN = re.compile(r"(?:runs|config|brand|pipeline|recipes)[\\/][^\s'\"]+")
ASSET_PATTERN = re.compile(r"brand[\\/][^\s'\"]+")
PATH_REFERENCE_PATTERN = re.compile(r"([\w./\\-]+\.(?:py|ps1|json|yaml|yml|cfg|ini|txt|csv|mp4|mov|png|jpg))")


SCRIPT_ROLE_HINTS = {
    "clipper": ["clip", "highlight"],
    "tracker": ["track", "tracker"],
    "stabilizer": ["stabil", "steady"],
    "branding": ["brand", "overlay", "watermark"],
    "packager": ["package", "render", "export"],
    "ingestor": ["ingest", "import", "download"],
    "utility": ["util", "helper", "common"],
}


def detect_script_role(path: Path, text: Optional[str]) -> Optional[str]:
    lowered = path.name.lower()
    candidates = lowered
    if text:
        candidates += " " + text[:500].lower()
    for role, hints in SCRIPT_ROLE_HINTS.items():
        if any(hint in candidates for hint in hints):
            return role
    return None


def detect_declared_tools(text: Optional[str]) -> List[str]:
    if not text:
        return []
    found = []
    for name, pattern in DECLARED_TOOL_PATTERNS.items():
        if pattern.search(text):
            found.append(name)
    return sorted(set(found))


def detect_external_cmds(text: Optional[str]) -> List[str]:
    if not text:
        return []
    found: Set[str] = set()
    for pattern in EXTERNAL_CMD_PATTERNS:
        match = pattern.findall(text)
        if match:
            if isinstance(match, list):
                if isinstance(match[0], tuple):
                    for tup in match:
                        for elem in tup:
                            if elem:
                                found.add(elem if isinstance(elem, str) else str(elem))
                else:
                    for elem in match:
                        if isinstance(elem, tuple):
                            for inner in elem:
                                if inner:
                                    found.add(str(inner))
                        else:
                            found.add(str(elem))
    return sorted(found)


def detect_references(text: Optional[str], pattern: re.Pattern[str]) -> List[str]:
    if not text:
        return []
    matches = pattern.findall(text)
    cleaned = sorted({m.replace("\\", "/") for m in matches})
    return cleaned


def detect_parameters(path: Path, text: Optional[str]) -> List[str]:
    if not text:
        return []
    params: Set[str] = set()
    if path.suffix == ".py":
        params.update(re.findall(r"--([A-Za-z0-9_-]+)", text))
    elif path.suffix == ".ps1" or path.suffix == ".psm1":
        param_block = re.findall(r"param\s*\(([^)]*)\)", text, flags=re.IGNORECASE | re.DOTALL)
        for block in param_block:
            params.update(re.findall(r"\$([A-Za-z0-9_]+)", block))
    return sorted(params)


def detect_notes(path: Path, text: Optional[str]) -> List[str]:
    notes: List[str] = []
    if text and re.search(r"C:\\Users\\", text):
        notes.append("hardcoded path found")
    if path.name.endswith(".bak"):
        notes.append("backup file")
    return notes


def collect_python_imports(path: Path, text: Optional[str], internal_modules: Set[str]) -> Tuple[List[str], List[str]]:
    if path.suffix != ".py" or not text:
        return [], []
    try:
        import ast

        tree = ast.parse(text)
    except Exception:  # noqa: BLE001
        return [], []
    internal: Set[str] = set()
    external: Set[str] = set()
    stdlib = getattr(sys, "stdlib_module_names", set())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                if node.module:
                    names = [node.module.split(".")[0]]
                else:
                    continue
            for name in names:
                if name in internal_modules:
                    internal.add(name)
                elif name in stdlib:
                    continue
                else:
                    external.add(name)
    return sorted(internal), sorted(external)


def gather_internal_modules(root: Path) -> Set[str]:
    internal: Set[str] = set()
    for path in root.iterdir():
        if path.is_dir() and (path / "__init__.py").exists():
            internal.add(path.name)
        elif path.suffix == ".py":
            internal.add(path.stem)
    return internal


def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def git_last_commit(root: Path, relpath: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "-1",
                "--pretty=format:%H|%ad|%an",
                "--date=iso",
                "--",
                relpath,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip()
        if result.returncode == 0 and output:
            return output
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("git last commit failed for %s: %s", relpath, exc)
    return None


def git_created_commit(root: Path, relpath: str) -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "log",
                "--diff-filter=A",
                "--pretty=format:%H|%ad|%an",
                "--date=iso",
                "--",
                relpath,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout.strip().splitlines()
        if result.returncode == 0 and output:
            return output[-1]
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("git created commit failed for %s: %s", relpath, exc)
    return None


def format_iso_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def scan_file(
    path: Path,
    root: Path,
    include_outputs: bool,
    max_hash_bytes: int,
    git_enabled: bool,
    internal_modules: Set[str],
) -> Tuple[FileRecord, Optional[str]]:
    repo_relpath = path.relative_to(root).as_posix()
    size = path.stat().st_size
    ext = path.suffix.lower()
    file_type = guess_file_type(ext, path)
    category = guess_category(path.relative_to(root), file_type)
    language = guess_language(ext)

    text_content, first_line = read_text_file(path) if ext in TEXT_EXTENSIONS else (None, None)
    if not text_content and file_type in {"config", "markdown", "script_py", "script_ps1"}:
        text_content, first_line = read_text_file(path)

    lines = None
    if text_content:
        lines = len([line for line in text_content.splitlines() if line and not line.endswith("...")])

    hash_value = None
    if size <= max_hash_bytes:
        hash_value = compute_sha1(path, max_hash_bytes)
    elif include_outputs:
        hash_value = compute_sha1_streaming(path)

    last_commit = git_last_commit(root, repo_relpath) if git_enabled else None
    created_commit = git_created_commit(root, repo_relpath) if git_enabled else None

    mtime_iso = format_iso_timestamp(path.stat().st_mtime)

    declared = detect_declared_tools(text_content)
    script_role = detect_script_role(path, text_content)
    out_refs = detect_references(text_content, OUT_PATH_PATTERN)
    in_refs = detect_references(text_content, IN_PATH_PATTERN)
    brand_refs = detect_references(text_content, ASSET_PATTERN)
    external_cmds = detect_external_cmds(text_content)
    params = detect_parameters(path, text_content)
    notes = detect_notes(path, text_content)
    path_refs = detect_references(text_content, PATH_REFERENCE_PATTERN)

    imports_internal: List[str] = []
    imports_external: List[str] = []
    if path.suffix == ".py":
        imports_internal, imports_external = collect_python_imports(path, text_content, internal_modules)

    record = FileRecord(
        repo_relpath=repo_relpath,
        abspath=str(path.resolve()),
        size_bytes=size,
        ext=ext or "",
        file_type=file_type,
        lines=lines,
        first_line=first_line,
        sha1=hash_value,
        git_last_commit=last_commit,
        git_created_commit=created_commit,
        last_modified_fs=mtime_iso,
        category_guess=category,
        language_guess=language,
        declared_tools=declared,
        script_role=script_role,
        references_out_paths=out_refs,
        references_in_paths=in_refs,
        references_brand_assets=brand_refs,
        invokes_external_cmds=external_cmds,
        imports_internal_modules=imports_internal,
        imports_external_modules=imports_external,
        parameters_detected=params,
        notes=notes,
    )

    return record, text_content


def build_folder_rollup(records: Sequence[FileRecord]) -> List[Dict[str, object]]:
    folders: Dict[str, Dict[str, object]] = {}
    folder_files: Dict[str, List[FileRecord]] = defaultdict(list)
    for record in records:
        folder = str(Path(record.repo_relpath).parent.as_posix()) or "."
        folder_files[folder].append(record)

    for folder, files in folder_files.items():
        counts = Counter(file.file_type for file in files)
        categories = Counter(file.category_guess for file in files)
        total_size = sum(file.size_bytes for file in files)
        largest = sorted(files, key=lambda f: f.size_bytes, reverse=True)[:5]
        folders[folder] = {
            "folder": folder,
            "files_count": len(files),
            "total_size_bytes": total_size,
            "file_type_counts": dict(counts),
            "category_counts": dict(categories),
            "largest_files_top5": ";".join(
                f"{f.repo_relpath}:{f.size_bytes / (1024 * 1024):.1f}MB" for f in largest
            ),
            "last_git_activity_iso": most_recent_git(files),
        }
    return list(folders.values())


def most_recent_git(files: Sequence[FileRecord]) -> Optional[str]:
    times: List[str] = []
    for f in files:
        if f.git_last_commit:
            parts = f.git_last_commit.split("|")
            if len(parts) >= 2:
                times.append(parts[1])
    if not times:
        return None
    return max(times)


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_repo_csv(path: Path, records: Sequence[FileRecord]) -> None:
    fieldnames = [
        "repo_relpath",
        "abspath",
        "size_bytes",
        "ext",
        "file_type",
        "lines",
        "first_line",
        "sha1",
        "git_last_commit",
        "git_created_commit",
        "last_modified_fs",
        "category_guess",
        "language_guess",
        "declared_tools",
        "script_role",
        "references_out_paths",
        "references_in_paths",
        "references_brand_assets",
        "invokes_external_cmds",
        "imports_internal_modules",
        "imports_external_modules",
        "parameters_detected",
        "used_by_signals",
        "suspect_orphan",
        "notes",
    ]

    rows = []
    for record in records:
        row = asdict(record)
        row["declared_tools"] = ";".join(record.declared_tools)
        row["references_out_paths"] = ";".join(record.references_out_paths)
        row["references_in_paths"] = ";".join(record.references_in_paths)
        row["references_brand_assets"] = ";".join(record.references_brand_assets)
        row["invokes_external_cmds"] = ";".join(record.invokes_external_cmds)
        row["imports_internal_modules"] = ";".join(record.imports_internal_modules)
        row["imports_external_modules"] = ";".join(record.imports_external_modules)
        row["parameters_detected"] = ";".join(record.parameters_detected)
        row["used_by_signals"] = ";".join(record.used_by_signals)
        row["notes"] = ";".join(record.notes)
        rows.append(row)

    write_csv(path, rows, fieldnames)


def write_repo_xlsx(path: Path, records: Sequence[FileRecord]) -> None:
    if openpyxl is None:
        LOGGER.warning("openpyxl not available; skipping XLSX export")
        return
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "repo_index"
    headers = [
        "repo_relpath",
        "abspath",
        "size_bytes",
        "ext",
        "file_type",
        "lines",
        "first_line",
        "sha1",
        "git_last_commit",
        "git_created_commit",
        "last_modified_fs",
        "category_guess",
        "language_guess",
        "declared_tools",
        "script_role",
        "references_out_paths",
        "references_in_paths",
        "references_brand_assets",
        "invokes_external_cmds",
        "imports_internal_modules",
        "imports_external_modules",
        "parameters_detected",
        "used_by_signals",
        "suspect_orphan",
        "notes",
    ]
    sheet.append(headers)
    for record in records:
        sheet.append([
            record.repo_relpath,
            record.abspath,
            record.size_bytes,
            record.ext,
            record.file_type,
            record.lines,
            record.first_line,
            record.sha1,
            record.git_last_commit,
            record.git_created_commit,
            record.last_modified_fs,
            record.category_guess,
            record.language_guess,
            ",".join(record.declared_tools),
            record.script_role,
            ",".join(record.references_out_paths),
            ",".join(record.references_in_paths),
            ",".join(record.references_brand_assets),
            ",".join(record.invokes_external_cmds),
            ",".join(record.imports_internal_modules),
            ",".join(record.imports_external_modules),
            ",".join(record.parameters_detected),
            ",".join(record.used_by_signals),
            record.suspect_orphan,
            ",".join(record.notes),
        ])

    sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


def compute_duplicates(records: Sequence[FileRecord]) -> List[Dict[str, object]]:
    clusters: List[Dict[str, object]] = []
    cluster_id = 1
    exact_groups: Dict[Tuple[str, int], List[FileRecord]] = defaultdict(list)
    for record in records:
        if record.sha1:
            exact_groups[(record.sha1, record.size_bytes)].append(record)

    for key, group in exact_groups.items():
        if len(group) <= 1:
            continue
        total_size = sum(file.size_bytes for file in group)
        clusters.append(
            {
                "cluster_id": f"exact-{cluster_id}",
                "representative_file": group[0].repo_relpath,
                "members": [f.repo_relpath for f in group],
                "total_size_bytes": total_size,
                "type": "exact",
            }
        )
        cluster_id += 1

    # Near-duplicates: normalize names (_BRAND/_POLISHED/_POST/_CLEAN etc.) and compare size ±2% or short Levenshtein

    def _normalize_name(p: str) -> str:
        stem = Path(p).stem.lower()
        stem = re.sub(r'(__WITH_OPENER|_BRAND|_POLISHED|_POST|_CLEAN.*)$', '', stem)
        stem = re.sub(r'(\bshot\b|\bclip\b|\breel\b)', '', stem)
        stem = re.sub(r'[^a-z0-9]+', '', stem)
        return stem

    # Bucket by normalized name to keep comparisons tractable
    buckets: Dict[str, List[FileRecord]] = defaultdict(list)
    for r in records:
        buckets[_normalize_name(r.repo_relpath)].append(r)

    for group in buckets.values():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda r: r.size_bytes)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if max(a.size_bytes, b.size_bytes) == 0:
                    size_close = True
                else:
                    diff = abs(a.size_bytes - b.size_bytes) / max(a.size_bytes, b.size_bytes)
                    size_close = diff <= 0.02
                lev = levenshtein_distance(Path(a.repo_relpath).name, Path(b.repo_relpath).name)
                if size_close or lev <= 3:
                    clusters.append(
                        {
                            "cluster_id": f"near-{cluster_id}",
                            "representative_file": a.repo_relpath,
                            "members": [a.repo_relpath, b.repo_relpath],
                            "total_size_bytes": a.size_bytes + b.size_bytes,
                            "type": "near",
                        }
                    )
                    cluster_id += 1
    return clusters


def write_duplicates_csv(path: Path, clusters: Sequence[Dict[str, object]]) -> None:
    fieldnames = ["cluster_id", "type", "representative_file", "members", "total_size_bytes"]
    rows = []
    for cluster in clusters:
        rows.append(
            {
                "cluster_id": cluster["cluster_id"],
                "type": cluster["type"],
                "representative_file": cluster["representative_file"],
                "members": ";".join(cluster["members"]),
                "total_size_bytes": cluster["total_size_bytes"],
            }
        )
    write_csv(path, rows, fieldnames)


def detect_used_by(records: Sequence[FileRecord], texts: Dict[str, str]) -> None:
    # Build simple index by basename
    name_to_paths: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        name_to_paths[Path(record.repo_relpath).name.lower()].append(record.repo_relpath)

    for record in records:
        used_by: Set[str] = set()
        target_path = Path(record.repo_relpath)
        if len(target_path.stem) <= 3:
            record.used_by_signals = []
            continue
        target_name = target_path.name.lower()
        for source_path, text in texts.items():
            if target_name in text.lower() and source_path != record.repo_relpath:
                used_by.add(source_path)
        record.used_by_signals = sorted(used_by)


def detect_orphans(records: Sequence[FileRecord], texts: Dict[str, str]) -> List[Dict[str, object]]:
    name_to_refs: Dict[str, Set[str]] = defaultdict(set)
    for source_path, text in texts.items():
        for match in PATH_REFERENCE_PATTERN.findall(text):
            name_to_refs[match.lower()].add(source_path)

    orphan_rows: List[Dict[str, object]] = []
    for record in records:
        reasons: List[str] = []
        name = Path(record.repo_relpath).name.lower()
        if not record.used_by_signals and name not in name_to_refs:
            reasons.append("no references detected")
        if record.git_last_commit:
            parts = record.git_last_commit.split("|")
            if len(parts) >= 2:
                try:
                    commit_time = datetime.fromisoformat(parts[1].split(" ")[0])
                    if (datetime.now() - commit_time).days > 180:
                        reasons.append("no recent git activity")
                except ValueError:
                    pass
        if record.category_guess not in {"source_code", "config"}:
            reasons.append("non-core category")
        if reasons and record.file_type in {"script_py", "script_ps1"}:
            record.suspect_orphan = True
            record.notes.extend(reasons)
            orphan_rows.append(
                {
                    "repo_relpath": record.repo_relpath,
                    "reasons": ";".join(reasons),
                }
            )
    return orphan_rows


def build_process_graph(records: Sequence[FileRecord]) -> Tuple[str, List[str]]:
    nodes: Set[str] = set()
    edges: Set[Tuple[str, str, str]] = set()
    for record in records:
        label = record.repo_relpath
        nodes.add(label)
        if record.file_type in {"script_py", "script_ps1", "script_sh", "script_bat"}:
            for ref in record.references_out_paths:
                edges.add((label, ref, "writes"))
            for ref in record.references_in_paths:
                edges.add((label, ref, "reads"))
            for ref in record.references_brand_assets:
                edges.add((label, ref, "uses_brand"))
            for used in record.used_by_signals:
                edges.add((used, label, "calls"))

    dot_lines = ["digraph process {", "  rankdir=LR;"]
    for node in sorted(nodes):
        dot_lines.append(f"  \"{node}\";")
    for src, dst, label in sorted(edges):
        dot_lines.append(f'  "{src}" -> "{dst}" [label="{label}"];')
    dot_lines.append("}")
    return "\n".join(dot_lines), sorted(nodes)


def render_graph_png(dot_path: Path, png_path: Path) -> None:
    try:
        import graphviz  # type: ignore

        graph = graphviz.Source(dot_path.read_text(encoding="utf-8"))
        graph.format = "png"
        graph.render(png_path.with_suffix(""), cleanup=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Graphviz rendering unavailable: %s", exc)


def generate_summary(
    summary_path: Path,
    records: Sequence[FileRecord],
    duplicates: Sequence[Dict[str, object]],
    folder_rollup: Sequence[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Repository Index Summary")
    lines.append("")

    sorted_by_size = sorted(records, key=lambda r: r.size_bytes, reverse=True)[:20]
    lines.append("## Top 20 Largest Files")
    lines.append("")
    for record in sorted_by_size:
        lines.append(
            f"- {record.repo_relpath} — {record.size_bytes / (1024 * 1024):.2f} MB"
        )
    lines.append("")

    lines.append("## Top 20 Noisiest Folders")
    lines.append("")
    largest_folders = sorted(
        folder_rollup, key=lambda r: r["total_size_bytes"], reverse=True
    )[:20]
    for folder in largest_folders:
        lines.append(
            f"- {folder['folder']} — {folder['total_size_bytes'] / (1024 * 1024):.2f} MB "
            f"across {folder['files_count']} files"
        )
    lines.append("")

    lines.append("## Duplicate Clusters")
    lines.append("")
    for cluster in duplicates[:20]:
        size_mb = cluster["total_size_bytes"] / (1024 * 1024)
        lines.append(
            f"- {cluster['cluster_id']} ({cluster['type']}): {cluster['representative_file']} — "
            f"{size_mb:.2f} MB across {len(cluster['members'])} files"
        )
    if not duplicates:
        lines.append("- None detected")
    lines.append("")

    suspected_outputs = [
        r
        for r in records
        if r.category_guess == "generated_output" or "out/" in r.repo_relpath
    ]
    lines.append("## Suspected Outputs")
    lines.append("")
    for record in suspected_outputs[:20]:
        lines.append(
            f"- {record.repo_relpath} ({record.file_type}, {record.size_bytes / (1024 * 1024):.1f} MB)"
        )
    if not suspected_outputs:
        lines.append("- None detected")
    lines.append("")

    dead_scripts = [r for r in records if r.suspect_orphan]
    lines.append("## Suspected Dead Scripts")
    lines.append("")
    for record in dead_scripts[:20]:
        lines.append(f"- {record.repo_relpath} — reasons: {', '.join(record.notes or ['see orphan signals'])}")
    if not dead_scripts:
        lines.append("- None detected")
    lines.append("")

    lines.append("## Next Cleanup Moves")
    lines.append("")
    if duplicates:
        lines.append("- Consolidate duplicate clusters to reclaim storage.")
    if suspected_outputs:
        lines.append("- Move or purge generated outputs from version control.")
    if dead_scripts:
        lines.append("- Review orphaned scripts for retirement or documentation.")
    lines.append("- Review top noisy folders for consolidation opportunities.")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def write_folder_rollup(path: Path, rollup: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "folder",
        "files_count",
        "total_size_bytes",
        "file_type_counts",
        "category_counts",
        "largest_files_top5",
        "last_git_activity_iso",
    ]
    rows = []
    for entry in rollup:
        rows.append(
            {
                "folder": entry["folder"],
                "files_count": entry["files_count"],
                "total_size_bytes": entry["total_size_bytes"],
                "file_type_counts": json.dumps(entry["file_type_counts"]),
                "category_counts": json.dumps(entry["category_counts"]),
                "largest_files_top5": entry["largest_files_top5"],
                "last_git_activity_iso": entry["last_git_activity_iso"],
            }
        )
    write_csv(path, rows, fieldnames)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(out_dir)

    LOGGER.info("Starting repository index for %s", root)

    include_patterns = compile_patterns(args.globs.split(";"))
    exclude_patterns = compile_patterns(args.exclude_globs.split(";"))

    effective_excludes = list(DEFAULT_EXCLUDES)
    if args.include_outputs:
        strip_prefixes = ("out/**", "runs/**")
        strip_exts = (
            "**/*.mp4",
            "**/*.mov",
            "**/*.mkv",
            "**/*.avi",
            "**/*.webm",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.tif",
            "**/*.psd",
        )
        effective_excludes = [
            p
            for p in effective_excludes
            if (p not in strip_prefixes and p not in strip_exts)
        ]
    effective_excludes.extend(exclude_patterns)

    files_to_scan: List[Path] = []
    for pattern in include_patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if should_exclude(path, effective_excludes, root):
                continue
            files_to_scan.append(path)

    files_to_scan = sorted(set(files_to_scan))
    LOGGER.info("Discovered %d files to scan", len(files_to_scan))

    git_enabled = is_git_repo(root)
    max_hash_bytes = int(args.max_hash_mb * 1024 * 1024)
    internal_modules = gather_internal_modules(root)

    records: List[FileRecord] = []
    texts: Dict[str, str] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(
                scan_file,
                path,
                root,
                args.include_outputs,
                max_hash_bytes,
                git_enabled,
                internal_modules,
            ): path
            for path in files_to_scan
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                record, text_content = future.result()
                records.append(record)
                if text_content:
                    texts[record.repo_relpath] = text_content
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed processing %s: %s", path, exc)

    records.sort(key=lambda r: r.repo_relpath)

    detect_used_by(records, texts)
    orphan_rows = detect_orphans(records, texts)

    rollup = build_folder_rollup(records)
    duplicates = compute_duplicates(records)

    repo_index_json = out_dir / "repo_index.json"
    repo_index_csv = out_dir / "repo_index.csv"
    repo_index_xlsx = out_dir / "repo_index.xlsx"
    folder_rollup_csv = out_dir / "folder_rollup.csv"
    duplicates_csv = out_dir / "duplicates.csv"
    process_graph_dot = out_dir / "process_graph.dot"
    process_graph_png = out_dir / "process_graph.png"
    orphan_csv = out_dir / "orphan_signals.csv"
    summary_md = out_dir / "SUMMARY.md"

    write_json(repo_index_json, [asdict(r) for r in records])
    write_repo_csv(repo_index_csv, records)
    write_repo_xlsx(repo_index_xlsx, records)
    write_folder_rollup(folder_rollup_csv, rollup)
    write_duplicates_csv(duplicates_csv, duplicates)
    write_csv(orphan_csv, orphan_rows, ["repo_relpath", "reasons"])

    dot_content, _ = build_process_graph(records)
    process_graph_dot.write_text(dot_content, encoding="utf-8")
    render_graph_png(process_graph_dot, process_graph_png)

    generate_summary(summary_md, records, duplicates, rollup)

    LOGGER.info("Repository index completed. Outputs:")
    for path in [
        repo_index_json,
        repo_index_csv,
        repo_index_xlsx,
        folder_rollup_csv,
        duplicates_csv,
        process_graph_dot,
        process_graph_png,
        orphan_csv,
        summary_md,
    ]:
        LOGGER.info("  %s", path)


if __name__ == "__main__":
    main()

