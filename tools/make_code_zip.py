"""Build dist/gemma-paper-code.zip for the private Kaggle dataset `gemma-paper-code` (reproducible).

Contents: harness/ (code + skills), configs/, kaggle/*.py, probe/*.py (no results), tasks_postcutoff/
(code + out/: tasks, snapshots, wheels, metadata). Never included: private notes, secrets, probe
results, competition data, the Modal wheelhouse, caches. Paths use '/' (Kaggle extracts on Linux;
PowerShell's Compress-Archive writes '\\', which broke an earlier upload).
Prints a VERSION marker (sha256 of the file list + contents) that cell 1/K3 can print on Kaggle,
to confirm the uploaded version is the latest.

    python -I tools/make_code_zip.py
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = [
    ("harness", "**/*"),
    ("configs", "*.json"),
    ("kaggle", "*.py"),
    ("probe", "*.py"),
    ("tasks_postcutoff", "*.py"),
    ("tasks_postcutoff", "*.md"),
    ("tasks_postcutoff/out", "**/*"),
]
EXCLUDE_PARTS = {"__pycache__", ".git", "results"}
EXCLUDE_SUFFIX = {".pyc"}


def files() -> list[Path]:
    out = set()
    for d, pat in INCLUDE:
        for p in (ROOT / d).glob(pat):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & EXCLUDE_PARTS) and p.suffix not in EXCLUDE_SUFFIX:
                out.add(p)
    return sorted(out)


def main() -> None:
    fs = files()
    h = hashlib.sha256()
    dst = ROOT / "dist" / "gemma-paper-code.zip"
    dst.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for p in fs:
            rel = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            h.update(rel.encode() + b"\0" + data)
            zi = zipfile.ZipInfo(rel, date_time=(2026, 1, 1, 0, 0, 0))  # fixed time -> reproducible zip
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
        version = h.hexdigest()[:12]
        z.writestr(zipfile.ZipInfo("CODE_VERSION.txt", date_time=(2026, 1, 1, 0, 0, 0)), version + "\n")
    print(f"{len(fs)} files, {dst.stat().st_size / 1e6:.1f} MB -> {dst}\nCODE_VERSION {version}")


if __name__ == "__main__":
    main()
