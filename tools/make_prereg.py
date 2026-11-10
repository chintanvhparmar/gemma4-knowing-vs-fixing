"""Freeze the pre-registration: SHA-256 of every file that defines the study, plus one overall hash.

    python -I tools/make_prereg.py            -> prereg/manifest.json + prints the OVERALL hash to post

The overall hash = sha256 over the sorted "path\\0sha256\\n" lines, so anyone can recompute it from the
published files. Covers: the pre-registration text, task definitions, task selection, conditions/prompts
generator, skills, runner, grading/overlay code, the web-search relay, and the Modal replica definition.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    "prereg/PREREGISTRATION.md",
    "configs/control_tasks.json",
    "configs/calibration_tasks.json",
    "tasks_postcutoff/out/tasks.jsonl",
    "tasks_postcutoff/out/meta.json",
    "harness/build_conditions.py",
    "harness/run_experiment.py",
    "harness/pc_overlay.py",
    "harness/gold_check.py",
    "harness/search_proxy.py",
    "modal_replica/replica.py",
    "modal_replica/install_wheelhouse.py",
    "modal_replica/wheelhouse_manifest.json",
    "modal_replica/spawn_main.py",
    "configs/main_tasks.json",
    "tasks_silent/build_silent.py",
    "probe/knowledge_probe.py",
    "probe/results/hand_labels_20261109.json",
    "probe/results/hand_labels_20261109_set2.json",
    "probe/results/hand_labels_20261110_set3.json",
]
GLOBS = ["harness/skills/**/*", "tasks_postcutoff/out/snapshots/*", "tasks_postcutoff/out/wheels/*",
         "tasks_silent/out*/tasks.jsonl", "tasks_silent/out*/meta.json", "tasks_silent/out*/snapshots/*"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    paths = {ROOT / f for f in FILES}
    for g in GLOBS:
        paths |= {p for p in ROOT.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    overall = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    out = dict(created=time.strftime("%Y-%m-%d %H:%M:%S %z"), overall_sha256=overall, n_files=len(files), files=files)
    (ROOT / "prereg").mkdir(exist_ok=True)
    (ROOT / "prereg" / "manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(files)} files -> prereg/manifest.json\nOVERALL SHA-256 (post this): {overall}")


if __name__ == "__main__":
    main()
