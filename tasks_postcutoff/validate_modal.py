"""Validate the post-cutoff tasks on Modal (CPU only, Python 3.12, one container per task).

For each task, 4 runs in fresh venvs:
  old lib, no fix  -> expect PASS (the code worked before the upgrade)
  new lib, no fix  -> expect FAIL (the upgrade breaks it)
  new lib, fix     -> expect PASS (solvable; the hidden test is fair)
  old lib, fix     -> recorded (is the reference fix backward compatible?)
Steps mirror the official harness as far as known (verified 8 Nov): unpack snapshot, apply
test_patch, editable install, `pytest <test files> -p no:anyio -o timeout=0 -q`, resolved = exit 0.
The AUTHORITATIVE check is still the official verify_task on Kaggle (gold-patch validator).

Run from the project folder:
  python -m modal run tasks_postcutoff/validate_modal.py
Writes tasks_postcutoff/out/validation.json.
"""
from __future__ import annotations

import json
import pathlib

import modal

import os

HERE = pathlib.Path(__file__).parent
# Task folder: default = the frozen post-cutoff set; TASKS_OUT=<dir> validates another set (e.g. tasks_silent/out)
OUT = pathlib.Path(os.environ["TASKS_OUT"]).resolve() if os.environ.get("TASKS_OUT") else HERE / "out"
PYTEST = "pytest==8.3.4"  # one of the pytest versions in the official wheels dir (seen 8 Nov)
OUT_NAME = ["validation.json"]

app = modal.App("gemma-postcutoff-validate")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "patch")
    .pip_install("uv")
    .add_local_dir(OUT, "/tasks")
)


def _sh(cmd: list[str], cwd: str | None = None, timeout: int = 600) -> tuple[int, str]:
    import subprocess

    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


def _apply(diff: str, cwd: str) -> tuple[int, str]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(diff)
    rc, out = _sh(["git", "apply", "--verbose", f.name], cwd)
    if rc != 0:  # fall back to GNU patch for traditional diffs
        rc2, out2 = _sh(["patch", "-p1", "--forward", "--no-backup-if-mismatch", "-i", f.name], cwd)
        return rc2, out + "\n[patch fallback]\n" + out2
    return rc, out


@app.function(image=image, cpu=2, memory=4096, timeout=1800)
def validate(task_id: str, offline: bool = False) -> dict:
    """offline=True: the NEW env installs only from /tasks/wheels (--no-index), as on Kaggle."""
    import shutil
    import tarfile
    import tempfile

    task = next(json.loads(l) for l in open("/tasks/tasks.jsonl") if json.loads(l)["instance_id"] == task_id)
    meta = json.load(open("/tasks/meta.json"))[task_id]
    res = {"task": task_id, "runs": {}}
    for env_name in ("old", "new"):
        venv = tempfile.mkdtemp(prefix=f"venv_{env_name}_")
        rc, out = _sh(["uv", "venv", venv, "--python", "3.12"])
        py = f"{venv}/bin/python"
        src = ["--no-index", "--find-links", "/tasks/wheels"] if (offline and env_name == "new") else []
        rc, out = _sh(["uv", "pip", "install", "--python", py, *src, PYTEST, "setuptools", *meta[env_name]],
                      timeout=900)
        res.setdefault("install_source", {})[env_name] = "offline wheels" if src else "PyPI"
        if rc != 0:
            res["runs"][env_name] = {"install_error": out[-2000:]}
            continue
        _, freeze = _sh(["uv", "pip", "freeze", "--python", py])
        res.setdefault("freeze", {})[env_name] = freeze.strip().splitlines()
        for fix in (False, True):
            work = tempfile.mkdtemp(prefix="work_")
            with tarfile.open(f"/tasks/snapshots/{task_id}.tgz") as t:
                t.extractall(work, filter="data")
            steps = {}
            steps["test_patch"] = _apply(task["test_patch"], work)
            if fix:
                steps["patch"] = _apply(task["patch"], work)
            # editable install of the workspace package, no deps (they are pinned above)
            steps["install_e"] = _sh(["uv", "pip", "install", "--python", py, "--no-deps", *src, "-e", "."], work)
            rc, out = _sh([py, "-m", "pytest", *meta["test_files"], "-p", "no:anyio", "-o", "timeout=0", "-q"], work)
            key = f"{env_name}_{'fix' if fix else 'nofix'}"
            res["runs"][key] = {"exit": rc, "passed": rc == 0, "tail": out[-1500:],
                                "apply_ok": all(v[0] == 0 for k, v in steps.items()),
                                "apply_log": {k: v[1][-400:] for k, v in steps.items() if v[0] != 0}}
            # No uninstall: the next `-e .` replaces this editable install. (Don't uninstall by the
            # repo name: for "postcutoff/marshmallow" that would remove the library under test.)
            shutil.rmtree(work, ignore_errors=True)
    r = res["runs"]
    get = lambda k: r.get(k, {}).get("passed")
    res["verdict"] = "OK" if (get("old_nofix") is True and get("new_nofix") is False and get("new_fix") is True) \
        else "CHECK"
    return res


@app.local_entrypoint()
def main(only: str = "", offline: bool = False):
    ids = [json.loads(l)["instance_id"] for l in (OUT / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    if only:
        ids = [i for i in ids if i in only.split(",")]
    results = list(validate.map(ids, kwargs={"offline": offline}))
    name = "validation_offline.json" if offline else "validation.json"
    (OUT / name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    OUT_NAME[0] = name
    print(f"\n{'task':32} old_nofix new_nofix new_fix old_fix  verdict")
    for r in results:
        g = lambda k: {True: "PASS", False: "FAIL", None: "-"}[r["runs"].get(k, {}).get("passed")]
        print(f"{r['task']:32} {g('old_nofix'):9} {g('new_nofix'):9} {g('new_fix'):7} {g('old_fix'):8} {r['verdict']}")
    print(f"\nsaved {OUT / OUT_NAME[0]}")
