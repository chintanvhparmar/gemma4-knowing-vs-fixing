"""Test harness/pc_overlay.py on Modal under Kaggle-like conditions (old host packages).

python -m modal run tasks_postcutoff/test_overlay_modal.py
"""
from __future__ import annotations

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
app = modal.App("gemma-overlay-test")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "patch")
    # "host" = old versions, like the Kaggle notebook (marshmallow 3.x, pandas 2.x) + pytest
    .pip_install("marshmallow==3.23.0", "pandas==2.2.3", "pytest==8.3.4")
    .add_local_dir(HERE / "out", "/tasks")
    .add_local_file(HERE.parent / "harness" / "pc_overlay.py", "/root/pc_overlay.py")
)

STUB = '''
def build_sanitized_env(work_dir, tmp_dir, extra_env=None):
    import os
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_dir)}
    env.update(extra_env or {})
    return env
'''


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def run() -> dict:
    import json, os, subprocess, sys, tarfile, tempfile
    sys.path.insert(0, "/root")
    # stand-in for swegemma.sandbox.subprocess (same function name + call style as the real one)
    stub = pathlib.Path("/tmp/stub/swegemma/sandbox"); stub.mkdir(parents=True)
    (stub.parent / "__init__.py").write_text(""); (stub / "__init__.py").write_text("")
    (stub / "subprocess.py").write_text(STUB)
    sys.path.insert(0, "/tmp/stub")
    import pc_overlay
    import swegemma.sandbox.subprocess as sbx

    out = {}
    ov = pc_overlay.build_overlay("/tasks/wheels", "/tmp/pc_overlay")
    out["overlay_versions"] = pc_overlay.overlay_versions(ov)
    out["overlay_has_pytest"] = any(p.name.startswith("pytest") for p in ov.iterdir())

    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open("/tasks/tasks.jsonl")}
    meta = json.load(open("/tasks/meta.json"))

    def run_task(tid, fix, overlay_on):
        work = tempfile.mkdtemp()
        with tarfile.open(f"/tasks/snapshots/{tid}.tgz") as t:
            t.extractall(work, filter="data")
        for diff in ([tasks[tid]["test_patch"]] + ([tasks[tid]["patch"]] if fix else [])):
            f = tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False); f.write(diff); f.close()
            subprocess.run(["git", "apply", f.name], cwd=work, check=True)
        (pc_overlay.enable if overlay_on else (lambda _ov: pc_overlay.disable()))(ov)
        env = sbx.build_sanitized_env(work_dir=pathlib.Path(work), tmp_dir=pathlib.Path("/tmp"),
                                      extra_env={"PYTHONPATH": work})
        r = subprocess.run([sys.executable, "-m", "pytest", *meta[tid]["test_files"], "-q", "-p", "no:anyio"],
                           cwd=work, env=env, capture_output=True, text=True)
        return {"passed": r.returncode == 0, "PYTHONPATH": env["PYTHONPATH"].replace(work, "<ws>"),
                "last": (r.stdout.strip().splitlines() or [""])[-1]}

    for tid in ("pc_marshmallow_validates", "pc_pandas_epoch_seconds", "pc_marshmallow_context"):
        out[tid] = {"host_nofix": run_task(tid, False, False),
                    "overlay_nofix": run_task(tid, False, True),
                    "overlay_fix": run_task(tid, True, True)}
    pc_overlay.disable()
    out["selftest_after_disable"] = pc_overlay.selftest()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(run.remote(), indent=2))
