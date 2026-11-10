"""Give post-cutoff tasks their NEW library versions inside swegemma's subprocess sandbox.

Why (verified on Kaggle 9 Nov, cells C/D):
  * swegemma `install_test_dependencies` returns immediately for the subprocess sandbox, so task
    wheels are NEVER installed in subprocess mode (container_setup.py).
  * Sandbox venvs use system_site_packages=True plus a `_host_env.pth`, so tests import the
    NOTEBOOK's packages (sandbox/subprocess.py:37, 336, 389-400). On Kaggle these are old
    (marshmallow 3.26.2, pandas 2.3.3, ...), so the official check gave 3/12.
  * Every sandbox command's environment comes from `build_sanitized_env(extra_env={'PYTHONPATH':
    workspace, ...})` (sandbox/subprocess.py:483-494), called by bare name, so it can be wrapped.

Fix: unpack our wheels once into an overlay folder, and while post-cutoff tasks run set
PYTHONPATH = workspace:overlay. PYTHONPATH comes before site-packages for any interpreter, so
the new versions win for the agent's commands, the skills and the grader's pytest, while the
notebook (vLLM, harness) and the control tasks are untouched. The pytest stack is NOT put in the
overlay, so both task groups use the same pytest. Grading code is unchanged.

Do not run control and post-cutoff tasks concurrently in one process while the overlay is enabled.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

EXCLUDE = {"pytest", "pluggy", "iniconfig", "packaging", "setuptools"}  # keep the host's pytest stack
CHECK = ("marshmallow", "polars", "click", "scikit-learn", "django", "networkx", "pandas", "starlette", "numpy")

_state = {"overlay": None, "orig": None}


def _dist(wheel_name: str) -> str:
    return re.sub(r"[-_.]+", "-", wheel_name.split("-")[0]).lower()


def build_overlay(wheels_dir, target, python: str | None = None) -> Path:
    """pip install --target the wheels (minus EXCLUDE) into `target`. Uses the host's pip."""
    wheels_dir, target = Path(wheels_dir), Path(target)
    wheels = sorted(p for p in wheels_dir.glob("*.whl") if _dist(p.name) not in EXCLUDE)
    if target.exists():
        import shutil
        shutil.rmtree(target)
    cmd = [python or sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet",
           "--disable-pip-version-check", "--target", str(target), *map(str, wheels)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"overlay install failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return target


def overlay_versions(overlay, python: str | None = None) -> dict:
    """Versions a fresh interpreter sees with PYTHONPATH=overlay (the way the sandbox will)."""
    code = ("import importlib.metadata as m, json\n"
            f"out={{}}\nfor p in {list(CHECK)!r}:\n"
            "    try: out[p]=m.version(p)\n"
            "    except Exception as e: out[p]=repr(e)\n"
            "print(json.dumps(out))")
    env = {**os.environ, "PYTHONPATH": str(overlay)}
    r = subprocess.run([python or sys.executable, "-c", code], capture_output=True, text=True, env=env)
    import json
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-1000:]}


def enable(overlay) -> None:
    """Wrap swegemma's sandbox env builder so PYTHONPATH = <workspace>:<overlay>."""
    import swegemma.sandbox.subprocess as sbx

    if _state["orig"] is None:
        _state["orig"] = sbx.build_sanitized_env
    orig = _state["orig"]
    _state["overlay"] = str(overlay)

    def build_sanitized_env_with_overlay(*args, **kwargs):
        env = orig(*args, **kwargs)
        ov = _state["overlay"]
        if ov:
            pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (pp + os.pathsep + ov) if pp else ov
        return env

    sbx.build_sanitized_env = build_sanitized_env_with_overlay


def disable() -> None:
    import swegemma.sandbox.subprocess as sbx

    _state["overlay"] = None
    if _state["orig"] is not None:
        sbx.build_sanitized_env = _state["orig"]


def selftest(workspace="/tmp/_ov_ws") -> str:
    """Call the (wrapped) env builder the way exec() does and return the resulting PYTHONPATH."""
    import swegemma.sandbox.subprocess as sbx

    Path(workspace).mkdir(parents=True, exist_ok=True)
    env = sbx.build_sanitized_env(work_dir=Path(workspace), tmp_dir=Path("/tmp"),
                                  extra_env={"PYTHONPATH": workspace})
    return env.get("PYTHONPATH", "<missing>")
