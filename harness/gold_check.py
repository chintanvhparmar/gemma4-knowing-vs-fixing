"""Gold/empty check with the OFFICIAL grader, identical on Kaggle and on the Modal replica.

For each task: grade the reference fix ("gold", must pass) and an empty patch ("empty", must fail)
with swegemma's own Evaluator (agent step replaced by a fixed patch, as in kaggle/cell_B and
cell_B2). Post-cutoff tasks are graded with the per-task package overlay (harness/pc_overlay.py).
Writes one JSON line per (mode, task) plus an environment header line, so two platforms can be
compared with harness/compare_gold.py.

Kaggle (any notebook, after the official cells 1 and 2):
    import sys; sys.path.insert(0, str(CODE / 'harness')); import gold_check
    gold_check.run(DATA_DIR, WORKING_DIR, CODE, WORKING_DIR / 'gold_check_kaggle.jsonl', platform='kaggle-gpu')
Modal: modal_replica/replica.py::gold_check (same function).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.metadata as md
import inspect
import json
import platform as _platform
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

KEY_PKGS = ["swegemma", "adk-submission", "pytest", "fastapi", "starlette", "pydantic", "rich", "httpx",
            "networkx", "polars", "pandas", "marshmallow", "click", "scikit-learn", "django", "torch"]


def _run_sync(fn, *args, **kwargs):
    async def _call():
        r = fn(*args, **kwargs)
        return (await r) if inspect.isawaitable(r) else r
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():  # Jupyter
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_call())).result()
    return asyncio.run(_call())


def _summary(out: str) -> str:
    lines = [l for l in out.splitlines() if re.search(r"\d+ (passed|failed|errors?)\b", l)]
    return lines[-1].strip() if lines else ""


def _env(platform: str) -> dict:
    v = {}
    for k in KEY_PKGS:
        try:
            v[k] = md.version(k)
        except md.PackageNotFoundError:
            v[k] = None
    return dict(header=True, platform=platform, python=sys.version.split()[0], machine=_platform.platform(),
                n_packages=len(list(md.distributions())), versions=v, time=time.strftime("%Y-%m-%d %H:%M:%S"))


def run(data_dir, working_dir, code_dir, out_path, platform: str, control_ids=None, pc=True, limit=0):
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    data_dir, working_dir, code_dir, out_path = map(Path, (data_dir, working_dir, code_dir, out_path))
    sys.path.insert(0, str(code_dir / "harness"))
    import pc_overlay

    if control_ids is None:
        control_ids = json.loads((code_dir / "configs" / "control_tasks.json").read_text())["control_tasks"]
    by_id = {t.instance_id: t for t in load_tasks(data_dir / "tasks.jsonl")}
    control = [by_id[i] for i in control_ids]
    pc_dir = code_dir / "tasks_postcutoff" / "out"
    pc_tasks = load_tasks(pc_dir / "tasks.jsonl") if pc else []
    if limit:
        control, pc_tasks = control[:limit], pc_tasks[:limit]

    agent_dir = working_dir / "gold_check_agent"  # official sample agent; never run (patch is fixed)
    shutil.rmtree(agent_dir, ignore_errors=True)
    shutil.copytree(data_dir / "sample_submission", agent_dir)

    # wheels dir as in cell B: official wheels + ours (ours win). Not installed by the subprocess sandbox.
    def _pkg(name):
        return re.sub(r"[-_.]+", "-", name.split("-")[0]).lower()
    pc_wheels = working_dir / "gold_check_pc_wheels"
    shutil.rmtree(pc_wheels, ignore_errors=True)
    pc_wheels.mkdir(parents=True)
    ours = {_pkg(p.name) for p in (pc_dir / "wheels").glob("*.whl")}
    for p in (data_dir / "wheels").glob("*.whl"):
        if _pkg(p.name) not in ours:
            shutil.copy2(p, pc_wheels)
    for p in (pc_dir / "wheels").glob("*.whl"):
        shutil.copy2(p, pc_wheels)

    mode = {"patch": {}}
    if inspect.iscoroutinefunction(Evaluator._run_agent_sandbox):
        async def _fixed(self, task, *a, **k):
            return mode["patch"][task.instance_id], None, None
    else:
        def _fixed(self, task, *a, **k):
            return mode["patch"][task.instance_id], None, None
    original = Evaluator._run_agent_sandbox
    Evaluator._run_agent_sandbox = _fixed

    def evaluator(snapshots_dir, tasks_path, wheels_dir, results_dir):
        limits, gen = build_submission_limits()
        return Evaluator(EvalConfig(
            tasks_path=tasks_path, snapshots_dir=snapshots_dir, results_dir=results_dir, submission_dir=agent_dir,
            models={}, sandbox="subprocess", timeout_seconds=600, verification_timeout_seconds=900,
            max_time_minutes=30.0, max_tool_calls=100, max_turns=None, limits=limits, generation_constraints=gen,
            graph_dir=None, embeddings_dir=None, wheels_dir=wheels_dir, verbose=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_env(platform)) + "\n")

    def grade(ev, group, tasks):
        for label in ("gold", "empty"):
            mode["patch"] = {t.instance_id: (t.patch if label == "gold" else "") for t in tasks}
            for i, t in enumerate(tasks, 1):
                t0 = time.time()
                try:
                    r = _run_sync(ev.evaluate_task, task=t, task_index=i, total_tasks=len(tasks))
                    out = r.test_output or ""
                    rec = dict(group=group, mode=label, id=t.instance_id, resolved=bool(r.resolved),
                               exit_code=r.test_exit_code, error=getattr(r, "error", None),
                               summary=_summary(out), test_tail=out[-3000:])
                except Exception as e:
                    rec = dict(group=group, mode=label, id=t.instance_id, resolved=False, exit_code=None,
                               error=f"HARNESS EXCEPTION: {e!r}", summary="", test_tail=traceback.format_exc()[-3000:])
                rec["duration"] = round(time.time() - t0, 1)
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{group} {label} {i}/{len(tasks)}] {t.instance_id:30} resolved={str(rec['resolved']):5} "
                      f"exit={rec['exit_code']} {rec['duration']:>6}s {rec['summary'][:70]}"
                      + (f"  ERR: {str(rec['error'])[:120]}" if rec["error"] else ""), flush=True)

    try:
        grade(evaluator(data_dir / "snapshots", data_dir / "tasks.jsonl", data_dir / "wheels",
                        working_dir / "gold_check_results" / "control"), "control", control)
        if pc_tasks:
            ov = pc_overlay.build_overlay(pc_dir / "wheels", working_dir / "gold_check_pc_overlay")
            pc_overlay.enable(ov)
            try:
                grade(evaluator(pc_dir / "snapshots", pc_dir / "tasks.jsonl", pc_wheels,
                                working_dir / "gold_check_results" / "postcutoff"), "postcutoff", pc_tasks)
            finally:
                pc_overlay.disable()
    finally:
        Evaluator._run_agent_sandbox = original

    recs = [json.loads(l) for l in out_path.read_text().splitlines()[1:]]
    g = {(r["group"], r["id"]): r["resolved"] for r in recs if r["mode"] == "gold"}
    e = {(r["group"], r["id"]): r["resolved"] for r in recs if r["mode"] == "empty"}
    ok = [k for k in g if g[k] and not e.get(k)]
    print(f"\n=== {platform}: {len(ok)}/{len(g)} OK (gold passes AND empty fails) -> {out_path}")
    print("problems:", {f"{k[0]}/{k[1]}": (g[k], e.get(k)) for k in g if k not in ok})
    return out_path
