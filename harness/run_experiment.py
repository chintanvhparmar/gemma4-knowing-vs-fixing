"""Run (condition × task × repeat) through the OFFICIAL swegemma harness and record everything.

Designed to be imported in a Kaggle (or Nebius) notebook AFTER:
  cell 1  wheelhouse install          (official)
  cell 2  data + tasks loaded         (DATA_DIR, WORKING_DIR, tasks)
  cell 4  vLLM server started         (models = server_instance.create_model_registry(...))

    import sys; sys.path.insert(0, '/kaggle/input/<our-code-dataset>/harness')
    from build_conditions import build_all
    from run_experiment import run_batch
    agent_dirs = build_all(DATA_DIR / 'sample_submission', WORKING_DIR / 'agents', model=MODEL_NAME,
                           conditions=['C0', 'C3'])
    run_batch(agent_dirs, task_ids=CONTROL + POSTCUTOFF, tasks=tasks, repeats=2, models=models,
              data_dir=DATA_DIR, out_dir=WORKING_DIR / 'runs', concurrency=3, max_hours=8.0)

What it records per run (one JSON line in <out_dir>/runs.jsonl; full test output and patch in
<out_dir>/<cond>/rep<r>/<task>.json):
  condition, task, repeat, platform, resolved, test_exit_code, error, end_status,
  tool_calls, llm_calls, duration_s, prompt/completion/cached tokens (trace final_metrics),
  event-type histogram, compaction events, skill usage (from the trace), patch size + files.
Resumable: (condition, task, repeat) already in runs.jsonl are skipped.
Time guard: no new runs start after `max_hours` (Kaggle sessions end at 9-12 h).
"""
from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import inspect
import json
import os
import platform
import re
import shutil
import time
from pathlib import Path

PATCH_FILE_RX = re.compile(r"^\+\+\+ b/(\S+)", re.M)
try:  # same switch as the agent builder, so prompts and tools stay consistent
    from build_conditions import GRAPH_TOOLS
except ImportError:
    GRAPH_TOOLS = False


def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():  # inside Jupyter
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


COMPACTION_DROP = 0.5  # prompt shrinks to <50% of the previous step => compaction happened
SKILL_TOOLS = ("load_skill", "load_skill_resource", "run_skill_script")  # = swegemma _SKILL_TOOL_NAMES


def _trace_stats(trace_path: str | None) -> dict:
    """Stats from an ATIF-v1.7 trace (format verified on Run 1 traces, 9 Nov).

    Verified format: top keys schema_version/agent/steps/final_metrics; each step has
    source, message, extra.event_type (system_instruction|task_prompt|thinking|final),
    metrics.prompt_tokens, tool_calls[{function_name, arguments{...}}].
    - Compaction is NOT logged as an event. It shows as a sharp drop in per-step
      prompt_tokens (Run 1: 15,527 -> 5,113 and 14,570 -> 3,677, just after crossing the
      14,336-token threshold), so we count those drops.
    - Skill use is counted ONLY from tool_calls (not message text: the system prompt and
      the agent's thinking mention skill names too).
      UNVERIFIED: the argument names of run_skill_script (no skill trace yet), so we match
      the skill/script name anywhere in the JSON of `arguments`. Check on the first C1/C3 trace.
    """
    s = dict(prompt_tokens=None, completion_tokens=None, cached_tokens=None, steps=None,
             event_types={}, tool_hist={}, compactions=0, max_step_prompt_tokens=None,
             skill_calls={}, search_calls=0, docs_calls=0)
    if not trace_path or not Path(trace_path).is_file():
        return s
    try:
        t = json.loads(Path(trace_path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return s
    fm = t.get("final_metrics") or {}
    s.update(prompt_tokens=fm.get("total_prompt_tokens"), completion_tokens=fm.get("total_completion_tokens"),
             cached_tokens=fm.get("total_cached_tokens"), steps=fm.get("total_steps"))
    ev, tools, skills = collections.Counter(), collections.Counter(), collections.Counter()
    prev = None
    peak = 0
    for st in t.get("steps", []):
        ev[str((st.get("extra") or {}).get("event_type", "?"))] += 1
        pt = (st.get("metrics") or {}).get("prompt_tokens")
        if isinstance(pt, int):
            if prev and pt < COMPACTION_DROP * prev:
                s["compactions"] += 1
            prev, peak = pt, max(peak, pt)
        for tc in st.get("tool_calls") or []:
            name = tc.get("function_name") or "?"
            tools[name] += 1
            if name in SKILL_TOOLS:
                args = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
                if "web_search" in args or "web-search" in args:
                    skill = "web-search"
                elif "pkg_info" in args or "library-docs" in args:
                    skill = "library-docs"
                else:
                    skill = "other"
                skills[f"{skill}/{name}"] += 1  # e.g. library-docs/load_skill, library-docs/run_skill_script
    s["event_types"], s["tool_hist"], s["skill_calls"] = dict(ev), dict(tools), dict(skills)
    s["max_step_prompt_tokens"] = peak or None
    # "used the capability" = actually RAN the script (loading instructions alone doesn't count)
    s["search_calls"] = skills.get("web-search/run_skill_script", 0)
    s["docs_calls"] = skills.get("library-docs/run_skill_script", 0)
    s["skill_loads"] = sum(v for k, v in skills.items() if k.endswith("/load_skill"))
    return s


def _result_fields(res) -> dict:
    """All fields of the official TaskResult, whatever its type (dataclass / pydantic / plain)."""
    for attr in ("model_dump", "dict"):
        if callable(getattr(res, attr, None)):
            try:
                return dict(getattr(res, attr)())
            except Exception:
                pass
    try:
        return dict(vars(res))
    except TypeError:
        return {}


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)[:500]


def _done_keys(runs_path: Path) -> set:
    if not runs_path.exists():
        return set()
    keys = set()
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["condition"], r["task"], r["repeat"]))
    return keys


def run_batch(agent_dirs: dict, task_ids: list[str], tasks: list, repeats: int, models,
              data_dir: Path, out_dir: Path, concurrency: int = 3, max_hours: float = 8.0,
              wheels_dir: Path | None = None, extra_snapshots: dict | None = None,
              platform_name: str | None = None, tasks_path: Path | None = None,
              snapshots_dir: Path | None = None) -> Path:
    """agent_dirs: {'C0': Path, ...}; task_ids: instance_ids to run; tasks: loaded Task objects.

    Defaults use the official dev set under data_dir. For the post-cutoff group pass our own
    tasks_path / snapshots_dir / wheels_dir (tasks_postcutoff/out/...), with a separate out_dir.
    """
    from adk_submission import validate_directory
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    import yaml

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    by_id = {t.instance_id: t for t in tasks}
    missing = [i for i in task_ids if i not in by_id]
    if missing:
        raise ValueError(f"unknown task ids: {missing}")
    plat = platform_name or ("kaggle" if os.path.exists("/kaggle") else platform.node())
    limits, gen_constraints = build_submission_limits()

    evaluators = {}
    for cond, adir in agent_dirs.items():
        validate_directory(Path(adir), limits=limits)  # fails fast on bad agent.yaml / skills
        ev = yaml.safe_load((Path(adir) / "eval_config.yaml").read_text())["evaluation"]
        cfg = EvalConfig(
            tasks_path=Path(tasks_path) if tasks_path else data_dir / "tasks.jsonl",
            snapshots_dir=Path(snapshots_dir) if snapshots_dir else data_dir / "snapshots",
            results_dir=out_dir / cond, submission_dir=Path(adir), models=models, sandbox="subprocess",
            timeout_seconds=int(ev.get("timeout_seconds", 900)),
            verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 900)),
            max_time_minutes=float(ev.get("max_time_minutes", 15)),
            max_tool_calls=int(ev.get("max_tool_calls", 60)), max_turns=ev.get("max_turns"),
            limits=limits, generation_constraints=gen_constraints,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(  # official notebook values
                compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
            # No graph/embeddings dirs when graph tools are off: then agent_runner never adds the
            # "Code Intelligence Tools" prompt section, for any task (build_conditions change 4).
            graph_dir=str(data_dir / "graphs") if GRAPH_TOOLS else None,
            embeddings_dir=str(data_dir / "embeddings") if GRAPH_TOOLS else None,
            wheels_dir=Path(wheels_dir) if wheels_dir else data_dir / "wheels", verbose=False,
        )
        evaluators[cond] = Evaluator(cfg)

    done = _done_keys(runs_path)
    jobs = [(c, tid, r) for r in range(1, repeats + 1) for tid in task_ids for c in agent_dirs
            if (c, tid, r) not in done]
    print(f"[run_batch] {len(jobs)} runs to do ({len(done)} already done); concurrency={concurrency}", flush=True)
    t_start = time.time()
    lock = asyncio.Lock()

    async def one(slot: int, job):
        cond, tid, rep = job
        task = by_id[tid]
        t0 = time.time()
        try:
            # Don't assume the official signature: pass slot_id only if accepted; await only if async
            # (the official notebook calls it through a generic run_sync helper).
            fn = evaluators[cond].evaluate_task
            kwargs = dict(task=task, task_index=1, total_tasks=1)
            if "slot_id" in inspect.signature(fn).parameters:
                kwargs["slot_id"] = slot
            if inspect.iscoroutinefunction(fn):
                res = await fn(**kwargs)
            else:
                res = await asyncio.to_thread(fn, **kwargs)
            patch = getattr(res, "agent_patch", None) or ""
            # The harness writes results_dir/traces/trace_<instance_id>.json (save_trace_artifact,
            # verified 8 Nov). Repeats would overwrite it, so copy it into this run's folder now.
            # TaskResult has trace_path (verified 9 Nov); fall back to the save_trace_artifact location.
            src_trace = Path(getattr(res, "trace_path", None) or (out_dir / cond / "traces" / f"trace_{tid}.json"))
            d = out_dir / cond / f"rep{rep}"
            d.mkdir(parents=True, exist_ok=True)
            trace_path = None
            if src_trace.is_file():
                trace_path = d / f"{tid}.trace.json"
                shutil.copy2(src_trace, trace_path)
            fields = {k: _jsonable(v) for k, v in _result_fields(res).items()
                      if k not in ("agent_patch", "test_output", "trace")}  # trace is saved as a file
            # Field names verified on Kaggle 9 Nov: error_message (not error), status, total_llm_calls,
            # total_tokens, duration_seconds.
            rec = dict(resolved=bool(getattr(res, "resolved", False)),
                       test_exit_code=getattr(res, "test_exit_code", None),
                       error=getattr(res, "error_message", None), status=_jsonable(getattr(res, "status", None)),
                       tool_calls=getattr(res, "tool_calls", None), llm_calls=getattr(res, "total_llm_calls", None),
                       total_tokens=getattr(res, "total_tokens", None),
                       patch_chars=len(patch), patch_files=PATCH_FILE_RX.findall(patch),
                       **_trace_stats(str(trace_path) if trace_path else None))
            full = dict(agent_patch=patch, test_output=getattr(res, "test_output", "") or "",
                        trace_path=str(trace_path) if trace_path else None, result_fields=fields)
        except Exception as e:  # harness crash: record, don't stop the batch
            rec = dict(resolved=False, test_exit_code=None, error=f"HARNESS EXCEPTION: {e!r}")
            full = {}
        rec = dict(condition=cond, task=tid, repo=task.repo, repeat=rep, platform=plat,
                   duration_s=round(time.time() - t0, 1), ts=time.strftime("%Y-%m-%dT%H:%M:%S"), **rec)
        d = out_dir / cond / f"rep{rep}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.json").write_text(json.dumps({**rec, **full}, ensure_ascii=False), encoding="utf-8")
        async with lock:
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{cond} r{rep}] {tid:22} resolved={rec['resolved']!s:5} exit={rec.get('test_exit_code')} "
              f"tools={rec.get('tool_calls')} tok_in={rec.get('prompt_tokens')} search={rec.get('search_calls')} "
              f"docs={rec.get('docs_calls')} {rec['duration_s']}s", flush=True)

    async def main():
        queue = list(jobs)
        sem = asyncio.Semaphore(concurrency)

        busy = set()  # (cond, task) pairs running now: they share results_dir/traces/trace_<id>.json

        async def worker(slot):
            while queue:
                if time.time() - t_start > max_hours * 3600:
                    print(f"[run_batch] time guard reached ({max_hours} h); stopping slot {slot}", flush=True)
                    return
                idx = next((i for i, j in enumerate(queue) if (j[0], j[1]) not in busy), None)
                if idx is None:  # everything left is already running in another slot
                    await asyncio.sleep(5)
                    continue
                job = queue.pop(idx)
                busy.add((job[0], job[1]))
                try:
                    async with sem:
                        await one(slot, job)
                finally:
                    busy.discard((job[0], job[1]))

        await asyncio.gather(*(worker(s) for s in range(concurrency)))

    _run_sync(main())
    print(f"[run_batch] finished in {(time.time() - t_start) / 60:.1f} min -> {runs_path}", flush=True)
    return runs_path
