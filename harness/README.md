# harness/

Experiment code for Paper 1. It wraps the **official** competition harness (`swegemma`,
`adk_submission`, google-adk from the Kaggle wheelhouse) and changes no grading code.

| File | Purpose | Tested |
|---|---|---|
| `build_conditions.py` | Builds agent folders C0–C4 from the official `sample_submission` (adapters removed, fixed budget, /tmp scratch rule, condition prompt + skills) | ✅ locally on a replica of the sample agent |
| `skills/library-docs/` | Offline skill (C3/C4): installed version, public API, signatures, source, grep, bundled changelog | ✅ `pkg_info.py` locally |
| `skills/web-search/` | Tavily skill (C1/C2/C4); logs queries + URLs to `$SKILL_LOG` | ⚠ needs TAVILY_API_KEY in the sandbox environment (Nebius) |
| `run_experiment.py` | Runs condition × task × repeat via `Evaluator.evaluate_task`; resumable; concurrency via `slot_id`; saves patch, test output, tokens, compaction and skill counts | ⚠ needs Kaggle (swegemma) |
| `sample_control.py` | Control-task selection from gold/empty-patch validation: filters, near-duplicate merge (same named function/class changed), stratified sample, seed 2026 | ✅ run on real validation output 9 Nov → `configs/control_tasks.json` (30 tasks) |
| `../probe/knowledge_probe.py` | Knowledge probe (OLD/NEW/SAFE/UNCLEAR) against any OpenAI-compatible endpoint | ✅ endpoint logic; Gemma 4 not on Token Factory → run on Kaggle vLLM |

## To verify on Kaggle before the main runs
1. **Skill registration**: does `validate_directory()` accept `run_skill_script` /
   `load_skill_resource` in the tools list plus a `skills/` folder? Check HARNESS_README's skills
   section. `run_batch` calls `validate_directory` first, so it fails fast.
2. **Skill script execution**: does `run_skill_script` run with the task's venv Python (so
   `pkg_info.py` sees the task's installed libraries)?
3. **Concurrency**: does `evaluate_task(..., slot_id=k)` run safely in parallel (3 slots)?
4. **Trace parsing**: ✅ partly verified 9 Nov on the Run 1 traces (ATIF-v1.7). Tool calls are in
   `steps[].tool_calls[].function_name/arguments`. Compaction has NO event: it is detected as
   per-step `prompt_tokens` dropping below 50% of the previous step (Run 1: 1 compaction per task).
   The old text-search heuristics were wrong (0 compactions; skill names in the system prompt
   counted as calls) and were replaced. Still UNVERIFIED: the `run_skill_script` argument
   names. Check on the first C1/C3 trace.
6. **TaskResult field names** used by `run_batch` (`trace_path`, `end_status`, `total_llm_calls`)
   are UNVERIFIED (read with `getattr`, so they would silently be None). Print
   `vars(result)` for one task before the first batch.
5. **C1 sandbox env**: TAVILY_API_KEY visible inside the subprocess sandbox on Nebius.

## Code-graph tools: OFF for all tasks (decision, 9 Nov)
Checked on Kaggle (swegemma `harness/agent_runner.py:416-466`): the "Code Intelligence Tools"
prompt section and working graph tools exist only for tasks with precomputed graph/embedding
files, which come from a Google-internal pipeline we can't reproduce for the post-cutoff tasks.
To keep prompts and tools identical across task groups, `build_conditions.GRAPH_TOOLS = False`
removes `get_code_neighbors`, `search_similar_code`, `get_code_subgraph` from every agent YAML,
and `run_batch` passes no graph/embeddings dirs. Disclosed in the paper as a deviation from
the official sample agent. Check on Kaggle: `validate_directory()` accepts the edited YAMLs,
and no prompt file names the removed tools (the builder prints a warning if one does).

## Budget (identical for all conditions)
60 tool calls, 15 min per task, 900 s command timeout (`BUDGET` in `build_conditions.py`).
Official notebook compaction settings kept (interval 15, overlap 2, threshold 14,336 tokens,
retention 5).
