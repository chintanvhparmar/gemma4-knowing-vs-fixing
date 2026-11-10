"""Build one ADK agent folder per experimental condition from the OFFICIAL sample agent.

Why generate instead of hand-writing: C0 stays byte-identical to the official
`sample_submission` except for documented changes, which keeps results transferable
to the competition harness and makes the paper's method easy to verify.

Changes applied to EVERY condition (incl. C0), documented in the paper:
  1. Sample LoRA adapters removed (`adapter:` lines + adapters/ dir): plain 31B base model.
  2. eval_config.yaml replaced by our fixed budget (BUDGET below), identical for all conditions.
  3. One line appended to the system prompt: scratch/repro files go under /tmp
     (Run 1 showed scratch files leaking into submitted patches).
  4. (GRAPH_TOOLS = False) the 3 code-graph tools are removed from every agent YAML. Reason
     (verified 9 Nov in swegemma agent_runner.py:416-466): the graph prompt section and working
     graph tools exist only for tasks with precomputed graph/embedding files, which come from a
     Google-internal pipeline we can't reproduce for our post-cutoff tasks. Removing them for ALL
     tasks keeps prompts and tools identical across task groups. run_batch also passes no
     graph/embeddings dirs. (Run 1 traces: the agent made 0 graph-tool calls in 2 tasks.)

Condition-specific changes:
  C0  no extra tools (official baseline + the 3 changes above)
  C1  + `web-search` skill (optional to use; prompt only states it is available)
  C2  + `web-search` skill + staleness HINT in the prompt
  C3  + `library-docs` skill (offline: installed version, signatures, docstrings, source)
  C4  + both skills
Skills are ADK Skill directories (SKILL.md with YAML front matter + scripts/), declared with a
`skills:` list in agent.yaml (verified 9 Nov: adk_submission schema.py + resolvers/tools.py); the
harness's SkillToolset then provides load_skill / load_skill_resource / run_skill_script, and
scripts run inside the task sandbox. (An earlier version wrongly added the tool names to `tools:`;
validate_directory accepted that, so it is not a sufficient check on its own.)

Usage (Kaggle notebook, after the data cell):
    from build_conditions import build_all
    dirs = build_all(DATA_DIR / 'sample_submission', WORKING_DIR / 'agents', model=MODEL_NAME)
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS_SRC = HERE / "skills"

BUDGET = {  # identical for every condition; reported in the paper
    "timeout_seconds": 900,
    "max_tool_calls": 60,
    "max_time_minutes": 15,
    "max_turns": None,
}

SCRATCH_RULE = (  # dry run 9 Nov: write_file/edit_file reject paths outside the repository
    "\n\n## Scratch files\n"
    "Create any scratch or reproduction scripts under `/tmp` with `run_command` (e.g. a heredoc: "
    "`cat > /tmp/repro.py <<'EOF' ... EOF`), never inside the repository, so they are not included "
    "in your patch. `write_file` and `edit_file` only work inside the repository.\n"
)

SEARCH_NOTE = (
    "\n\n## Extra capability: web search\n"
    "A `web-search` skill is available (see its SKILL.md). Using it is optional.\n"
)
DOCS_NOTE = (
    "\n\n## Extra capability: local library documentation\n"
    "A `library-docs` skill is available (see its SKILL.md). It inspects the library versions "
    "actually installed in this environment. Using it is optional.\n"
)
HINT = (
    "\n\n## Important\n"
    "The libraries installed in this environment may be newer than your training data, and their "
    "APIs may have changed. Check the installed version and its documentation before relying on "
    "an API from memory.\n"
)

CONDITIONS = {
    "C0": dict(skills=[], prompt=""),
    "C1": dict(skills=["web-search"], prompt=SEARCH_NOTE),
    "C2": dict(skills=["web-search"], prompt=SEARCH_NOTE + HINT),
    "C3": dict(skills=["library-docs"], prompt=DOCS_NOTE),
    "C4": dict(skills=["web-search", "library-docs"], prompt=SEARCH_NOTE + DOCS_NOTE),
}
SKILL_TOOLS = ["run_skill_script", "load_skill_resource"]
GRAPH_TOOLS = False  # see change 4 in the module docstring
GRAPH_TOOL_NAMES = ["get_code_neighbors", "search_similar_code", "get_code_subgraph"]


def _remove_graph_tools(folder: Path) -> list[str]:
    """Remove '- <graph tool>' list items from all agent YAMLs; return what was removed."""
    removed = []
    for y in [folder / "agent.yaml", *sorted((folder / "sub_agents").glob("*.yaml"))]:
        if not y.is_file():
            continue
        t = y.read_text(encoding="utf-8")
        for name in GRAPH_TOOL_NAMES:
            t, n = re.subn(rf"^[ \t]*-[ \t]*{name}[ \t]*\n", "", t, flags=re.M)
            if n:
                removed.append(f"{y.relative_to(folder)}:{name}")
        y.write_text(t, encoding="utf-8")
    # Prompt text naming the graph tools (exact lines verified in the official sample agent, 9 Nov):
    #   prompts/system.md:11   "- If the problem statement does not provide explicit file paths, use
    #                           `search_similar_code` with keywords ..."  -> bullet dropped
    #   prompts/analyzer.md:4  "Use `search_similar_code`, `get_code_neighbors`, `get_code_subgraph`,
    #                           and `read_file` to locate ..."            -> "Use `read_file` to locate ..."
    sysmd = folder / "prompts" / "system.md"
    if sysmd.is_file():
        t, n = re.subn(r"^[ \t]*- If the problem statement does not provide explicit file paths, use "
                       r"`search_similar_code`.*\n", "", sysmd.read_text(encoding="utf-8"), flags=re.M)
        sysmd.write_text(t, encoding="utf-8")
        if n:
            removed.append("prompts/system.md: search_similar_code bullet")
    ana = folder / "prompts" / "analyzer.md"
    if ana.is_file():
        t, n = re.subn(r"`search_similar_code`, `get_code_neighbors`, `get_code_subgraph`, and `read_file`",
                       "`read_file`", ana.read_text(encoding="utf-8"))
        ana.write_text(t, encoding="utf-8")
        if n:
            removed.append("prompts/analyzer.md: graph tools from the tool sentence")
    # warn if any prompt still names a graph tool
    for md in folder.rglob("*.md"):
        hits = [n for n in GRAPH_TOOL_NAMES if n in md.read_text(encoding="utf-8")]
        if hits:
            print(f"WARNING: {md.relative_to(folder)} mentions graph tools {hits}; review the prompt")
    return removed


def _yaml_budget() -> str:
    lines = ["# Generated by harness/build_conditions.py: identical budget for every condition", "evaluation:"]
    for k, v in BUDGET.items():
        if v is not None:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"


def _strip_adapters_and_set_model(folder: Path, model: str) -> None:
    for y in [folder / "agent.yaml", *sorted((folder / "sub_agents").glob("*.yaml"))]:
        if y.is_file():
            t = y.read_text(encoding="utf-8")
            t = re.sub(r"^(\s*model:\s*).*$", rf"\g<1>{model}", t, flags=re.M)
            t = re.sub(r"^\s*adapter:\s*.*\n?", "", t, flags=re.M)
            y.write_text(t, encoding="utf-8")
    shutil.rmtree(folder / "adapters", ignore_errors=True)


def _add_skills(agent_yaml: Path, skills: list[str]) -> None:
    """Declare skills the way adk_submission expects (verified 9 Nov on Kaggle):
    `skills: [<relative dir>, ...]` on the agent config (schema.py:288-291). resolve_skills()
    builds ONE SkillToolset from it (resolvers/tools.py:129-194) that provides load_skill /
    load_skill_resource / run_skill_script itself, so the tool names are NOT added to `tools:`.
    Scripts run inside the task sandbox (swegemma AdkSandboxCodeExecutor, agent_runner.py:609)."""
    t = agent_yaml.read_text(encoding="utf-8")
    if re.search(r"^skills:", t, flags=re.M):
        raise RuntimeError("agent.yaml already has a skills: key")
    if not t.endswith("\n"):
        t += "\n"
    t += "skills:\n" + "".join(f"  - skills/{s}\n" for s in skills)
    agent_yaml.write_text(t, encoding="utf-8")


def build_condition(sample_dir: Path, out_dir: Path, cond: str, model: str) -> Path:
    spec = CONDITIONS[cond]
    dst = out_dir / cond
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(sample_dir, dst)
    _strip_adapters_and_set_model(dst, model)
    if not GRAPH_TOOLS:
        removed = _remove_graph_tools(dst)
        print(f"[{cond}] graph tools removed: {removed}")
    (dst / "eval_config.yaml").write_text(_yaml_budget(), encoding="utf-8")
    sysmd = dst / "prompts" / "system.md"
    sysmd.write_text(sysmd.read_text(encoding="utf-8").rstrip() + SCRATCH_RULE + spec["prompt"], encoding="utf-8")
    if spec["skills"]:
        for s in spec["skills"]:
            shutil.copytree(SKILLS_SRC / s, dst / "skills" / s)
        _add_skills(dst / "agent.yaml", spec["skills"])
    return dst


def build_all(sample_dir, out_dir, model: str, conditions=None) -> dict[str, Path]:
    sample_dir, out_dir = Path(sample_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {c: build_condition(sample_dir, out_dir, c, model) for c in (conditions or CONDITIONS)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    a = ap.parse_args()
    for c, p in build_all(a.sample_dir, a.out_dir, a.model).items():
        print(c, "->", p)
