"""Main-run analysis, as pre-registered (prereg/PREREGISTRATION.md §7-8).

Input: a local copy of the Volume folder runs/<run>/ (cond/batch/runs.jsonl + traces):
    python -m modal volume get kaggle-replica-data /runs/main_v1 analysis/data/
    python -I analysis/analyze.py analysis/data/main_v1
Output: analysis/results_<run>.md (tables for the paper) + analysis/results_<run>.json.

Unit = task (share of its R runs resolved), paired across conditions. 95% CIs: 10,000 task-level bootstrap
resamples, seed 2026. Paired test: exact sign-flip permutation on per-task differences (all 2^n flips when
n <= 16, else 10,000 random flips). Descriptive; no significance claims.
"""
from __future__ import annotations

import itertools
import json
import random
import statistics as st
import sys
from pathlib import Path

GROUPS = {"L1": ["L1_postcutoff", "L1_silent"], "L2": ["L2_silent_vague"], "L3": ["L3_silent_nospec"],
          "control": ["control"]}
CONDS = ["C0", "C1", "C2", "C3", "C4"]
CONTRASTS = [("C1", "C0"), ("C2", "C0"), ("C3", "C0"), ("C4", "C0"), ("C2", "C1"), ("C3", "C1")]
CAP_S = 890
SEED = 2026


def load(root: Path) -> list[dict]:
    rows = []
    for cond in CONDS:
        for batch in sum(GROUPS.values(), []):
            p = root / cond / batch / "runs.jsonl"
            if p.exists():
                for l in p.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        r = json.loads(l)
                        r["cond"], r["batch"] = cond, batch
                        r["group"] = next(g for g, bs in GROUPS.items() if batch in bs)
                        r["trace_file"] = str(root / cond / batch / cond / f"rep{r['repeat']}" / f"{r['task']}.trace.json")
                        rows.append(r)
    return rows


def trace_order(path: str) -> dict:
    """Was the first look-up (search / docs skill script) before the first repo edit?"""
    p = Path(path)
    if not p.exists():
        return {}
    calls = []
    for s in json.loads(p.read_text(encoding="utf-8")).get("steps", []):
        for tc in s.get("tool_calls") or []:
            calls.append((tc.get("function_name"), json.dumps(tc.get("arguments") or {})))
    look = [i for i, (n, a) in enumerate(calls) if n == "run_skill_script"]
    edit = [i for i, (n, a) in enumerate(calls) if n in ("edit_file", "write_file") and "/tmp/" not in a]
    return dict(first_lookup=look[0] if look else None, first_edit=edit[0] if edit else None,
                lookup_before_edit=bool(look) and (not edit or look[0] < edit[0]))


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def per_task(rows, cond, group):
    out = {}
    for r in rows:
        if r["cond"] == cond and r["group"] == group:
            out.setdefault(r["task"], []).append(bool(r["resolved"]))
    return {t: sum(v) / len(v) for t, v in out.items()}


def boot_ci(diffs, n=10_000):
    rng = random.Random(SEED)
    if not diffs:
        return (None, None)
    ms = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return (round(ms[int(0.025 * n)], 3), round(ms[int(0.975 * n) - 1], 3))


def signflip_p(diffs):
    obs = abs(st.mean(diffs)) if diffs else 0
    if not diffs or obs == 0:
        return 1.0
    if len(diffs) <= 16:
        flips = list(itertools.product([1, -1], repeat=len(diffs)))
    else:
        rng = random.Random(SEED)
        flips = [[rng.choice([1, -1]) for _ in diffs] for _ in range(10_000)]
    hits = sum(abs(st.mean(d * s for d, s in zip(diffs, f))) >= obs - 1e-12 for f in flips)
    return round(hits / len(flips), 4)


def main(root: str) -> None:
    root = Path(root)
    rows = load(root)
    for r in rows:
        r.update(trace_order(r["trace_file"]))
        th = r.get("tool_hist") or {}
        r["repo_edits"] = th.get("edit_file", 0) + th.get("write_file", 0)
        r["lookups"] = (r.get("search_calls") or 0) + (r.get("docs_calls") or 0)
        r["at_cap"] = (r.get("duration_s") or 0) >= CAP_S
        r["ctx_error"] = "ContextWindow" in str(r.get("error"))
        r["timeout"] = "session timeout" in str(r.get("error"))
        r["harness_error"] = bool(r.get("error")) and not (r["ctx_error"] or r["timeout"])
    md = [f"# Main-run results ({root.name})", "", f"{len(rows)} runs loaded.", ""]
    summary = {}
    md += ["## Per condition and group", "",
           "| group | cond | runs | resolved | tasks (mean share) | med tool calls | med prompt tok | med time s | med repo edits | runs with look-up | look-up before 1st edit | at cap | timeout | ctx-err | harness-err |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for c in CONDS:
            R = [r for r in rows if r["group"] == g and r["cond"] == c]
            if not R:
                continue
            pt = per_task(rows, c, g)
            s = dict(runs=len(R), resolved=sum(r["resolved"] for r in R), task_mean=round(st.mean(pt.values()), 3),
                     tool_calls=med([r.get("tool_calls") for r in R]), prompt_tokens=med([r.get("prompt_tokens") for r in R]),
                     duration=med([r.get("duration_s") for r in R]), edits=med([r["repo_edits"] for r in R]),
                     lookup_runs=sum(r["lookups"] > 0 for r in R),
                     lookup_before_edit=sum(bool(r.get("lookup_before_edit")) for r in R),
                     at_cap=sum(r["at_cap"] for r in R), timeout=sum(r["timeout"] for r in R),
                     ctx=sum(r["ctx_error"] for r in R),
                     harness=sum(r["harness_error"] for r in R))
            summary[f"{g}/{c}"] = s
            md.append(f"| {g} | {c} | {s['runs']} | {s['resolved']} | {s['task_mean']} | {s['tool_calls']} | "
                      f"{s['prompt_tokens']} | {s['duration']} | {s['edits']} | {s['lookup_runs']} | "
                      f"{s['lookup_before_edit']} | {s['at_cap']} | {s['timeout']} | {s['ctx']} | {s['harness']} |")
    md += ["", "## Contrasts (per-task resolve share, paired; 95% bootstrap CI; exact sign-flip p)", "",
           "| group | contrast | tasks | mean diff | 95% CI | p |", "|---|---|---|---|---|---|"]
    contrasts = {}
    for g in GROUPS:
        for a, b in CONTRASTS:
            A, B = per_task(rows, a, g), per_task(rows, b, g)
            ts = sorted(set(A) & set(B))
            if not ts:
                continue
            d = [A[t] - B[t] for t in ts]
            ci, p = boot_ci(d), signflip_p(d)
            contrasts[f"{g}/{a}-{b}"] = dict(n=len(ts), mean=round(st.mean(d), 3), ci=ci, p=p)
            md.append(f"| {g} | {a} − {b} | {len(ts)} | {round(st.mean(d), 3)} | {ci} | {p} |")
    md += ["", "## Per task (resolved runs / runs)", "", "| group | task | " + " | ".join(CONDS) + " |",
           "|---|---|" + "---|" * len(CONDS)]
    for g in GROUPS:
        for t in sorted({r["task"] for r in rows if r["group"] == g}):
            cells = []
            for c in CONDS:
                R = [r for r in rows if r["group"] == g and r["cond"] == c and r["task"] == t]
                cells.append(f"{sum(r['resolved'] for r in R)}/{len(R)}" if R else "-")
            md.append(f"| {g} | {t} | " + " | ".join(cells) + " |")
    out = Path(__file__).parent / f"results_{root.name}"
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(dict(summary=summary, contrasts=contrasts), indent=1), encoding="utf-8")
    print("\n".join(md[:60]))
    print(f"\nsaved {out.with_suffix('.md')}")


if __name__ == "__main__":
    main(sys.argv[1])
