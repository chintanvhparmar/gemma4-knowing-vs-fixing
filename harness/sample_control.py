"""Select the control task set from the validated public dev tasks.

Inputs (from the gold-patch-validator notebook output):
  gold_results.jsonl   one line per task: id, repo, resolved (reference fix passes?), ...
  empty_results.jsonl  one line per gold-passing task: id, empty_resolved (passes with NO fix?)
  tasks.jsonl          official dev tasks (for patch files + created_at)

Rules (all reported in the paper):
  1. keep tasks whose reference fix passes the official verification in OUR environment
  2. drop tasks that also pass with an empty patch (trivially solvable)
  3. de-duplicate bug families: same repo AND the reference patches change the same named
     function/class (git hunk-header context) in a shared source file -> keep the
     earliest-created one. Conservative: may also merge distinct bugs in one large function.
  4. stratified random sample by repo (proportional, at least 1 per repo), fixed seed

Usage: python harness/sample_control.py gold_results.jsonl empty_results.jsonl tasks.jsonl \
           --n 30 --seed 2026 --out configs/control_tasks.json
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

FILE_RX = re.compile(r"^\+\+\+ b/(\S+)")
HUNK_RX = re.compile(r"^@@ [^@]*@@\s*(.*)$")
TEST_RX = re.compile(r"(^|/)tests?/|(^|/)test_")
NAME_RX = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)")


def _hunks(patch: str) -> set[tuple[str, str]]:
    """{(source file, enclosing def/class name)} changed by a unified diff.

    Parsed line by line (the dev-set patches have '---/+++' headers but no 'diff --git'
    lines). The name comes from git's hunk-header context. Hunks without a def/class
    context (imports, module-level constants), dunder methods and test files are ignored.
    Line numbers are NOT used: tasks are fixed against snapshots years apart, so the
    same line number can be unrelated code.
    """
    out: set[tuple[str, str]] = set()
    cur = None
    for line in patch.splitlines():
        m = FILE_RX.match(line)
        if m:
            cur = None if TEST_RX.search(m.group(1)) else m.group(1)
            continue
        if line.startswith("--- "):
            continue
        h = HUNK_RX.match(line)
        if h and cur:
            n = NAME_RX.match(h.group(1).strip())
            # dunder methods (__init__, __rich_console__...) are ambiguous without the class
            if n and not n.group(1).startswith("__"):
                out.add((cur, n.group(1)))
    return out


def _overlap(h1: set, h2: set) -> bool:
    """True if both patches change the same named function/class in the same source file."""
    return bool(h1 & h2)


def load_jsonl(p) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("empty")
    ap.add_argument("tasks")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="configs/control_tasks.json")
    a = ap.parse_args()

    gold = {r["id"]: r for r in load_jsonl(a.gold)}
    empty = {r["id"]: r for r in load_jsonl(a.empty)}
    tasks = {t["instance_id"]: t for t in load_jsonl(a.tasks)}

    log = collections.OrderedDict()
    log["total_tasks"] = len(tasks)
    passing = [i for i, r in gold.items() if r.get("resolved")]
    log["reference_fix_passes"] = len(passing)
    trivial = [i for i in passing if empty.get(i, {}).get("empty_resolved")]
    usable = [i for i in passing if i not in trivial and i in tasks]
    log["pass_without_fix_excluded"] = sorted(trivial)
    log["usable_after_filters"] = len(usable)

    # near-duplicate = same repo AND same (source file, function/class) changed
    hunks ={i: _hunks(tasks[i].get("patch", "")) for i in usable}
    parent = {i: i for i in usable}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x_idx, x in enumerate(usable):
        for y in usable[x_idx + 1:]:
            if tasks[x]["repo"] == tasks[y]["repo"] and _overlap(hunks[x], hunks[y]):
                parent[find(x)] = find(y)
    families = collections.defaultdict(list)
    for i in usable:
        families[find(i)].append(i)
    dedup, dropped = [], []
    for members in families.values():
        members.sort(key=lambda i: tasks[i].get("created_at", ""))
        dedup.append(members[0])
        dropped += members[1:]
    log["near_duplicate_families"] = sorted(
        [{"kept": m[0], "dropped": m[1:],
          "shared": sorted({f"{f}:{n}" for x in m for y in m if x != y for f, n in hunks[x] & hunks[y]})}
         for m in families.values() if len(m) > 1], key=lambda d: d["kept"])
    log["near_duplicates_dropped"] = sorted(dropped)
    log["usable_after_dedup"] = len(dedup)

    by_repo = collections.defaultdict(list)
    for i in sorted(dedup):
        by_repo[tasks[i]["repo"]].append(i)
    rng = random.Random(a.seed)
    n = min(a.n, len(dedup))
    quota = {r: max(1, round(n * len(v) / len(dedup))) for r, v in by_repo.items()}
    while sum(quota.values()) > n:  # trim the largest quota first
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        r = max(by_repo, key=lambda r: len(by_repo[r]) - quota[r])
        quota[r] += 1
    chosen = []
    for r, ids in sorted(by_repo.items()):
        chosen += rng.sample(ids, min(quota[r], len(ids)))
    log["pool_by_repo"] = {r: len(v) for r, v in sorted(by_repo.items())}
    log["chosen_by_repo"] = dict(collections.Counter(tasks[i]["repo"] for i in chosen))
    log["seed"] = a.seed

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"control_tasks": sorted(chosen),
                               "created_at": {i: tasks[i].get("created_at") for i in sorted(chosen)},
                               "selection_log": log}, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    print(f"\n{len(chosen)} control tasks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
