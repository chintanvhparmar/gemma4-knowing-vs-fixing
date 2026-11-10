"""Compare two gold_check.py outputs (e.g. Kaggle GPU notebook vs Modal replica).

Equivalent per (group, mode, task) = same `resolved`, same `exit_code`, same test counts
(passed / failed / errors / skipped / xfailed / xpassed; the duration and warning count are
reported but not required to match).

    python -I harness/compare_gold.py <a.jsonl> <b.jsonl>
"""
from __future__ import annotations

import json
import re
import sys


def load(p):
    lines = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    head = lines[0] if lines and lines[0].get("header") else {}
    return head, {(r["group"], r["mode"], r["id"]): r for r in lines if not r.get("header")}


def counts(summary: str) -> dict:
    c = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)", summary)}
    if "error" in c:
        c["errors"] = c.pop("error") + c.get("errors", 0)
    return c


def warnings(summary: str) -> int:
    m = re.search(r"(\d+) warnings?", summary)
    return int(m.group(1)) if m else 0


def main(a, b):
    ha, ra = load(a)
    hb, rb = load(b)
    print(f"A: {ha.get('platform')} python {ha.get('python')} pkgs {ha.get('n_packages')}")
    print(f"B: {hb.get('platform')} python {hb.get('python')} pkgs {hb.get('n_packages')}")
    vd = {k: (ha.get('versions', {}).get(k), hb.get('versions', {}).get(k))
          for k in set(ha.get("versions", {})) | set(hb.get("versions", {}))
          if ha.get("versions", {}).get(k) != hb.get("versions", {}).get(k)}
    print("key version differences:", vd or "none")
    keys = sorted(set(ra) | set(rb))
    same, diff, missing, warn_diff = 0, [], [], []
    for k in keys:
        if k not in ra or k not in rb:
            missing.append(k)
            continue
        x, y = ra[k], rb[k]
        if (x["resolved"], x["exit_code"], counts(x["summary"])) == (y["resolved"], y["exit_code"], counts(y["summary"])):
            same += 1
            if warnings(x["summary"]) != warnings(y["summary"]):
                warn_diff.append(k)
        else:
            diff.append(k)
    print(f"\nidentical outcome: {same}/{len(keys)} | different: {len(diff)} | only on one side: {len(missing)} | "
          f"same outcome but different warning count: {len(warn_diff)}")
    for k in diff:
        print(f"  DIFF {'/'.join(k)}:\n     A resolved={ra[k]['resolved']} exit={ra[k]['exit_code']} {ra[k]['summary']}"
              f"\n     B resolved={rb[k]['resolved']} exit={rb[k]['exit_code']} {rb[k]['summary']}")
    for k in missing:
        print(f"  MISSING {'/'.join(k)} in {'B' if k in ra else 'A'}")
    for k in warn_diff:
        print(f"  warnings {'/'.join(k)}: A {warnings(ra[k]['summary'])} | B {warnings(rb[k]['summary'])}")
    print("\nEQUIVALENT:", not diff and not missing)
    return not diff and not missing


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
