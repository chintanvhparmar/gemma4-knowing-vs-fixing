"""EXPLORATORY follow-up (not pre-registered; decided 10 Nov after the main runs).

C1 (web search) on the L3 no-spec tasks with the harness's "Environment is offline" line removed,
to test whether that instruction suppressed search. Same model, sampling, budget, tasks, R=2.

    python -m modal deploy modal_replica/replica.py
    python -I modal_replica/spawn_expl.py start
    python -I modal_replica/spawn_expl.py status
"""
from __future__ import annotations

import json
import pathlib
import sys

import modal

RUN = "expl_nooffline_v1"
STORE = pathlib.Path(__file__).resolve().parent / "env" / f"expl_calls_{RUN}.json"

if sys.argv[1] == "start":
    f = modal.Function.from_name("kaggle-replica", "_condition_runs")
    fc = f.spawn(cond="C1", run_name=RUN, repeats=2, concurrency=3, search_budget=200,
                 batches="L3_silent_nospec", drop_offline=True)
    STORE.write_text(json.dumps({"C1": fc.object_id}))
    print("started", fc.object_id)
else:
    cid = json.loads(STORE.read_text())["C1"]
    try:
        print("FINISHED", modal.FunctionCall.from_id(cid).get(timeout=0))
    except TimeoutError:
        print("running")
