"""Start / check the MAIN runs on the DEPLOYED Modal app (jobs keep running if this laptop process dies).

    python -m modal deploy modal_replica/replica.py                     # once (and after code changes)
    python -I modal_replica/spawn_main.py start  --run-name main_v1 [--conds C0,C1,C2,C3,C4]
    python -I modal_replica/spawn_main.py status --run-name main_v1     # finished / running per condition

Call IDs are stored in modal_replica/env/main_calls_<run>.json. Re-starting the same run name resumes
(run_batch skips finished runs). Search budget per web-search condition: 1440 (= 7,200 / 5).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import modal

HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "status"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--conds", default="C0,C1,C2,C3,C4")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    store = HERE / "env" / f"main_calls_{a.run_name}.json"
    if a.action == "start":
        f = modal.Function.from_name("kaggle-replica", "_condition_runs")
        calls = {}
        for c in [x.strip() for x in a.conds.split(",") if x.strip()]:
            fc = f.spawn(cond=c, run_name=a.run_name, repeats=a.repeats)
            calls[c] = fc.object_id
            print(f"started {c}: {fc.object_id}")
        old = json.loads(store.read_text()) if store.exists() else {}
        store.write_text(json.dumps({**old, **calls}, indent=1))
        print("saved", store)
    else:
        calls = json.loads(store.read_text())
        for c, cid in calls.items():
            fc = modal.FunctionCall.from_id(cid)
            try:
                r = fc.get(timeout=0)
                print(f"{c}: FINISHED {r}")
            except TimeoutError:
                print(f"{c}: running")
            except Exception as e:
                print(f"{c}: ERROR {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
