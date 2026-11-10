"""web-search skill: web search (Tavily, via the local relay), printed as plain text for the agent.

Usage: web_search.py "<query>" [max_results]
The script holds no key: it posts the query to the search relay on 127.0.0.1:8765
(harness/search_proxy.py), because the official sandbox passes no API keys to scripts
(adk_eval_core _SAFE_ENV_KEYS, verified 9 Nov). The relay caps the query length, enforces
the search budget and logs every query for the paper.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

SNIPPET = 600
RELAY = "http://127.0.0.1:8765/search"  # fixed: the sandbox strips environment variables


def parse_args(argv: list[str]) -> tuple[str, int]:
    """Accept every form ADK's run_skill_script can produce (google/adk/tools/skill_toolset.py:638-653,
    verified 9 Nov): a list (["<query>", "5"] or the query split into words), positional_args (ADK
    inserts "--" first) and a dict (--query "<query>" --max_results 5)."""
    toks = [t for t in argv if t != "--"]
    opts, pos, i = {}, [], 0
    while i < len(toks):
        if toks[i].startswith("--") and i + 1 < len(toks):
            opts[toks[i][2:]] = toks[i + 1]
            i += 2
        else:
            pos.append(toks[i])
            i += 1
    n_raw = opts.get("max_results") or opts.get("n")
    if n_raw is None and len(pos) > 1 and pos[-1].isdigit():
        n_raw = pos.pop()
    query = opts.get("query") or opts.get("q") or " ".join(pos)
    try:
        n = int(n_raw) if n_raw is not None else 5
    except ValueError:
        n = 5
    return query, max(1, min(8, n))


def main() -> int:
    query, n = parse_args(sys.argv[1:])
    if not query.strip():
        print(__doc__)
        print(f"(received arguments: {sys.argv[1:]!r}; expected e.g. [\"pandas 3.0 datetime unit change\"])")
        return 1
    # Cap the query length: searches need keywords, and the cap prevents sending
    # large chunks of task code/text to a third-party API (data-security rule 2.4.b).
    query = " ".join(query.split())[:200]
    body = json.dumps({"query": query, "n": n}).encode()
    req = urllib.request.Request(RELAY, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error", "")
        except ValueError:
            msg = ""
        print(f"error: web search failed ({msg or e.code})")
        return 1
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"error: web search unavailable ({type(e).__name__}); you may be offline")
        return 1
    results = data.get("results", [])[:n]
    if not results:
        print("no results")
        return 0
    for i, x in enumerate(results, 1):
        snippet = (x.get("content") or "").replace("\n", " ")[:SNIPPET]
        print(f"[{i}] {x.get('title', '')}\n    {x.get('url', '')}\n    {snippet}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
