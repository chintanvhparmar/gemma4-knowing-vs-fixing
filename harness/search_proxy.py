"""Local web-search relay for the `web-search` skill (C1/C2/C4 only; never on Kaggle L4).

Why (verified 9 Nov in adk_eval_core/sandbox/subprocess_sandbox.py:24-54): the official sandbox
passes only _SAFE_ENV_KEYS (PATH, LANG, LC_ALL, TERM, USER, LOGNAME, SYSTEMROOT, LD_LIBRARY_PATH,
TMPDIR, TEST_TMPDIR) + HOME/TMPDIR/PYTHONNOUSERSITE to everything it runs, so TAVILY_API_KEY never
reaches the skill script. Instead of changing the official sandbox, this relay runs next to the
harness on 127.0.0.1 and holds the key; the skill script posts its query here.
  * the key never enters the sandbox (the agent cannot read, print or leak it)
  * one place enforces the data-security limits: query <= 200 chars, <= 8 results
  * a hard budget on the total number of searches (cost cap)
  * every request is logged (time, query, n, result URLs, status) for the paper

    python harness/search_proxy.py --port 8765 --log /data/search_log.jsonl --budget 7200
Key: env TAVILY_API_KEY of the relay process (never printed or logged).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("SEARCH_UPSTREAM_URL", "https://api.tavily.com/search")  # override only in tests
MAX_QUERY, MAX_N = 200, 8


class Relay:
    def __init__(self, key: str, log_path: str, budget: int):
        self.key, self.log_path, self.budget = key, log_path, budget
        self.used = 0
        self.lock = threading.Lock()

    def log(self, rec: dict) -> None:
        with self.lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def search(self, query: str, n: int) -> tuple[int, dict]:
        query = " ".join(str(query).split())[:MAX_QUERY]
        n = max(1, min(MAX_N, int(n)))
        rec = dict(t=time.time(), query=query, n=n)
        if not query:
            return 400, {"error": "empty query"}
        with self.lock:
            if self.used >= self.budget:
                rec["status"] = "budget_exhausted"
                over = True
            else:
                self.used += 1
                over = False
        if over:
            self.log(rec)
            return 429, {"error": "search budget exhausted"}
        body = json.dumps({"query": query, "max_results": n, "search_depth": "basic"}).encode()
        req = urllib.request.Request(UPSTREAM, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            rec["status"] = f"upstream_http_{e.code}"
            self.log(rec)
            return 502, {"error": f"search service error {e.code}"}
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            rec["status"] = f"upstream_{type(e).__name__}"
            self.log(rec)
            return 502, {"error": "search service unreachable"}
        results = [dict(title=x.get("title", ""), url=x.get("url", ""), content=x.get("content", ""))
                   for x in (data.get("results") or [])[:n]]
        rec.update(status="ok", urls=[x["url"] for x in results], used=self.used)
        self.log(rec)
        return 200, {"results": results}


def make_handler(relay: Relay):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/search":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                code, out = relay.search(req.get("query", ""), req.get("n", 5))
            except (ValueError, TypeError):
                code, out = 400, {"error": "bad request"}
            self._send(code, out)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "used": relay.used, "budget": relay.budget})
            self._send(404, {"error": "not found"})

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):  # no access log (queries are logged by Relay, the key never is)
            pass
    return H


def serve(port: int, log_path: str, budget: int) -> ThreadingHTTPServer:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise SystemExit("TAVILY_API_KEY is not set for the relay process")
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Relay(key, log_path, budget)))
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default="/tmp/search_log.jsonl")
    ap.add_argument("--budget", type=int, default=7200)  # 80% of the account's 9,000 credits (9 Nov)
    a = ap.parse_args()
    s = serve(a.port, a.log, a.budget)
    print(f"search relay on 127.0.0.1:{a.port} (budget {a.budget})", flush=True)
    s.serve_forever()
