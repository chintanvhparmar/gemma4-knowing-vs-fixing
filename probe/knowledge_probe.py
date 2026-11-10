"""Knowledge probe: does Gemma 4 know about post-cutoff library API changes?

For each candidate change (research/postcutoff_candidates.md) we ask a neutral
coding question that naturally elicits the affected API, without hinting that it
changed, and classify the answer with regexes:
  OLD     -> writes the old usage (fails on the new version): model is unaware -> good task
  NEW     -> uses the new API: model already knows -> drop
  SAFE    -> uses an alternative that works on both versions: weak trap
  MIXED   -> both old and new patterns appear
  UNCLEAR -> none matched; inspect manually
We also ask each library's "latest version you know" (knowledge horizon).

Endpoint: Nebius Token Factory (OpenAI-compatible). Key from env NEBIUS_API_KEY
(never printed or logged). Base URL from env NEBIUS_BASE_URL or the default below.
Only token COUNTS are logged; prompts and answers are stored for the paper's
appendix (they contain no personal data).

Usage:
  Token Factory:  python probe/knowledge_probe.py [--model MODEL_ID] [--out DIR]
  Kaggle / vLLM:  python probe/knowledge_probe.py --base-url http://127.0.0.1:8000/v1 \
                      --key-env LOCAL_API_KEY --model <served model id>
Stdlib only (urllib), so it runs on any Python 3.9+.

Note (8 Nov 2026): the Token Factory key's /models list has no Gemma 4 (only
google/gemma-3-27b-it); every Gemma 4 id returned 404. Plan: run this probe
against the 4-bit model on Kaggle's local vLLM (the exact contest model).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")
SYSTEM = "You are a helpful Python expert. Answer concisely."

CODE = " Output only the Python code, no explanation."

# id, library, version that made the change, prompt, regex lists
PROBES = [
    dict(id="marshmallow_validates_data_key", lib="marshmallow", ver="4.0.0",
         prompt="Using marshmallow, write a Schema `UserSchema` with an `age` Integer field and a "
                "validation method decorated with `@validates('age')` that raises ValidationError if age < 0." + CODE,
         old=[r"def \w+\(\s*self\s*,\s*\w+\s*\)\s*(->[^:]+)?:"], new=[r"data_key", r"def \w+\(\s*self\s*,\s*\w+\s*,\s*\*\*\w+"], safe=[]),
    dict(id="marshmallow_pass_collection", lib="marshmallow", ver="4.0.0",
         prompt="In marshmallow, write a `@post_load` hook on a schema that receives the whole list of items "
                "at once when the schema is loaded with many=True." + CODE,
         old=[r"pass_many\s*=\s*True"], new=[r"pass_collection"], safe=[]),
    dict(id="polars_cast_str_to_date", lib="polars", ver="2.0.0",
         prompt="In polars, convert the String column 'd' (values like '2024-01-31') of DataFrame `df` to the Date dtype." + CODE,
         old=[r"cast\(\s*pl\.Date"], new=[r"str\.to_date"], safe=[r"str\.strptime"]),
    dict(id="polars_streaming_engine", lib="polars", ver="2.0.0",
         prompt="In polars, collect the LazyFrame `lf` using the streaming engine." + CODE,
         old=[r"streaming\s*=\s*True"], new=[r"engine\s*=\s*['\"]streaming"], safe=[]),
    dict(id="polars_explode_empty", lib="polars", ver="2.0.0",
         prompt="In polars, what happens to rows whose list is empty when you call `df.explode('col')` with default arguments? "
                "Answer in one sentence.",
         old=[r"\bnull\b", r"\bNone\b"], new=[r"(dropped|removed|disappear|no row|omitted|excluded)"], safe=[]),
    dict(id="pandas_datetime_unit_us", lib="pandas", ver="3.0.0",
         prompt="Given a pandas Series `s` of ISO date strings, compute Unix epoch seconds as int64 (parse with pd.to_datetime first)." + CODE,
         old=[r"10\s*\*\*\s*9", r"1e9", r"1_000_000_000", r"1000000000"], new=[r"as_unit"], safe=[r"Timedelta\(\s*['\"]1\s*s", r"timestamp\(\)"]),
    dict(id="pandas_value_counts_sort_false", lib="pandas", ver="3.0.0",
         prompt="In pandas, in what order does `DataFrame.value_counts(sort=False)` return its rows? Answer in one sentence.",
         old=[r"sort(ed)? by (the )?(index|label|values|group)", r"(index|label|group(ing)? key)s? order", r"lexicograph"],
         new=[r"(first (appear|occur)|order of appearance|input order|original order|order in which)"], safe=[]),
    dict(id="pandas_zoneinfo_default", lib="pandas", ver="3.0.0",
         prompt="Given a timezone-aware pandas Timestamp `ts` (created with tz='Europe/Berlin'), get the IANA timezone name string "
                "(e.g. 'Europe/Berlin') from `ts.tz`." + CODE,
         old=[r"\.zone\b"], new=[r"\.key\b"], safe=[r"str\(\s*ts\.tz\s*\)", r"tzname"]),
    dict(id="click_cli_runner_mix_stderr", lib="click", ver="8.2.0",
         prompt="Write a pytest test for a click command `cli` that checks stdout and stderr SEPARATELY using click.testing.CliRunner." + CODE,
         old=[r"mix_stderr\s*=\s*False"], new=[], safe=[r"result\.stderr"]),
    dict(id="click_command_suffix", lib="click", ver="8.2.0",
         prompt="In click, I decorate a function named `init_cmd` with `@cli.command()` and give no explicit name. "
                "What command name do users type on the command line? Answer with the name only.",
         old=[r"init-cmd", r"init_cmd"], new=[r"^\W*init\W*$", r"`init`"], safe=[]),
    dict(id="scipy_sqrtm_disp", lib="scipy", ver="1.18.0",
         prompt="Compute the matrix square root of a NumPy array `A` with scipy.linalg, and also obtain scipy's error estimate." + CODE,
         old=[r"disp\s*=\s*False"], new=[], safe=[]),
    dict(id="scipy_rankdata_dtype", lib="scipy", ver="1.18.0",
         prompt="What dtype does `scipy.stats.rankdata([3, 1, 2], method='ordinal')` return? Answer with the dtype only.",
         old=[r"\bint"], new=[r"float"], safe=[]),
    dict(id="scipy_sph_harm", lib="scipy", ver="1.17.0",
         prompt="Compute the spherical harmonic Y_l^m for l=2, m=1 at polar angle theta and azimuth phi using scipy.special." + CODE,
         old=[r"sph_harm\((?!_)"], new=[r"sph_harm_y"], safe=[]),
    dict(id="numpy_eig_complex", lib="numpy", ver="2.5.0",
         prompt="If I call `numpy.linalg.eig` on a real 2x2 matrix whose eigenvalues are all real, what is the dtype of the "
                "returned eigenvalues array? Answer with the dtype only.",
         old=[r"float"], new=[r"complex"], safe=[]),
    dict(id="sqlalchemy_filter_by_join", lib="sqlalchemy", ver="2.1.0",
         prompt="SQLAlchemy 2.x ORM: build a select of `User` joined to `Address` (User.addresses relationship) keeping only the "
                "user whose primary key `id` is 5. Use the most concise style." + CODE,
         old=[r"filter_by\(\s*id\s*="], new=[], safe=[r"where\(\s*User\.id", r"filter\(\s*User\.id"]),
    dict(id="fastapi_strict_content_type", lib="fastapi", ver="0.132.0",
         prompt="A legacy client POSTs JSON bodies to my FastAPI endpoint WITHOUT any Content-Type header. "
                "Show the minimal FastAPI code so the endpoint still parses the JSON body into a Pydantic model." + CODE,
         old=[], new=[r"strict_content_type"], safe=[r"Request\b.*\.json\(\)|await request\.(body|json)"]),
    dict(id="sklearn_adaboost_algorithm", lib="scikit-learn", ver="1.8.0",
         prompt="Create a scikit-learn AdaBoostClassifier that uses the SAMME algorithm with 100 estimators." + CODE,
         old=[r"algorithm\s*="], new=[], safe=[]),
    dict(id="starlette_lifespan", lib="starlette", ver="1.0.0",
         prompt="Write a minimal Starlette (not FastAPI) app that connects to a database at startup and disconnects at shutdown." + CODE,
         old=[r"on_startup", r"on_shutdown", r"on_event", r"add_event_handler"], new=[r"lifespan"], safe=[]),
    dict(id="django_checkconstraint_condition", lib="django", ver="6.0",
         prompt="Django: in a model's Meta, add a database CheckConstraint ensuring the `price` field is >= 0." + CODE,
         old=[r"check\s*=\s*(models\.)?Q"], new=[r"condition\s*="], safe=[]),
    dict(id="langchain_llmchain", lib="langchain", ver="1.0.0",
         prompt="LangChain (Python): build a simple chain that fills a prompt template with a `topic` and calls a chat model." + CODE,
         old=[r"LLMChain"], new=[r"langchain_classic"], safe=[r"\|\s*\w*llm|\|\s*model|prompt\s*\|"]),
]

# Round 2 (9 Nov 2026). Every change checked against a primary source before adding:
#   networkx 3.6 (2025-11-24): node_link_data default edges="edges" (3.4.2 default key "links",
#     with a FutureWarning already in 3.4.2 = before the cutoff)  [node_link.py at both tags]
#   marshmallow 4.0.0: Schema.context removed (deprecated 3.24.0, 2025-01-06)  [CHANGELOG.rst]
#   click 8.2.0: help shown by no_args_is_help exits with code 2, not 0  [CHANGES.rst]
#   numpy 2.4.0 (2025-12-20): float() of an ndim>0 array raises TypeError (deprecated 1.25)  [2.4.0-notes]
#   matplotlib 3.11.0 (2026-06-12): module-level cm.get_cmap removed (deprecated 3.7)  [cm.py tags]
#   setuptools 82.0.0 (2026-02-08): pkg_resources removed  [NEWS.rst]
#   scikit-learn 1.8.0: LogisticRegression(multi_class=) removed ("deprecated" in 1.6.0)  [_logistic.py]
#   pandas 3.0.0: offsets.Day is a calendar day across DST (was fixed 24 h)  [whatsnew v3.0.0]
PROBES2 = [
    dict(id="networkx_node_link_edges", lib="networkx", ver="3.6",
         prompt="Using networkx, convert graph `G` to a JSON-serialisable dict with `node_link_data`, then return the "
                "list of edge dicts from that result." + CODE,
         old=[r"\[\s*['\"]links['\"]\s*\]"], new=[r"\[\s*['\"]edges['\"]\s*\]"], safe=[r"edges\s*=\s*['\"]"]),
    dict(id="marshmallow_schema_context", lib="marshmallow", ver="4.0.0",
         prompt="In marshmallow, a `fields.Method` field must use the current request's user (not stored on the object) "
                "while dumping. Show how to pass the user into the schema and read it in the method." + CODE,
         old=[r"context\s*=", r"self\.context"], new=[r"ContextVar", r"experimental\.context", r"\bContext\("], safe=[]),
    dict(id="click_no_args_is_help_exit", lib="click", ver="8.2.0",
         prompt="A click group `cli` is run with no arguments, so click prints the help page because of "
                "`no_args_is_help`. What process exit code does it return? Answer with the number only.",
         old=[r"^\W*0\W*$"], new=[r"^\W*2\W*$"], safe=[]),
    dict(id="numpy_float_ndim1", lib="numpy", ver="2.4.0",
         prompt="Write a Python function `to_float(a)` that takes a NumPy array `a` of shape (1,) and returns its value "
                "as a Python float." + CODE,
         # (?<![\w.]) so the function name `to_float(a)` itself doesn't count (caught by the offline test)
         old=[r"(?<![\w.])float\(\s*a\s*\)"], new=[], safe=[r"\.item\(\)", r"a\[0\]", r"\.flat\["]),
    dict(id="matplotlib_cm_get_cmap", lib="matplotlib", ver="3.11.0",
         prompt="In matplotlib, get the 'viridis' colormap object and sample 5 evenly spaced RGBA colours from it." + CODE,
         old=[r"cm\.get_cmap"], new=[], safe=[r"colormaps\s*\[", r"colormaps\.get_cmap", r"(plt|pyplot)\.get_cmap"]),
    dict(id="setuptools_pkg_resources", lib="setuptools", ver="82.0.0",
         prompt="In Python, get the installed version string of the distribution 'requests' at runtime." + CODE,
         old=[r"pkg_resources"], new=[], safe=[r"importlib\.metadata", r"importlib_metadata"]),
    dict(id="sklearn_logreg_multi_class", lib="scikit-learn", ver="1.8.0",
         prompt="Create a scikit-learn LogisticRegression that does multinomial (softmax) multiclass classification "
                "with the lbfgs solver." + CODE,
         old=[r"multi_class\s*="], new=[], safe=[]),
    dict(id="pandas_day_dst", lib="pandas", ver="3.0.0",
         prompt="In pandas: ts = pd.Timestamp('2025-03-08 08:00', tz='US/Eastern'). What is ts + pd.offsets.Day(1)? "
                "Answer with the resulting Timestamp only.",
         old=[r"09:00"], new=[r"08:00"], safe=[]),
]

# Round 3 (10 Nov): SILENT behaviour changes (no error, no warning on the new version), found by a
# research pass with changelog quotes + old/new runs (re-verify on cp312 before building):
#   pandas 3.0.0 (2026-01-21): groupby(observed=False), several keys: unobserved prod/any/all -> 1/False/True (was NaN)  [whatsnew v3.0.0, GH 55738]
#   scikit-learn 1.8.0 (2025-12-10): median_absolute_error with sample_weight uses averaged weighted percentile  [whats_new v1.8, #30787]
#   polars 2.0.0 (2026-10-06): read_csv(schema=...) matched by name, not position  [docs.pola.rs/releases/upgrade/2]
#   click 8.2.0+ (2025-05-10): Option.envvar with flag_value -> flag_value / None instead of the env value  [click changes 8.2.0, 8.2.2]
#   pandas 3.0.0: Float64 arithmetic that produced NaN now produces NA  [whatsnew v3.0.0]
PROBES3 = [
    dict(id="pandas_groupby_observed_multikey_prod", lib="pandas", ver="3.0.0",
         prompt="In pandas: df = pd.DataFrame({'k1': pd.Categorical(list('aabb'), categories=list('abc')), "
                "'k2': [1, 1, 1, 2], 'v': [2, 3, 4, 5]}). What does "
                "`df.groupby(['k1', 'k2'], observed=False)['v'].prod().tolist()` return? Answer with the list only.",
         old=[r"nan", r"NaN"], new=[r"\[\s*6\s*,\s*1\s*,\s*4\s*,\s*5\s*,\s*1\s*,\s*1\s*\]"], safe=[]),
    dict(id="sklearn_median_ae_weighted", lib="scikit-learn", ver="1.8.0",
         prompt="What does `sklearn.metrics.median_absolute_error([0, 0, 0, 0], [1, 2, 3, 4], sample_weight=[1, 1, 1, 1])` "
                "return? Answer with the number only.",
         old=[r"^\W*2(\.0+)?\W*$"], new=[r"^\W*2\.50*\W*$"], safe=[]),
    dict(id="polars_read_csv_schema_by_name", lib="polars", ver="2.0.0",
         prompt="In polars, a CSV file has the header `a,b`. You call `pl.read_csv(path, schema={'b': pl.String, 'a': pl.String})`. "
                "Are the schema entries matched to the file's columns by position or by name? Answer with one word.",
         old=[r"position"], new=[r"\bname\b"], safe=[]),
    dict(id="click_flag_value_envvar", lib="click", ver="8.2.0",
         prompt="A click command has `@click.option('--mode', is_flag=True, flag_value='fast', envvar='MODE')`. It is run "
                "with the environment variable MODE=slow and without `--mode`. What value does the `mode` parameter receive? "
                "Answer with the value only.",
         old=[r"slow"], new=[r"fast", r"None", r"False"], safe=[]),
    dict(id="pandas_float64_nan_is_na", lib="pandas", ver="3.0.0",
         prompt="In pandas: s = pd.Series([0.0, 1.0], dtype='Float64'); r = s / 0. What does `r.isna().tolist()` return? "
                "Answer with the list only.",
         old=[r"\[\s*False\s*,\s*False\s*\]"], new=[r"\[\s*True\s*,\s*False\s*\]"], safe=[]),
]

PROBE_SETS = {"1": PROBES, "2": PROBES2, "3": PROBES3}
LIBS = sorted({p["lib"] for p in PROBES})


def http(method: str, path: str, key: str, body: dict | None = None, timeout: int = 120) -> dict:
    req = urllib.request.Request(BASE_URL + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


MAX_TOKENS = 8192  # set from --max-tokens; Gemma 4 may think first, so 700 could truncate answers
# Gemma 4's chat template has `enable_thinking`, default false (chat_template.jinja line 186,
# checked 9 Nov). Smoke test with the default: reasoning_tokens = 0. --thinking turns it on.
THINKING = False


def chat(key: str, model: str, prompt: str) -> tuple[str, dict]:
    """Returns (answer text, info). info = usage + finish_reason + reasoning_chars, so a
    truncated or thinking-only answer is visible instead of silently labelled UNCLEAR."""
    body = {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    if THINKING:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    for attempt in range(3):
        try:
            r = http("POST", "/chat/completions", key, body, timeout=600)
            ch = r["choices"][0]
            msg = ch["message"]
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            usage = r.get("usage") or {}
            info = dict(usage, finish_reason=ch.get("finish_reason"), reasoning_chars=len(reasoning),
                        reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            return (msg.get("content") or ""), info
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None


def classify(text: str, p: dict) -> str:
    hit = lambda pats: any(re.search(x, text, re.I | re.M) for x in pats)
    o, n, s = hit(p["old"]), hit(p["new"]), hit(p["safe"])
    if o and n:
        return "MIXED"
    if n:
        return "NEW"
    if o:
        return "OLD"
    if s:
        return "SAFE"
    return "UNCLEAR"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="model id; default = auto-detect Gemma 4 31B")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (default: Token Factory)")
    ap.add_argument("--key-env", default="NEBIUS_API_KEY", help="env var holding the API key")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: only the first N probes, 1 horizon question")
    ap.add_argument("--label", default="", help="free text for the report header, e.g. 'vLLM 0.33.0, L40S, QAT w4a16'")
    ap.add_argument("--thinking", action="store_true", help="send chat_template_kwargs enable_thinking=true")
    ap.add_argument("--set", default="1", choices=sorted(PROBE_SETS), help="probe set: 1 = round 1, 2 = round 2")
    a = ap.parse_args()

    global BASE_URL, MAX_TOKENS, THINKING
    MAX_TOKENS = a.max_tokens
    THINKING = a.thinking
    if a.base_url:
        BASE_URL = a.base_url.rstrip("/")
    key = os.environ.get(a.key_env) or ("EMPTY" if a.base_url else None)
    if not key:
        print(f"{a.key_env} is not set", file=sys.stderr)
        return 2

    model = a.model
    if not model:
        ids = [m["id"] for m in http("GET", "/models", key).get("data", [])]
        cands = [i for i in ids if re.search(r"gemma-?4", i, re.I) and re.search(r"31b", i, re.I)] or \
                ([ids[0]] if a.base_url and len(ids) == 1 else [])
        if not cands:
            print("No Gemma 4 31B model found. Gemma-like ids:", [i for i in ids if "gemma" in i.lower()], file=sys.stderr)
            return 3
        model = sorted(cands, key=len)[0]
    print(f"Model: {model}   Base URL: {BASE_URL}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    stamp += f"_set{a.set}" if a.set != "1" else ""
    stamp += "_think" if THINKING else "_nothink"  # two runs in the same minute must not overwrite
    jpath = out / f"probe_{stamp}.jsonl"
    tot_in = tot_out = 0
    rows = []

    all_probes = PROBE_SETS[a.set]
    all_libs = sorted({p["lib"] for p in all_probes})
    libs, probes = (all_libs[:1], all_probes[:a.limit]) if a.limit else (all_libs, all_probes)
    with jpath.open("w", encoding="utf-8") as f:
        for lib in libs:
            q = f"What is the most recent released version of the Python library `{lib}` that you know of? Answer with the version number only."
            ans, u = chat(key, model, q)
            tot_in += u.get("prompt_tokens", 0); tot_out += u.get("completion_tokens", 0)
            rec = dict(kind="horizon", lib=lib, answer=ans.strip()[:200], usage=u)
            f.write(json.dumps(rec) + "\n")
            print(f"[horizon] {lib:14} -> {ans.strip()[:60]}")
        for p in probes:
            ans, u = chat(key, model, p["prompt"])
            tot_in += u.get("prompt_tokens", 0); tot_out += u.get("completion_tokens", 0)
            label = classify(ans, p)
            rows.append((p, label, u))
            f.write(json.dumps(dict(kind="probe", id=p["id"], lib=p["lib"], change_version=p["ver"],
                                    label=label, prompt=p["prompt"], answer=ans, usage=u)) + "\n")
            print(f"[probe]   {p['id']:34} {label:8} finish={u.get('finish_reason')} "
                  f"out_tok={u.get('completion_tokens')} reasoning_tok={u.get('reasoning_tokens')} "
                  f"reasoning_chars={u.get('reasoning_chars')}")

    md = out / f"probe_{stamp}.md"
    with md.open("w", encoding="utf-8") as f:
        f.write(f"# Knowledge probe {stamp}\n\nModel: `{model}` at `{BASE_URL}` {a.label}; "
                f"temperature 0, max_tokens {MAX_TOKENS}, enable_thinking={THINKING}.\n\n")
        f.write("| id | library | change ver | label | finish | out tokens |\n|---|---|---|---|---|---|\n")
        for p, label, u in rows:
            f.write(f"| {p['id']} | {p['lib']} | {p['ver']} | {label} | {u.get('finish_reason')} | "
                    f"{u.get('completion_tokens')} |\n")
        f.write(f"\nTokens: {tot_in} prompt + {tot_out} completion.\n")
    counts = {k: sum(1 for _, l, _u in rows if l == k) for k in ("OLD", "NEW", "SAFE", "MIXED", "UNCLEAR")}
    print(f"\nSummary: {counts}\nTokens: {tot_in} prompt + {tot_out} completion\nSaved: {jpath}\n       {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
