"""library-docs skill: offline inspection of installed Python packages.

Commands: version | api | sig | source | grep | changelog  (see SKILL.md)
Prints plain text; output is capped so it fits in the agent's context.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import pkgutil
import re
import sys
from pathlib import Path

MAX = 6000


def out(s: str) -> None:
    print(s if len(s) <= MAX else s[:MAX] + f"\n... [truncated {len(s) - MAX} chars]")


def resolve(dotted: str):
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(f"cannot import {dotted!r}")


def cmd_version(dist: str) -> None:
    try:
        out(f"{dist} {md.version(dist)}")
    except md.PackageNotFoundError:
        out(f"{dist}: not installed (distribution name may differ from the import name)")


def cmd_api(module: str) -> None:
    m = importlib.import_module(module)
    names = getattr(m, "__all__", None) or [n for n in dir(m) if not n.startswith("_")]
    subs = []
    if hasattr(m, "__path__"):
        subs = [x.name for x in pkgutil.iter_modules(m.__path__) if not x.name.startswith("_")]
    out(f"module {module} ({getattr(m, '__file__', '?')})\npublic names: {', '.join(sorted(names))}"
        + (f"\nsubmodules: {', '.join(sorted(subs))}" if subs else ""))


def cmd_sig(dotted: str) -> None:
    obj = resolve(dotted)
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(signature unavailable)"
    out(f"{dotted}{sig}\n\n{inspect.getdoc(obj) or '(no docstring)'}")


def cmd_source(dotted: str) -> None:
    obj = resolve(dotted)
    try:
        lines = inspect.getsource(obj).splitlines()
    except (TypeError, OSError) as e:
        out(f"source unavailable: {e}")
        return
    out(f"# {inspect.getsourcefile(obj)}\n" + "\n".join(lines[:200]))


def cmd_grep(module: str, pattern: str) -> None:
    m = importlib.import_module(module)
    root = Path(m.__file__).parent if hasattr(m, "__path__") else Path(m.__file__)
    files = root.rglob("*.py") if root.is_dir() else [root]
    rx, hits = re.compile(pattern), []
    for f in files:
        try:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{f.relative_to(root.parent)}:{i}: {line.strip()}")
                    if len(hits) >= 60:
                        break
        except OSError:
            continue
        if len(hits) >= 60:
            break
    out("\n".join(hits) or "no matches")


def cmd_changelog(dist: str) -> None:
    try:
        d = md.distribution(dist)
    except md.PackageNotFoundError:
        out(f"{dist}: not installed")
        return
    found = []
    for f in d.files or []:
        if re.search(r"(CHANGE|HISTORY|NEWS|RELEASE)", f.name, re.I):
            p = Path(d.locate_file(f))
            if p.is_file():
                found.append((p, p.read_text(errors="replace")))
    meta = d.metadata.get_payload() if hasattr(d.metadata, "get_payload") else ""
    if found:
        p, txt = found[0]
        out(f"# {p}\n{txt}")
    elif meta and re.search(r"(change|release)", meta, re.I):
        out("# from package metadata (long description)\n" + meta)
    else:
        out(f"no changelog bundled with {dist} {d.version}")


CMDS = {"version": cmd_version, "api": cmd_api, "sig": cmd_sig, "source": cmd_source,
        "grep": cmd_grep, "changelog": cmd_changelog}

def normalize_argv(argv: list[str]) -> list[str]:
    """Accept every form ADK's run_skill_script can produce (google/adk/tools/skill_toolset.py:638-653,
    verified 9 Nov): args as a list (["version", "pandas"] or one string "version pandas"),
    positional_args (ADK inserts "--" first) and args as a dict (--command version --target pandas)."""
    import shlex

    toks = [t for t in argv if t != "--"]
    opts, pos, i = {}, [], 0
    while i < len(toks):
        if toks[i].startswith("--") and i + 1 < len(toks):
            opts[toks[i][2:]] = toks[i + 1]
            i += 2
        else:
            pos.append(toks[i])
            i += 1
    if len(pos) == 1 and " " in pos[0]:
        pos = shlex.split(pos[0])
    cmd = opts.pop("command", None) or opts.pop("cmd", None)
    if cmd is None and pos:
        cmd = pos.pop(0)
    rest = pos + [v for k, v in opts.items()]  # remaining named values in the order given
    return ([cmd] if cmd else []) + rest


if __name__ == "__main__":
    argv = normalize_argv(sys.argv[1:])
    if len(argv) < 2 or argv[0] not in CMDS:
        print(__doc__)
        print(f"(received arguments: {sys.argv[1:]!r}; expected e.g. [\"version\", \"pandas\"])")
        sys.exit(1)
    try:
        CMDS[argv[0]](*argv[1:])
    except Exception as e:  # report errors to the agent as text
        print(f"error: {type(e).__name__}: {e}")
        sys.exit(1)
