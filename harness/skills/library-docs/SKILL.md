---
name: library-docs
description: Inspect the Python libraries actually installed in this environment (version, public API, signatures, docstrings, source, bundled changelog). Works offline.
---

# library-docs

Use this skill to check how an installed library really works **in this environment**,
instead of relying on memory. Everything runs offline against the installed packages.

Run it with `run_skill_script` (`skill_name`: `library-docs`, `file_path`: `scripts/pkg_info.py`),
passing one of these commands as `args`, e.g. `args: ["version", "pandas"]`:

| Command | What it prints |
|---|---|
| `version <distribution>` | Installed version, e.g. `version pandas` |
| `api <module>` | Public names in a module, e.g. `api marshmallow` |
| `sig <dotted.name>` | Signature and docstring, e.g. `sig marshmallow.validates` |
| `source <dotted.name>` | Source code of a function or class (first 200 lines) |
| `grep <module> <regex>` | Matching lines in the installed package source (max 60) |
| `changelog <distribution>` | Changelog / release notes bundled with the package, if any |

Tip: check the installed `version` first; then `sig` or `source` for the API you plan to use.
