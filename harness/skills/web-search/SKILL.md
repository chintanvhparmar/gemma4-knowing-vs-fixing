---
name: web-search
description: Search the web (Tavily) for documentation, changelogs, migration guides or error messages. Requires internet access.
---

# web-search

Use this skill to look something up on the web.

Run it with `run_skill_script` (`skill_name`: `web-search`, `file_path`: `scripts/web_search.py`),
passing the query (and optionally max_results) as `args`, e.g.
`args: ["pandas 3.0 datetime resolution change", "5"]`.

It prints up to `max_results` (default 5, max 8) results: title, URL and a text snippet.

Tips:
- Include the library name and, if known, the installed version in the query.
- Prefer official documentation, changelogs and release notes over blog posts.
