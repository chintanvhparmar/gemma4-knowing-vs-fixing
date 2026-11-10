# Post-cutoff API change candidates (research, 8 Nov 2026)

Purpose: source list for Paper 1's post-cutoff task set (PLAN.md §5.3).
Model cutoff: January 2025 (Gemma 4 model card). Release dates come from the
PyPI JSON API; changes were checked against official changelogs, migration
guides or tagged source. **Nothing has been executed yet.** Every failure mode
must be confirmed by our validator (old usage fails, reference fix passes)
and by a knowledge probe on the 4-bit model.

⚠ = deprecated with warnings before Jan 2025 (the model may know the new way):
rank lower and check with the probe.

## Ranked Top 20 (max 3 per library)

| Rank | Library / version (release) | Change (old → new) | Old usage fails how | First announced | Notes |
|---|---|---|---|---|---|
| 1 | marshmallow 4.0.0 (2025-04-17) | `@validates` methods now receive a `data_key=` kwarg | `TypeError` for classic `def validate_x(self, value)` | 4.0.0 | pure Python |
| 2 | polars 2.0.0 (2026-10-06) | Casting String → Date/Datetime/Time removed; use `str.to_date()` | `InvalidOperationError` | 2.0 | check wheel packaging |
| 3 | pandas 3.0.0 (2026-01-21) | Parsed datetimes infer `datetime64[us]`, not `[ns]` | Wrong result: epoch via `astype('int64')//10**9` is 1000x off | 3.0 (rc0 Dec 2025) | needs numpy, dateutil, tzdata |
| 4 | click 8.2.0 (2025-05-10) | `CliRunner(mix_stderr=...)` removed; `result.output` includes stderr | `TypeError` / assertion fails | 8.2.0 | pure Python |
| 5 | scipy 1.18.0 (2026-06-19) | `sqrtm/logm/signm(disp=...)` removed | `TypeError` (old call returned `(X, errest)`) | 1.16.0 (Jun 2025) | ~35 MB wheel |
| 6 | SQLAlchemy 2.1.0 (2026-09-24) | `filter_by()` searches all FROM entities; ambiguous names raise | `AmbiguousColumnError` | 2.1 | sqlite in-memory test |
| 7 | FastAPI 0.132.0 (2026-02-23) | JSON bodies require a JSON Content-Type (`strict_content_type`) | 422 for clients without the header | 0.132.0 | "new required API" case |
| 8 | marshmallow 4.0.0 | `pass_many` → `pass_collection` | `TypeError` | 4.0.0, no deprecation | pure Python |
| 9 | numpy 2.5.0 (2026-06-21) | `linalg.eig/eigvals` always return complex | Wrong dtype / `json.dumps` `TypeError` | 2.5 | no deps |
| 10 | pandas 3.0.0 | `value_counts(sort=False)` keeps input order | Wrong order | 3.0, never warned | |
| 11 | polars 2.0.0 | `collect(streaming=True)` → `engine="streaming"` | `TypeError` (ArgumentRemovedError) | 1.25 (Mar 2025) | |
| 12 | scipy 1.18.0 | `stats.rankdata` always float | `IndexError` when ranks are used as indices | 1.18 | |
| 13 | click 8.2.0 | Command names drop `_cmd/_command/_group/_grp` suffix | Usage error, exit code 2 | 8.2.0 | |
| 14 | scikit-learn 1.8.0 (2025-12-10) | `AdaBoostClassifier(algorithm=...)` removed | `TypeError` | 1.6.0 (Dec 2024, at the cutoff) | numpy/scipy/joblib deps |
| 15 | Starlette 1.0.0 (2026-03-22) | `on_startup/on_shutdown`, `@app.route`, `@app.middleware` removed | `TypeError` / `AttributeError` | ⚠ 0.26 (2023) | strong old prior |
| 16 | pandas 3.0.0 | `zoneinfo` default instead of `pytz` | `AttributeError` on `tz.zone` / `tz.localize` | 3.0, no warning | |
| 17 | scipy 1.17.0 (2026-01-10) | `special.sph_harm` removed → `sph_harm_y` (new argument order) | `AttributeError`; naive port gives wrong values | 1.15.0 (Jan 2025) | |
| 18 | polars 2.0.0 | `explode()` default `empty_as_null=False` | Wrong row count | 2.0 | |
| 19 | Django 6.0 (2025-12-03) | `CheckConstraint(check=)` → `condition=` | `TypeError` | ⚠ 5.1 (Aug 2024) | |
| 20 | LangChain 1.0.0 (2025-10-17) | `langchain.chains.LLMChain` etc. moved to `langchain-classic` | `ModuleNotFoundError` | 1.0 | heavy deps |

Alternates (lighter deps): networkx 3.6 `node_link_data` "links" → "edges";
scikit-learn `LogisticRegression(multi_class=)` removed ⚠; matplotlib 3.11
`cm.get_cmap` removed ⚠; setuptools 82 `pkg_resources` removed ⚠; numpy 2.4
`float(arr)` on ndim>0 raises ⚠; click 8.2 `no_args_is_help` exit code 2;
marshmallow 4.0 `Schema(context=)` removed; pandas 3.0 `offsets.Day` across DST.

Weaker or excluded: pytest 8.4 (awkward fit); openai 3.0 / httpx2 (escape
hatch exists); huggingface_hub 1.0/2.0 (network-bound); cachetools 6.0
`MRUCache`; typer 0.16/0.26 `mix_stderr`; httpx 0.28 (pre-cutoff, possibly a
control). Excluded after checking: sklearn `LassoCV(n_alphas=)` (still
present); urllib3 `getheaders()` (restored in a patch release); FastAPI
`on_event` (re-implemented); pydantic 2.12–2.14 (no hard breaks).

## Caveats to resolve when building tasks
- Confirm each failure by running it (validator) and run the knowledge probe.
- Check that wheels for CPython 3.12 manylinux exist for the exact versions
  (polars 2.0 packaging, scipy size, LangChain dependency tree).
- Dates of 2025-04/05 (marshmallow 4.0, click 8.2) are after the cutoff but
  inside the planned mid-2025 margin: keep them only if the probe shows the
  model doesn't know them.
- Spread: avoid more than 3 per library; prefer pure-Python packages for
  offline bundling (marshmallow, click, SQLAlchemy, Starlette).

Primary sources: pandas whatsnew v3.0.0; numpy release notes 2.4.0/2.5.0;
scipy release notes 1.17.0/1.18.0; scikit-learn whats_new v1.6 + 1.8.0
source; click CHANGES.md; marshmallow CHANGELOG.rst; polars upgrade guide 2.md;
Starlette release-notes.md; FastAPI release-notes.md; SQLAlchemy
migration_21.rst; Django 6.0 release notes; LangChain v1 migration docs.
