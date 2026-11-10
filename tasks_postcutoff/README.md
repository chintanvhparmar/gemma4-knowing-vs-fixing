# tasks_postcutoff/

The 12 post-cutoff tasks for Paper 1. Each one is a small project that worked on the OLD
library version and breaks after upgrading to the NEW version (the latest release on 9 Nov 2026).
The changes come from the knowledge probe: Gemma 4 writes the old usage (`probe/results/hand_labels_*.json`).

| Task | Library old → new | Symptom after the upgrade |
|---|---|---|
| pc_marshmallow_validates | marshmallow 3.23.0 → 4.4.0 | TypeError: unexpected kwarg `data_key` |
| pc_marshmallow_context | marshmallow 3.23.0 → 4.4.0 | TypeError: unexpected kwarg `context` |
| pc_polars_streaming | polars 1.19.0 → 2.1.0 | ArgumentRemovedError: `streaming` |
| pc_click_command_suffix | click 8.1.8 → 8.5.1 | No such command 'init-cmd'. Did you mean 'init'? |
| pc_click_no_args_exit | click 8.1.8 → 8.5.1 | exit code 2 instead of 0 |
| pc_sklearn_adaboost | scikit-learn 1.6.1 → 1.9.1 | TypeError: unexpected kwarg `algorithm` |
| pc_sklearn_logreg_multiclass | scikit-learn 1.6.1 → 1.9.1 | TypeError: unexpected kwarg `multi_class` |
| pc_django_checkconstraint | Django 5.1.4 → 6.1.3 | TypeError: unexpected kwarg `check` |
| pc_networkx_node_link | networkx 3.4.2 → 3.7 | KeyError: 'links' |
| pc_pandas_epoch_seconds | pandas 2.2.3 → 3.1.1 | silent: values 1000x too small |
| pc_pandas_tz_zone | pandas 2.2.3 → 3.1.1 | AttributeError: ZoneInfo has no `zone` |
| pc_starlette_lifespan | starlette 0.41.3 → 1.10.0 | TypeError: unexpected kwarg `on_startup` |

## Files
- `build_tasks.py`: task definitions → `out/tasks.jsonl` (the same 8 fields as the dev set),
  `out/snapshots/<id>.tgz`, `out/meta.json` (versions, probe id, test files).
- `validate_modal.py`: CPU-only Modal check, 4 runs per task in fresh Python 3.12 venvs.
- `out/wheels/`: 31 wheels (136 MB) = the exact NEW-version environment (from the validation
  freeze), for offline Kaggle runs. `out/wheels_new.txt` = the pins.

## Validation (9 Nov 2026)
`validation.json` (PyPI) and `validation_offline.json` (new env from `out/wheels` only, `--no-index`):
**12/12 OK**. For each task: old lib without the fix PASSES, new lib without the fix FAILS (each
failure reason checked by hand; it matches the issue text), new lib with the fix PASSES. The
reference fix also passes on the old lib for 10/12; the exceptions are marshmallow context and
polars, whose new APIs don't exist in the old versions.

Issue texts quote the real error messages verbatim (e.g. click's "Did you mean 'init'?" hint is
kept, as a real user would paste it). They describe the symptom only, with no fix hint (C2 adds the hint).

## Checked on Kaggle (9 Nov, read-only cells A/A2/A3)
1. **Snapshot format:** official snapshots are git repos (`.git` included), members prefixed `./`.
   The harness commits the workspace and tags `_swegemma_baseline` at setup; `submit_patch` runs
   `git diff --binary _swegemma_baseline`. → Our snapshots are now real git repos (fixed author and
   date, so commit hashes are reproducible; LF checked on raw bytes); `base_commit` = that hash.
   Re-validated: 12/12 OK (PyPI and offline).
2. **Wheels:** `sandbox/setup.py` installs the HIGHEST version per package from `/wheels`
   (`--no-index`). Cell B merges the official wheels with ours, dropping the official copies of
   any package we ship, so the new library versions win.
3. **Code-graph tools:** graph/embedding files come from a Google-internal pipeline (no builder
   in swegemma). Without them, the agent's prompt lacks the "Code Intelligence Tools" section.
   → Graph tools are switched off for ALL tasks and conditions (`build_conditions.GRAPH_TOOLS = False`,
   and no graph dirs in run_batch / Cell B), so prompts and tools are identical across groups.
4. Per-group config: done (`run_batch(tasks_path=, snapshots_dir=, wheels_dir=)`).

## Official grading on Kaggle (9 Nov): 12/12 OK, with the overlay
- Without help: 3/12. Kaggle's subprocess sandbox never installs task wheels and tests import the
  notebook's (old) packages (swegemma container_setup.py / sandbox/subprocess.py, verified).
- With `harness/pc_overlay.py` (our wheels prepended to PYTHONPATH for post-cutoff tasks only):
  **12/12**: reference fix passes AND empty patch fails under the official `verify_task`
  (`kaggle/cell_B2_overlay_check.py`). The notebook's own packages are unchanged.
- Agent runs MUST enable the overlay for this group (and only for it).
