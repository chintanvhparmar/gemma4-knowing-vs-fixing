# Fixing Is Easy, Knowing Isn't

Code, tasks, traces and analysis for the Kaggle *Gemma 4 Developer Agent Paper Track* entry:
**does a Gemma 4 coding agent notice when its library knowledge is out of date, and do web search or
offline docs help?** The paper, as submitted, is `paper/WRITEUP_SUBMITTED.md` (Kaggle: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/fixing-is-easy-knowing-isnt).

## Headline results (330 runs, Gemma 4 31B IT QAT, official harness)
| Task group | C0 none | C1 web | C2 web+hint | C3 docs | C4 both |
|---|---|---|---|---|---|
| L1 example in issue (16) | 31/32 | 32/32 | 32/32 | 32/32 | 31/32 |
| L2 spec in repo only (3) | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| L3 no spec (4) | 3/8 | 3/8 | 4/8 | 4/8 | 3/8 |
| Control (10) | 12/20 | 12/20 | 10/20 | 12/20 | 11/20 |

The agent made a look-up in 5 of 264 tool-enabled runs.

**Exploratory follow-up (not pre-registered):** `results/expl_nooffline_v1/` re-runs C1 on L3 with
the harness's "Environment is offline" task-message line removed (8 runs): 3/8 resolved and searches in
2/8 runs, both on scikit-learn. Launched with `modal_replica/spawn_expl.py`. To support it,
`modal_replica/replica.py` gained two optional settings after the freeze (`batches`, `drop_offline`).
Their defaults reproduce the main runs, so that file's hash differs from the manifest (diff: `modal_replica/replica_postfreeze.diff`).

## Layout
| Path | What |
|---|---|
| `prereg/` | Pre-registration (frozen before the main runs) and SHA-256 manifest (overall `993e0afe…c35231`, posted on the competition forum) |
| `tasks_postcutoff/` | 12 post-cutoff tasks: builder, task definitions, snapshots, validation (old passes / new fails / fix passes) |
| `tasks_silent/` | 4 silent-change tasks in 3 variants (`out`, `out_vague`, `out_nospec`) |
| `probe/` | Knowledge probe (28 library changes), raw answers and hand labels |
| `harness/` | Condition builder (prompt additions), skills (`web-search`, `library-docs`), web-search relay, runner, grading overlay |
| `modal_replica/` | Replica of the Kaggle GPU notebook on Modal, plus equivalence checks |
| `kaggle/` | Notebook cells used on Kaggle (environment export, fingerprints, calibration) |
| `configs/` | Task selection (control sample, calibration set, main-run groups) |
| `results/main_v1/` | Per-run metrics for all 330 runs (`runs.jsonl`), search logs, traces for our 23 tasks |
| `analysis/` | `analyze.py` and its output |
| `paper/` | Writeup and figures |

## Reproduce the analysis
```
python -I analysis/analyze.py results/main_v1
```
This writes `analysis/results_main_v1.md`. Only the Python standard library is needed.

## What is NOT included, and why
Competition data is not redistributed (competition rule 2.4.b):
- **Dev-set tasks and their traces.** For the 10 control tasks, only per-run metrics are included. As a
  result, the "look-up before 1st edit" column for control tasks reads 0 here; it is 1 for C4 in the paper.
- **The official sample agent's prompts.** In our traces, the system prompt and the task-message template
  are replaced by a note. Our own prompt additions are in `harness/build_conditions.py`.
- **The official harness wheels** (swegemma, adk-submission, adk-eval-core) and the Kaggle-built
  vLLM/safetensors wheels. These are available to competition participants on Kaggle.
- **`configs/control_tasks.json` is redacted** (the functions touched by dev-set gold fixes were removed),
  so its hash differs from `prereg/manifest.json`.
- **The 31 PyPI wheels for the post-cutoff tasks** (136 MB) are left out for size. Rebuild them with
  `pip download -r tasks_postcutoff/out/wheels_new.txt`; their SHA-256 hashes are in `prereg/manifest.json`.
  Every other pre-registered file, except `modal_replica/replica.py` (see above), matches the
  manifest byte for byte.

## Licence
Apache License 2.0 (`LICENSE`), the licence type named in the competition rules. Third-party code
that appears in traces (e.g. excerpts of pandas, click or scikit-learn source) remains under its own
licence.

## Disclosure
Claude (Anthropic) assisted with code, analysis and drafting. All tasks and issue texts are synthetic.
