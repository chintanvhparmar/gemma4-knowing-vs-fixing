# Fixing Is Easy, Knowing Isn't

### A pre-registered study of whether a Gemma 4 coding agent notices when its library knowledge is out of date, and whether web search or offline docs help

Chintan Parmar

---

## Abstract

Libraries change after a model's training cutoff. We separate **fixing** (adapting code once there is
a signal that a library changed) from **knowing** (realising, without such a signal, that one's beliefs
about a library may be stale). We tested Gemma 4 31B IT (4-bit, training cutoff January 2025) in the official competition
harness on 33 tasks:
- 12 bugs caused by library changes released after the cutoff;
- 4 "silent" changes, where old code runs without error but returns a different result, each in three
  versions with less and less specification;
- 10 ordinary bug-fix tasks as controls.

Each task ran under five pre-registered conditions (no tools, web search, web search plus a staleness
hint, offline docs, both), giving 330 runs. With the expected behaviour written down anywhere, the agent
fixed these bugs without tools (31/32 and 6/6 runs). With no specification it dropped to 3/8, and two of
four tasks failed in all 10 of their runs. The tools did not change this (3–4/8 in every condition)
because the agent almost never used them: 5 look-ups in 264 tool-enabled runs, and zero after a warning
that its knowledge might be outdated. In most failed runs its reasoning noted that the library behaved
"strangely", yet it invented an explanation instead of checking. The harness's "offline" task text
blocked a few searches, but removing it (exploratory) left the hardest failures unchanged. Unused tools
still added 25–30% prompt tokens. The bottleneck is not access to current information but recognising
the need for it.

---

## 1. Introduction

A coding agent writes code for the library versions its model saw in training, but projects upgrade.
Between January 2025 and our experiments, pandas 3, scikit-learn 1.8, click 8.2, marshmallow 4,
Django 6, polars and Starlette all changed behaviour. The standard remedy is tool access: let the
agent search or read documentation [5, 6]. That assumes the agent notices when it needs to look
something up. We test this with the competition's own model and harness:
- **RQ1:** Does missing knowledge stop the agent, depending on how much expected behaviour is written down?
- **RQ2:** Does the agent use optional look-up tools, and when? Does a staleness hint change this?
- **RQ3:** Do the tools raise success where knowledge is missing?
- **RQ4:** Do they cost or harm where not needed?

**Contributions.**
1. **A task design separating knowing from fixing:** one silent change, presented with an example (L1),
   a repository-only specification (L2), or none (L3).
2. **A pre-registered five-condition comparison** in the official harness, on a proven-equivalent
   replica of the Kaggle environment; the plan's SHA-256 manifest was posted before the main runs [8].
3. **Trace evidence of whether and when the agent looks things up,** and what it does when surprised,
   plus practical guidance for teams that deploy or review coding agents.

---

## 2. Related Work

**Version-aware code generation.** VersiCode [2] and GitChameleon [3] show that code LLMs often generate
version-incorrect code. CodeUpdateArena [1] finds that prepending update documentation is not enough for
models to use synthetic API updates. These benchmarks test single-turn generation; we study a
*multi-turn agent* that can run code, read the installed library and optionally search.

**Agents and tools.** SWE-bench [4] set the issue-resolution format the competition harness follows.
ReAct [5] and Toolformer [6] showed that models can interleave reasoning with tool calls.
Self-knowledge-guided retrieval [7] argues that retrieval should depend on whether the model knows the
answer, and SelfAware [9] shows that LLMs recognise the limits of their knowledge only partially. We
give a software-engineering instance: the tools work, but the agent rarely decides it needs them.

---

## 3. Method

### 3.1 Model and harness
- **Model:** Gemma 4 31B IT, QAT w4a16 (Kaggle version 2), vLLM 0.19.1 with the official server
  configuration and sampling (temperature 0.2, top-p 0.95, thinking budget 4,096; no seed).
- **Agent:** the official sample agent (swegemma 0.2.11) with its standard tools: read, edit and write
  file; run command; status; submit patch.
- **Changes in every condition:** we removed the LoRA adapters (base model) and the code-graph tools.
- **Budget:** 900 s, 60 tool calls and a 15-minute session per run.

### 3.2 Tasks
**Post-cutoff tasks (12).**
- **Probe:** we asked the model about 28 library changes released after January 2025, each checked
  against its changelog, and labelled the answers by hand (an automatic labeller was wrong on 9 of 40).
- **Selection:** the 12 changes the model answered wrongly became tasks, spanning 8 libraries:
  marshmallow, polars, click, scikit-learn, Django, networkx, pandas and Starlette.
- **Task form:** a small repository correct for the old version and broken by the new one, plus an
  issue describing the symptom.
- **Validation:** hidden test passes on the old version, fails on the new, passes with the reference
  fix; 12/12 offline and in the official grader.
- **Hinted subgroup (pre-registered):** in 7 of the 12 issues, the issue or its error message names the
  API element involved.

**Silent-change tasks (4 changes).** In each, old code runs without error or warning on the new version
but returns a different result:
- pandas 3 `groupby(observed=False)` with unobserved categories;
- scikit-learn 1.8 weighted `median_absolute_error`;
- click 8.2+ boolean flags set via environment variables;
- pandas 3 Float64 missing values (NaN becomes NA).

Each change has three versions, all with the same hidden test and reference fix:
- **L1, informative:** the issue gives example values.
- **L2, vague:** the issue gives none, but the README or docstrings define the expected behaviour.
- **L3, no-spec:** the defining text is removed, so only the old library "knew" the behaviour.

The scikit-learn vague version equalled its no-spec version and was dropped from L2. All versions were
validated at exact pins; all task texts are synthetic, written by us.

**Control tasks (10).** Sampled with a fixed seed (2026) from the public development set (FastAPI, Rich)
among tasks whose official fix passes and whose empty patch fails.

**Analysis groups (33 tasks):** L1 = 12 post-cutoff + 4 silent-informative; L2 = 3 silent-vague;
L3 = 4 silent no-spec; control = 10.

### 3.3 Conditions

| ID | Extra tools | Prompt addition |
|---|---|---|
| C0 | none | — |
| C1 | web search (Tavily, via a local relay; query ≤ 200 chars, ≤ 8 results) | "A web-search skill is available… Using it is optional." |
| C2 | as C1 | C1 note + "libraries … may be newer than your training data" |
| C3 | offline library docs (installed version, API list, signatures, docstrings, source) | "A library-docs skill is available… Using it is optional." |
| C4 | both | both notes |

Tools are harness *skills*: the agent sees a one-line description and decides itself whether to use one.
The API key stays in the relay, outside the sandbox; every query is logged. The notes above go in the
system prompt. The official task message, unchanged in all conditions, still says the environment is
"offline" and asks the agent not to search outside the repository. We kept it, to test the harness as
shipped (see Limitations).

### 3.4 Platform and equivalence
Kaggle GPU notebooks must run with internet disabled, so web search cannot run there. We therefore ran
**all** conditions on one platform: a replica of the Kaggle GPU notebook on Modal, built from Kaggle's
public image plus the notebook's own install steps.

**Equivalence checks:** identical `pip freeze` (969 packages), grading (84/84 gold-patch checks),
repository snapshots (30/30) and model files (8/8 sha256). The Hugging Face copy of the model has a
newer chat template that renders tool calls differently, so we used Kaggle's.

**Transfer check (pilot):** the no-tools agent on 22 tasks × 2 repeats on both platforms agreed on 21/22
tasks; the exception also flips between repeats on one platform. Hardware: one L40S (Kaggle: 4× L4).

### 3.5 Pre-registration and statistics
- **Frozen before any main run:** tasks, groups, conditions, R = 2 repeats and analysis. The SHA-256
  manifest covers 85 files (overall hash `993e0afe…c35231`).
- **Unit of analysis:** the task (share of its runs resolved), paired across conditions.
- **Statistics:** 95% bootstrap confidence intervals (10,000 task-level resamples, seed 2026) and exact
  sign-flip permutation tests. With 4 L3 tasks we report effect sizes and intervals, not significance.
- **Exclusions:** only infrastructure failures could be excluded, and there were none. Runs that hit the
  15-minute limit (27) or the model's context window (10) count as failures.

---

## 4. Results

### 4.1 Pilot: our first premise was wrong
We first assumed that "the model does not know the change" meant "the agent will fail the task." The
no-tools pilot disproved this: post-cutoff tasks 24/24, silent changes with an example 8/8, with a
repository-only specification 7/8, with no specification 4/8. Although the model had answered every post-cutoff change wrongly in the probe, error messages, failing
tests and the installed source let it recover the new API. We rebuilt the study around this
*specification gradient* before freezing the plan: a valid task is not necessarily a hard one.

### 4.2 RQ1: knowing versus fixing

![Fig. 1](fig1_resolve_by_group.png)

**Table 1.** Resolved runs / runs.

| Group | C0 none | C1 web | C2 web+hint | C3 docs | C4 both |
|---|---|---|---|---|---|
| L1 example in issue (16) | 31/32 | 32/32 | 32/32 | 32/32 | 31/32 |
| L2 spec in repo only (3) | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| L3 no spec (4) | 3/8 | 3/8 | 4/8 | 4/8 | 3/8 |
| Control (10) | 12/20 | 12/20 | 10/20 | 12/20 | 11/20 |

**Written specification (L1, L2): 188 of 190 runs resolved.** Missing knowledge barely mattered.
The pre-registered hinted/not-hinted split shows no difference (70/70 vs 49/50; the miss was a
context-window overflow).

**No specification (L3): 17 of 40 runs resolved,** but very unevenly:

| L3 task | Resolved (all conditions) |
|---|---|
| pandas groupby | 9/10 |
| scikit-learn weighted metric | 8/10 |
| click environment flag | 0/10 |
| pandas Float64 missing values | 0/10 |

### 4.3 How the agent fails without a specification
Of the 23 unresolved L3 runs, **21 ended with a submitted "fix"**; the other 2 ran out of context. All
20 click and Float64 runs submitted a fix. In the six of those traces we read in full, the same pattern
appeared every time:
1. The agent notices that the library behaves unexpectedly, in its own words:
   - "This is extremely weird."
   - "Wow! … This is definitely a bug in click or a change in behavior."
   - "Pandas 3.1.1? That's a very new version (or a future version)."
2. It then **invents a plausible but wrong theory of what changed.** For example, it decided that
   `to_numpy()` had "started honouring `na_value`", which is false; the actual change is how missing
   values are represented.
3. It patches the code to match its theory, checks the patch with a script it wrote itself, and submits
   with confidence.

Under our pre-registered failure labels, these are "other: confabulated change". None of the 20 runs
used the search or docs tools. Several inspected the installed library through shell commands, but
that shows only the *current* behaviour, not what changed.

**Keyword count:** "weird", "strange", "very new" and similar appear in 22 of the 23 unresolved runs;
in 15 of the 18 with a tool available, no look-up followed. The staleness signal was there, unused.

### 4.4 RQ2: does the agent use the tools?

![Fig. 2](fig2_lookup_usage.png)

**Overall usage.** The agent made at least one look-up in 5 of 264 tool-enabled runs (1.9%; Fig. 2).

- **Where:** none on L1 or L2, and none in any run of C2 or C3.
- **When:** all 5 look-ups came before the first edit, and 4 were on L3 tasks.
- **Quality:** the queries were precise, e.g. "scikit-learn median_absolute_error sample_weight change".
  When it searches, it searches well; the failure is in deciding to search.
- **The hint did nothing.** The staleness hint (C2), which explicitly warns that libraries may be newer
  than the training data, produced no look-ups at all (C1 without the hint: 1).
- **Offline docs alone (C3) were never used.**
- **The "offline" task text blocked some searches.** In 8 of 198 web-condition runs the agent planned a
  look-up and then followed the task message instead of the tool note: "let's check the pandas 3.1.1
  documentation or changes. Wait, I don't have internet access…" (C1, L3). In the other runs a
  keyword search found no sign that it considered a look-up.

### 4.5 RQ3: do the tools help where knowledge is missing?
**Table 2.** L3, paired per-task difference in resolve share versus C0.

| Contrast | Mean | 95% CI |
|---|---|---|
| C1 − C0 | 0.00 | [0.00, 0.00] |
| C2 − C0 | +0.125 | [0.00, 0.375] |
| C3 − C0 | +0.125 | [0.00, 0.375] |
| C4 − C0 | 0.00 | [−0.375, 0.375] |

All exact p-values were 1.0. The +0.125 is one scikit-learn run each in C2 and C3 with no look-up, so
run-to-run variation. Click and Float64 never received a look-up in 40 runs. The tools did not raise
success because they were almost never used.

### 4.6 RQ4: cost and harm
**Cost of unused tools.** Skill descriptions are added to every model call. On L1, median prompt tokens
per run rose from 49.2k (C0) to 61.7–63.7k (C1–C4), an increase of 25–30%. Median tool calls stayed about the
same (8 vs 8–8.5) and no look-ups were made.

**Harm on control tasks.** C1 and C3 matched C0 exactly (12/20). C2 (10/20) and C4 (11/20) lost runs
on a single task, with no look-ups involved (C2 − C0 = −0.10, 95% CI [−0.30, 0.00]). Four control tasks
failed in every run of every condition.

### 4.7 Exploratory follow-up (not pre-registered): removing "offline"
Prompted by §4.4, we re-ran C1 on L3 (8 runs, decided after the main runs) with the "offline" line
removed and everything else unchanged. The agent never mentioned lacking internet. It searched in 2/8 runs
(main C1: 1/8), both on scikit-learn, and resolved 3/8, as before. On click and Float64 it made **no**
search and solved none of 4 runs, although its reasoning again called the behaviour "weird" or "strange".
The instruction suppressed some searches, but it does not explain the hardest failures.

---

## 5. Discussion

**Fixing is easy.** Given any concrete signal (an error, a failing test, written intent), the agent
adapted to unseen APIs by reading the installed source and experimenting, not by consulting
documentation. So knowledge-probe failures did not predict task failures.

**Knowing is the bottleneck.** When a change is silent and undocumented, nothing contradicts the agent.
It *does* register surprise, but lacks a policy that turns "this behaves unexpectedly" into "my
knowledge may be stale, so check"; it explains the surprise away instead. This matches the partial
self-knowledge reported for LLMs [7, 9], here with behavioural evidence from agent traces.

**A warning in the prompt is not a policy.** Even C2's explicit "check the installed version and its
documentation" produced no tool use.
More promising: inject release notes for libraries newer than the model's knowledge horizon, require
an API look-up before editing upgraded-library code, or treat surprise in the reasoning as a cue to search.

**Conflicting instructions silently disable tools.** The tool note sat in the system prompt; the
harness's task text said "offline"; the task text won, with no error or warning. Whoever adds a tool to
an agent should check every prompt layer for statements that contradict it.

**Specifications are the most effective "tool".** Every task with written expected behaviour was solved,
with or without search.

**Generalisation.** The design (one change, three specification levels, optional tools) is
model-agnostic and applies to any library, language or agent.

---

## 6. Limitations
- **Small task set.** 4 L3 tasks × 2 repeats; the intervals are wide and we make no significance claims.
- **Conflicting instruction.** The official task message says "offline"; removing it (§4.7, 8
  exploratory runs) did not change the hard cases, but is too small to rule out an effect elsewhere.
- **One model, one agent design.**
- **Synthetic tasks.** Each task isolates a single change; real upgrades mix many.
- **Hardware.** One L40S, not Kaggle's 4× L4 (slower; per-task outcomes still agreed on 21/22).
- **Small control set.** 10 tasks (a pre-registered budget decision); four fail in every condition.
- **Python only.** In typed ecosystems (e.g. Java) compile errors may turn L3-like changes into L1-like ones.
- **Keyword-based surprise count.** It is supported by manual reading of the click and Float64 traces,
  not by full manual labelling.

---

## 7. Implications for Teams Using Coding Agents
1. **"Has web search" does not mean "stays current."** Measure how often your agent actually looks
   things up (here 2%), and check its prompts for contradictions such as "offline".
2. **Write expected behaviour down.** Tests with concrete expected values turn silent changes into
   failures agents fix.
3. **Review agent changes around dependency upgrades carefully.** On two of four silent changes the
   agent was confidently wrong in 20 of 20 runs.
4. **Unused tools are not free.** Tool descriptions added 25–30% to prompt tokens.

---

## 8. Reproducibility, Ethics and Disclosure
- **Pre-registration:** manifest of 85 files, overall SHA-256
  `993e0afe22a5738bb9605a68652ad52c8b6002115c71601504ae2ad124c35231`, posted before the main runs [8].
- **Code, tasks, traces, analysis:** GITHUB_URL (no competition data; rule 2.4.b).
- **Compute:** about $34 of cloud GPU time (330 main + 8 exploratory runs, pilots, probes); 8 web
  searches in total across these runs. For §4.7, `modal_replica/replica.py` gained two optional settings after the freeze
  (defaults reproduce the main runs; diff in git history).
- **AI assistance:** Claude (Anthropic) assisted with code, data analysis and drafting. The author
  designed the study, made all decisions, verified the results and is responsible for the text.

---

## References
[1] Z. L. Liu, S. Pandit, X. Ye, E. Choi, G. Durrett. *CodeUpdateArena: Benchmarking Knowledge Editing on API Updates.* arXiv:2407.06249, 2024.
[2] T. Wu et al. *VersiCode: Towards Version-controllable Code Generation.* arXiv:2406.07411, 2024.
[3] *GitChameleon: Unmasking the Version-Switching Capabilities of Code Generation Models.* arXiv:2411.05830, 2024; expanded as *GitChameleon: Evaluating AI Code Generation Against Python Library Version Incompatibilities*, arXiv:2507.12367, 2025.
[4] C. E. Jimenez, J. Yang, et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770, 2023.
[5] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, Y. Cao. *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629, 2022.
[6] T. Schick et al. *Toolformer: Language Models Can Teach Themselves to Use Tools.* arXiv:2302.04761, 2023.
[7] Y. Wang, P. Li, M. Sun, Y. Liu. *Self-Knowledge Guided Retrieval Augmentation for Large Language Models.* Findings of EMNLP 2023. arXiv:2310.05002.
[8] Pre-registration hash post, Gemma 4 Developer Agent Paper Track forum: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/757614
[9] Z. Yin, Q. Sun, Q. Guo, J. Wu, X. Qiu, X. Huang. *Do Large Language Models Know What They Don't Know?* arXiv:2305.18153, 2023.
[10] E. Markowitz et al. *Google – The Gemma 4 Developer Agent Paper Track.* Kaggle, 2026. https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
