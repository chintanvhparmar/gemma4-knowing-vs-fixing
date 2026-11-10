# Kaggle CPU check for the post-cutoff tasks. Paste each CELL into the `gold-patch-validator`
# notebook AFTER its cells 1 (wheelhouse install) and 2 (DATA_DIR, WORKING_DIR, tasks, AGENT_DIR).
# Accelerator: None. Inputs: the competition data + our private dataset `gemma-paper-code`.
# Nothing here changes grading: CELL B reuses the official Evaluator exactly like the 73/129 run.

# =============================== CELL A: inspect (read-only) ===============================
import glob, io, json, re, shutil, tarfile, zipfile
from pathlib import Path

# A1. Where is our dataset? (Kaggle may or may not unzip the upload)
hits = glob.glob('/kaggle/input/**/tasks_postcutoff/out/tasks.jsonl', recursive=True)
if not hits:
    zips = glob.glob('/kaggle/input/**/gemma-paper-code*.zip', recursive=True)
    print('not unzipped; zip found:', zips)
    if zips:
        zipfile.ZipFile(zips[0]).extractall('/kaggle/working/code')
        hits = glob.glob('/kaggle/working/code/tasks_postcutoff/out/tasks.jsonl')
CODE = Path(hits[0]).parents[2]
PC = CODE / 'tasks_postcutoff' / 'out'
print('CODE =', CODE)
print('PC files:', sorted(p.name for p in PC.iterdir()))

# A2. Official snapshot format: .git inside? top-level folder?
snap = sorted((DATA_DIR / 'snapshots').glob('*'))[0]
print('\nofficial snapshot example:', snap.name)
with tarfile.open(snap) as t:
    names = t.getnames()
print('  members:', len(names), '| first 12:', names[:12])
print('  has .git:', any(n.split('/')[0] == '.git' or '/.git/' in n or n.startswith('./.git') for n in names))
print('  top-level entries:', sorted({n.lstrip('./').split('/')[0] for n in names})[:15])

# A3. How sandbox/setup.py installs wheels (lines mentioning pip / find-links / no-index / editable)
src = (DATA_DIR / 'sandbox' / 'setup.py').read_text()
print('\nsandbox/setup.py install lines:')
for i, l in enumerate(src.splitlines(), 1):
    if re.search(r'pip|find.links|no.index|editable|-e |wheel', l, re.I):
        print(f'{i:4}: {l}')

# A4. Code-graph tools: what happens when graphs/embeddings are missing for a task?
import swegemma, os
root = Path(swegemma.__file__).parent
print('\ngraph/embedding handling in swegemma:')
for p in sorted(root.rglob('*.py')):
    for i, l in enumerate(p.read_text(errors='ignore').splitlines(), 1):
        if re.search(r'graph_dir|embeddings_dir', l) and re.search(r'exist|not found|missing|None|warn|raise', l, re.I):
            print(f'  {p.relative_to(root)}:{i}: {l.strip()[:140]}')
print('official graph files example:', sorted(os.listdir(DATA_DIR / 'graphs'))[:3])

# =============================== CELL B: official grading of our 12 tasks ===============================
# Gold = reference fix must PASS; empty = no fix must FAIL. Same Evaluator/verify_task as the dev set.
import asyncio, collections, concurrent.futures, inspect, time, traceback
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

PC_TASKS = load_tasks(PC / 'tasks.jsonl')
print('loaded', len(PC_TASKS), 'post-cutoff tasks:', [t.instance_id for t in PC_TASKS][:3], '...')

# Wheels: official wheels + ours; for packages we ship, keep ONLY our version (so the new library wins)
def _pkg(name):  # wheel filename -> normalised distribution name
    return re.sub(r'[-_.]+', '-', name.split('-')[0]).lower()
PC_WHEELS = WORKING_DIR / 'pc_wheels'
shutil.rmtree(PC_WHEELS, ignore_errors=True); PC_WHEELS.mkdir()
ours = {_pkg(p.name) for p in (PC / 'wheels').glob('*.whl')}
n_off = 0
for p in (DATA_DIR / 'wheels').glob('*.whl'):
    if _pkg(p.name) not in ours:
        shutil.copy2(p, PC_WHEELS); n_off += 1
for p in (PC / 'wheels').glob('*.whl'):
    shutil.copy2(p, PC_WHEELS)
print(f'pc_wheels: {n_off} official + {len(ours)} ours')

def run_sync(fn, *args, **kwargs):
    """Works whether evaluate_task is async or not (not assumed)."""
    async def _call():
        r = fn(*args, **kwargs)
        return (await r) if inspect.isawaitable(r) else r
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():  # Jupyter already runs a loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(_call())).result()
    return asyncio.run(_call())

MODE = {'patch': {}}
if inspect.iscoroutinefunction(Evaluator._run_agent_sandbox):
    async def _fixed(self, task, *a, **k):
        return MODE['patch'][task.instance_id], None, None
else:
    def _fixed(self, task, *a, **k):
        return MODE['patch'][task.instance_id], None, None
Evaluator._run_agent_sandbox = _fixed

limits, gen_constraints = build_submission_limits()
cfg = EvalConfig(
    tasks_path=PC / 'tasks.jsonl', snapshots_dir=PC / 'snapshots',
    results_dir=WORKING_DIR / 'pc_gold_results', submission_dir=AGENT_DIR,
    models={}, sandbox='subprocess',
    timeout_seconds=600, verification_timeout_seconds=900, max_time_minutes=30.0,
    max_tool_calls=100, max_turns=None, limits=limits, generation_constraints=gen_constraints,
    graph_dir=None, embeddings_dir=None,  # graph tools off for all tasks (build_conditions change 4)
    wheels_dir=PC_WHEELS, verbose=False,
)
evaluator = Evaluator(cfg)

OUT = WORKING_DIR / 'pc_official_check.jsonl'
OUT.unlink(missing_ok=True)
for label, patches in (('gold', {t.instance_id: t.patch for t in PC_TASKS}),
                       ('empty', {t.instance_id: '' for t in PC_TASKS})):
    MODE['patch'] = patches
    for i, t in enumerate(PC_TASKS, 1):
        t0 = time.time()
        try:
            r = run_sync(evaluator.evaluate_task, task=t, task_index=i, total_tasks=len(PC_TASKS))
            out = r.test_output or ''
            lines = [l for l in out.splitlines() if re.search(r'\d+ (passed|failed|errors?)\b', l)]
            rec = dict(mode=label, id=t.instance_id, resolved=bool(r.resolved), exit_code=r.test_exit_code,
                       error=r.error, summary=lines[-1].strip() if lines else '', test_tail=out[-2000:])
        except Exception as e:
            rec = dict(mode=label, id=t.instance_id, resolved=False, exit_code=None,
                       error=f'HARNESS EXCEPTION: {e!r}', summary='', test_tail=traceback.format_exc()[-2000:])
        rec['duration'] = round(time.time() - t0, 1)
        with OUT.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        err = ('  ERR: ' + str(rec['error'])[:150]) if rec['error'] else ''
        print(f"[{label} {i}/{len(PC_TASKS)}] {t.instance_id:30} resolved={str(rec['resolved']):5} "
              f"exit={rec['exit_code']} {rec['duration']:>6}s {rec['summary'][:70]}{err}", flush=True)

recs = [json.loads(l) for l in OUT.read_text().splitlines()]
g = {r['id']: r['resolved'] for r in recs if r['mode'] == 'gold'}
e = {r['id']: r['resolved'] for r in recs if r['mode'] == 'empty'}
ok = [i for i in g if g[i] and not e.get(i)]
print(f'\n=== OFFICIAL CHECK: {len(ok)}/{len(g)} OK (reference fix passes AND empty patch fails) ===')
print('problems:', {i: (g[i], e.get(i)) for i in g if i not in ok})
