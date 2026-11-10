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
