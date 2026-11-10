# =============================== CELL B2: official grading WITH the post-cutoff overlay ===============================
# Run after CELL B in the same session (reuses CODE, PC, PC_TASKS, evaluator, MODE, run_sync).
# Builds harness/pc_overlay.py's overlay from our wheels, enables it, then re-grades the 12 tasks.
import importlib, importlib.metadata as md, json, re, sys, time, traceback

sys.path.insert(0, str(CODE / 'harness'))
import pc_overlay
importlib.reload(pc_overlay)

OV = pc_overlay.build_overlay(PC / 'wheels', WORKING_DIR / 'pc_overlay')
print('overlay versions (fresh interpreter, PYTHONPATH=overlay):', pc_overlay.overlay_versions(OV))
print('pytest in overlay?', any(p.name.lower().startswith('pytest') for p in OV.iterdir()))
pc_overlay.enable(OV)
print('real swegemma env builder -> PYTHONPATH =', pc_overlay.selftest())
print('notebook itself unchanged: marshmallow', md.version('marshmallow'), '| pandas', md.version('pandas'))

OUT2 = WORKING_DIR / 'pc_official_check_overlay.jsonl'
OUT2.unlink(missing_ok=True)
try:
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
            with OUT2.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            err = ('  ERR: ' + str(rec['error'])[:150]) if rec['error'] else ''
            print(f"[{label} {i}/{len(PC_TASKS)}] {t.instance_id:30} resolved={str(rec['resolved']):5} "
                  f"exit={rec['exit_code']} {rec['duration']:>6}s {rec['summary'][:70]}{err}", flush=True)
finally:
    pc_overlay.disable()  # never leave the overlay on for later cells
    print('overlay disabled; PYTHONPATH now =', pc_overlay.selftest())

recs = [json.loads(l) for l in OUT2.read_text().splitlines()]
g = {r['id']: r['resolved'] for r in recs if r['mode'] == 'gold'}
e = {r['id']: r['resolved'] for r in recs if r['mode'] == 'empty'}
ok = [i for i in g if g[i] and not e.get(i)]
print(f'\n=== OFFICIAL CHECK WITH OVERLAY: {len(ok)}/{len(g)} OK (reference fix passes AND empty patch fails) ===')
print('problems:', {i: (g[i], e.get(i)) for i in g if i not in ok})
for r in recs:
    if r['id'] not in ok and r['mode'] == ('gold' if not g[r['id']] else 'empty'):
        print(f"\n--- {r['mode']} {r['id']}\n{r['test_tail'][-1200:]}")
