# =============================== G2: dry run, 3 tasks x C0/C3 (6 agent runs) ===============================
# Run after G1. Same code path as the real runs: harness/run_experiment.run_batch (+ pc_overlay).
import json, time
from swegemma.models import load_tasks

CONTROL = json.loads((CODE / 'configs' / 'control_tasks.json').read_text())['control_tasks']
ctl_ids = [CONTROL[0], next(i for i in CONTROL if i.startswith('rich_'))]   # 1 FastAPI + 1 Rich
print('control dry-run tasks:', ctl_ids)

t0 = time.time()
run_experiment.run_batch(AG, ctl_ids, tasks, repeats=1, models=models, data_dir=DATA_DIR,
                         out_dir=WORKING_DIR / 'dry' / 'control', concurrency=2, max_hours=1.5,
                         platform_name='kaggle')

# Post-cutoff group: our tasks/snapshots + the overlay (ONLY for this batch)
PC = CODE / 'tasks_postcutoff' / 'out'
PC_TASKS = load_tasks(PC / 'tasks.jsonl')
OV = pc_overlay.build_overlay(PC / 'wheels', WORKING_DIR / 'pc_overlay')
pc_overlay.enable(OV)
print('overlay on:', pc_overlay.selftest())
try:
    run_experiment.run_batch(AG, ['pc_marshmallow_validates'], PC_TASKS, repeats=1, models=models,
                             data_dir=DATA_DIR, out_dir=WORKING_DIR / 'dry' / 'postcutoff', concurrency=2,
                             max_hours=1.0, platform_name='kaggle', tasks_path=PC / 'tasks.jsonl',
                             snapshots_dir=PC / 'snapshots', wheels_dir=PC / 'wheels')
finally:
    pc_overlay.disable()
    print('overlay off:', pc_overlay.selftest())
print(f'G2 total: {(time.time() - t0) / 60:.1f} min')
