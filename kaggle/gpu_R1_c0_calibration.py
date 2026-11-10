# =============================== R1: C0 platform calibration on Kaggle (pilot; excluded from main analysis) ===============================
# GPU notebook (L4x4, Pin 2026-07-01, dataset gemma-paper-code v7+). Run AFTER the official cells 1, 2 and
# "4. Start vLLM Server" (section 3 not needed). Same task list and repeats as on Modal:
#   python -m modal run --detach modal_replica/replica.py::agent_runs --conds C0 --group control    --tasks <cal control> --repeats 2 --run-name cal_c0
#   python -m modal run --detach modal_replica/replica.py::agent_runs --conds C0 --group postcutoff --tasks all           --repeats 2 --run-name cal_c0
# 22 tasks x 2 repeats = 44 runs, concurrency 3; ~1-1.5 h (estimate; GPU quota counts double on L4).
# Resumable: re-running this cell skips runs already in runs.jsonl. Download /kaggle/working/cal_c0_kaggle.zip at the end.
import glob, json, shutil, sys, time
from pathlib import Path

hits = glob.glob('/kaggle/input/**/gemma-paper-code/harness/run_experiment.py', recursive=True)
CODE = Path(hits[0]).parents[1]
ver = CODE / 'CODE_VERSION.txt'
print('CODE =', CODE, '| CODE_VERSION', ver.read_text().strip() if ver.exists() else 'MISSING (old upload?)')
sys.path.insert(0, str(CODE / 'harness'))
import build_conditions, run_experiment, pc_overlay
from swegemma.models import load_tasks

CAL = json.loads((CODE / 'configs' / 'calibration_tasks.json').read_text())
print('calibration:', len(CAL['control']), 'control +', len(CAL['postcutoff']), 'post-cutoff | MODEL_NAME =', MODEL_NAME)
AG = build_conditions.build_all(DATA_DIR / 'sample_submission', WORKING_DIR / 'agents', model=MODEL_NAME, conditions=['C0'])
OUT = WORKING_DIR / 'cal_c0'
t0 = time.time()
run_experiment.run_batch(AG, CAL['control'], tasks, repeats=2, models=models, data_dir=DATA_DIR,
                         out_dir=OUT / 'control', concurrency=3, max_hours=4.0, platform_name='kaggle')
PC = CODE / 'tasks_postcutoff' / 'out'
OV = pc_overlay.build_overlay(PC / 'wheels', WORKING_DIR / 'pc_overlay')
pc_overlay.enable(OV)
try:
    run_experiment.run_batch(AG, CAL['postcutoff'], load_tasks(PC / 'tasks.jsonl'), repeats=2, models=models,
                             data_dir=DATA_DIR, out_dir=OUT / 'postcutoff', concurrency=3, max_hours=4.0,
                             platform_name='kaggle', tasks_path=PC / 'tasks.jsonl',
                             snapshots_dir=PC / 'snapshots', wheels_dir=PC / 'wheels')
finally:
    pc_overlay.disable()
print(f'R1 total: {(time.time() - t0) / 60:.1f} min')
shutil.make_archive(str(WORKING_DIR / 'cal_c0_kaggle'), 'zip', OUT)
print('download:', WORKING_DIR / 'cal_c0_kaggle.zip')
