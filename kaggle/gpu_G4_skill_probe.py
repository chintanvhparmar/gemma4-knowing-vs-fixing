# =============================== G4: forced skill MECHANICS test (not an experimental condition) ===============================
# Run after G2/G3 with vLLM still up. A temporary copy of C3 is told to use the library-docs skill, on the
# marshmallow post-cutoff task with the overlay. Checks: is load_skill offered? does run_skill_script run
# pkg_info.py inside the sandbox, and does it see the overlay's marshmallow 4.4.0? Results are NOT used in the paper.
import json, shutil
from pathlib import Path

probe_dir = WORKING_DIR / 'agents' / 'C3_skillprobe'
shutil.rmtree(probe_dir, ignore_errors=True)
shutil.copytree(AG['C3'], probe_dir)
sm = probe_dir / 'prompts' / 'system.md'
sm.write_text(sm.read_text() + (
    "\n\n## Mandatory first step (mechanics test)\n"
    "Before anything else: call `load_skill` with skill_name `library-docs`, then call `run_skill_script` "
    "with skill_name `library-docs`, file_path `scripts/pkg_info.py` and the arguments `version marshmallow`. "
    "Quote the printed version in your final message.\n"))

OV = pc_overlay.build_overlay(PC / 'wheels', WORKING_DIR / 'pc_overlay')
pc_overlay.enable(OV)
try:
    run_experiment.run_batch({'C3probe': probe_dir}, ['pc_marshmallow_validates'], PC_TASKS, repeats=1, models=models,
                             data_dir=DATA_DIR, out_dir=WORKING_DIR / 'dry' / 'skillprobe', concurrency=1,
                             max_hours=0.5, platform_name='kaggle', tasks_path=PC / 'tasks.jsonl',
                             snapshots_dir=PC / 'snapshots', wheels_dir=PC / 'wheels')
finally:
    pc_overlay.disable()

for tr in (WORKING_DIR / 'dry' / 'skillprobe').rglob('*.trace.json'):
    steps = json.loads(tr.read_text()).get('steps', [])
    for s in steps:
        for tc in s.get('tool_calls') or []:
            name = tc.get('function_name')
            if name in ('load_skill', 'load_skill_resource', 'run_skill_script', 'list_skills') or 'skill' in (name or ''):
                print(f"\nTOOL {name} args={json.dumps(tc.get('arguments'))[:300]}")
                print('  ->', json.dumps(s.get('observation'))[:800])
    final = next((s.get('message') for s in reversed(steps) if (s.get('extra') or {}).get('event_type') == 'final'), '')
    print('\nFINAL MESSAGE:', (final or '')[:600])
rows = [json.loads(l) for l in (WORKING_DIR / 'dry' / 'skillprobe' / 'runs.jsonl').read_text().splitlines()]
print('\nrun summary:', {k: rows[0].get(k) for k in ('resolved', 'tool_calls', 'skill_calls', 'docs_calls', 'skill_loads', 'tool_hist', 'error')})
