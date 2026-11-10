# =============================== G1: checks before any agent run (GPU notebook, after the vLLM cell) ===============================
# Notebook: getting-started-gemma-4-developer-agent (L4x4, MODEL_NAME = 31B). Run its cells 1-4
# (wheelhouse, data, model, vLLM server -> `models`), SKIP its sample-evaluation cell, then run G1.
import glob, importlib, inspect, json, re, sys
from pathlib import Path

hits = glob.glob('/kaggle/input/**/gemma-paper-code/harness/run_experiment.py', recursive=True)
CODE = Path(hits[0]).parents[1]
sys.path.insert(0, str(CODE / 'harness'))
import build_conditions, run_experiment, pc_overlay
for m in (build_conditions, run_experiment, pc_overlay):
    importlib.reload(m)
print('CODE =', CODE, '| MODEL_NAME =', MODEL_NAME, '| models registry:', type(models).__name__)

# 1. Official TaskResult fields + evaluate_task signature (run_batch adapts to both)
from swegemma.models import TaskResult
from swegemma.evaluate import Evaluator
fields = getattr(TaskResult, 'model_fields', None) or getattr(TaskResult, '__dataclass_fields__', None) or vars(TaskResult)
print('\nTaskResult fields:', sorted(fields)[:60])
print('evaluate_task:', inspect.signature(Evaluator.evaluate_task), '| async:', inspect.iscoroutinefunction(Evaluator.evaluate_task))

# 2. Build C0 + C3 from the official sample agent and validate them with the official validator
AG = build_conditions.build_all(DATA_DIR / 'sample_submission', WORKING_DIR / 'agents', model=MODEL_NAME,
                                conditions=['C0', 'C3'])
from adk_submission import validate_directory
from swegemma.config import build_submission_limits
limits, _ = build_submission_limits()
for c, p in AG.items():
    try:
        out = validate_directory(Path(p), limits=limits)
        print(f'{c}: validate_directory OK -> {str(out)[:200]}')
    except Exception as e:
        print(f'{c}: validate_directory FAILED: {e!r}')
    y = (Path(p) / 'agent.yaml').read_text()
    tools_block = y.split('\nskills:')[0]
    print(f'   tools: {re.findall(r"^\s*-\s*([a-z_]+)\s*$", tools_block, re.M)}')
    print(f'   skills: key = {re.findall(r"^\s*-\s*(skills/[\w-]+)", y, re.M)} | '
          f'dirs = {sorted(x.name for x in (Path(p) / "skills").glob("*")) if (Path(p) / "skills").exists() else []}')

# 3. How the harness runs skills (where does run_skill_script execute? which python?)
import swegemma, google.adk as adk
print('\nwhere run_skill_script is defined/used (3 lines of context, max 4 hits per file):')
for base in (Path(swegemma.__file__).parent, Path(adk.__file__).parent):
    for f in sorted(base.rglob('*.py')):
        L = f.read_text(errors='ignore').splitlines()
        hits = [i for i, l in enumerate(L) if 'run_skill_script' in l]
        for i in hits[:4]:
            print(f'--- {f.relative_to(base.parent)}:{i+1}')
            for j in range(max(0, i - 3), min(len(L), i + 4)):
                print(f'{j+1:5}: {L[j][:150]}')
