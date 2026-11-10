# =============================== K3: GPU-notebook environment export + gold check (NO model needed) ===============================
# Run in the GPU notebook (Accelerator GPU L4 x4, "Pin to original environment (2026-07-01)") right after the
# official cells 1 (wheelhouse) and 2 (DATA_DIR, WORKING_DIR). Do NOT start vLLM: nothing here uses the GPU,
# but the GPU notebook's image is the environment we must compare with the Modal replica.
# Takes ~15-25 min. Outputs (download all three from /kaggle/working/):
#   env_export/freeze.txt, env_export/pytest_plugins.txt, env_export/key_versions.json  (K1, GPU image)
#   gold_check_kaggle_gpu.jsonl                                                          (gold/empty, 42 tasks)
import glob, sys, zipfile
from pathlib import Path

hits = glob.glob('/kaggle/input/**/tasks_postcutoff/out/tasks.jsonl', recursive=True)
if not hits:
    zips = glob.glob('/kaggle/input/**/gemma-paper-code*.zip', recursive=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall('/kaggle/working/code')
        hits = glob.glob('/kaggle/working/code/tasks_postcutoff/out/tasks.jsonl')
CODE = Path(hits[0]).parents[2]
ver = CODE / 'CODE_VERSION.txt'
print('CODE =', CODE, '| CODE_VERSION', ver.read_text().strip() if ver.exists() else 'MISSING (old upload?)')

# 1. K1 environment export (same script as the CPU run)
exec(open(CODE / 'kaggle' / 'cpu_K1_env_export.py').read())

# 2. Gold check, same code as on Modal (harness/gold_check.py)
sys.path.insert(0, str(CODE / 'harness'))
import gold_check
gold_check.run(DATA_DIR, WORKING_DIR, CODE, WORKING_DIR / 'gold_check_kaggle_gpu.jsonl', platform='kaggle-gpu-notebook')
