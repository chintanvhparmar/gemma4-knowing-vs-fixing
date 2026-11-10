# =============================== CELL C: why did empty patches pass? (read-only) ===============================
# Run after CELL B. Reads B's output file, the host package versions and the sandbox setup code.
import importlib.metadata as md, json, re
from pathlib import Path
import swegemma

root = Path(swegemma.__file__).parent

# 1. Versions installed in the NOTEBOOK (host) itself
print('host versions:')
for p in ('marshmallow', 'polars', 'click', 'scikit-learn', 'Django', 'networkx', 'pandas', 'starlette', 'httpx', 'pytest', 'numpy'):
    try:
        print(f'  {p:13} {md.version(p)}')
    except md.PackageNotFoundError:
        print(f'  {p:13} (not installed)')

# 2. Test output tails from CELL B for the informative cases
recs = {(r['mode'], r['id']): r for r in map(json.loads, (WORKING_DIR / 'pc_official_check.jsonl').read_text().splitlines())}
for key in (('empty', 'pc_marshmallow_validates'), ('gold', 'pc_marshmallow_context'), ('gold', 'pc_django_checkconstraint'),
            ('empty', 'pc_networkx_node_link')):
    r = recs[key]
    print(f'\n===== {key}  resolved={r["resolved"]} exit={r["exit_code"]} error={str(r["error"])[:200]}')
    print(r['test_tail'][-1500:])

# 3. How the sandbox venv is created (system site-packages? host python?)
for f in sorted((root / 'sandbox').glob('*.py')):
    L = f.read_text(errors='ignore').splitlines()
    for i, l in enumerate(L):
        if re.search(r'system.site.packages|venv|PYTHONPATH|sys\.path|site-packages', l):
            print(f'sandbox/{f.name}:{i+1}: {l.strip()[:150]}')

# 4. sandbox/setup.py: the fast path ("pre-baked wheels") and what it installs
src = (DATA_DIR / 'sandbox' / 'setup.py').read_text().splitlines()
print('\n##### sandbox/setup.py 30-60 and 330-400')
for j in list(range(29, 60)) + list(range(329, 400)):
    if j < len(src):
        print(f'{j+1:4}: {src[j]}')
