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
