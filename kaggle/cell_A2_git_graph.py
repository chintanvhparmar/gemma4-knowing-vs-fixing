# =============================== CELL A2: git + code-graph details (read-only) ===============================
# Run after CELL A. Prints source excerpts only; changes nothing.
import inspect, re
from pathlib import Path
import swegemma

root = Path(swegemma.__file__).parent

def show(relpath, pattern, ctx=3, maxhits=6):
    p = root / relpath
    L = p.read_text(errors='ignore').splitlines()
    hits = [i for i, l in enumerate(L) if re.search(pattern, l)]
    print(f'\n##### {relpath}  ({len(hits)} hits for /{pattern}/)')
    for i in hits[:maxhits]:
        for j in range(max(0, i - ctx), min(len(L), i + ctx + 1)):
            print(f'{j+1:5}: {L[j]}')
        print('   ...')

# 1. How the harness uses git (patch creation, base commit, checkout)
for f in sorted(root.rglob('*.py')):
    txt = f.read_text(errors='ignore')
    n = len(re.findall(r'["\']git["\']|git (diff|checkout|apply|add|commit|init|reset|status)|base_commit', txt))
    if n:
        print(f'git usage: {f.relative_to(root)}  ({n})')
show('harness/agent_runner.py', r'graph_dir|embeddings_dir', 6, 3)

# 2. Graph file lookup by repo/commit, and what happens if the file is missing
show('graph/graph_utils.py', r'def |\.json|exists\(|FileNotFound|raise|return None|warn', 1, 40)

# 3. Is there a graph BUILDER (to make graphs for our tasks the official way)?
import swegemma.graph as sg
print('\nswegemma.graph members:', [n for n in dir(sg) if not n.startswith('_')])
for f in sorted((root / 'graph').glob('*.py')):
    defs = re.findall(r'^def (\w+)\(', f.read_text(errors='ignore'), re.M)
    print(f'  graph/{f.name}: {defs}')

# 4. Embeddings: file naming and the model used to create them
import os
print('\nembeddings example files:', sorted(os.listdir(DATA_DIR / 'embeddings'))[:3])
show('graph/embedding_utils.py', r'model|SentenceTransformer|encode|dim|256', 1, 12)

# 5. The tools' behaviour when no graph exists for the task
show('tools/graph.py', r'not found|No graph|unavailable|except|return \{|error', 2, 10)
