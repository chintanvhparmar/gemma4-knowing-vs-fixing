# =============================== G1d: exact `skills:` format (read-only, CPU) ===============================
import re
from pathlib import Path
import importlib.util

ad = Path(importlib.util.find_spec('adk_submission').origin).parent

def lines(path, a, b):
    L = Path(path).read_text(errors='ignore').splitlines()
    print(f'\n##### {Path(path).name} lines {a}-{b}')
    for j in range(a - 1, min(b, len(L))):
        print(f'{j+1:4}: {L[j][:170]}')

# 1. How loaded_skills is built from the config, and whether skill tool NAMES in `tools:` matter
src = (ad / 'resolvers' / 'tools.py').read_text(errors='ignore').splitlines()
start = max(0, next((i for i, l in enumerate(src) if 'loaded_skills' in l), 200) - 40)
lines(ad / 'resolvers' / 'tools.py', start + 1, start + 75)

# 2. The schema field
lines(ad / 'schema.py', 280, 330)

# 3. The HARNESS_README: the agent.yaml example around "skills", and the full tree example
t = (DATA_DIR / 'HARNESS_README.md').read_text(errors='ignore')
for m in re.finditer(r'skills', t):
    print('\n---- README @', m.start(), '\n' + t[max(0, m.start() - 700): m.start() + 500])
    break
i = t.find('repo_navigation')
print('\n---- README tree around repo_navigation\n' + t[max(0, i - 1500): i + 200])
