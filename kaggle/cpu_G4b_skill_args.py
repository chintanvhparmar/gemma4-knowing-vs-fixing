# =============================== G4b: how run_skill_script passes arguments (read-only, CPU) ===============================
import re
from pathlib import Path
import google.adk as adk

st = Path(adk.__file__).parent / 'tools' / 'skill_toolset.py'
L = st.read_text(errors='ignore').splitlines()

# 1. The tool's declared parameters (what the model sees)
start = next(i for i, l in enumerate(L) if 'name="run_skill_script"' in l)
print(f'##### RunSkillScriptTool, lines {start - 5}-{start + 110}')
for j in range(start - 5, min(start + 110, len(L))):
    print(f'{j+1:4}: {L[j][:170]}')

# 2. How the script command line is built from those arguments
b = next(i for i, l in enumerate(L) if 'import runpy' in l)
print(f'\n##### script runner code, lines {b - 40}-{b + 75}')
for j in range(b - 40, min(b + 75, len(L))):
    print(f'{j+1:4}: {L[j][:170]}')
