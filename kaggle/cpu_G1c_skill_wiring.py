# =============================== G1c: how skills are wired (read-only, CPU notebook is fine) ===============================
# Run in gold-patch-validator after its cell 1 (wheelhouse). Answers: does listing a skill tool create the
# whole SkillToolset (incl. load_skill)? Is a code executor configured, and where do skill scripts run?
import re
from pathlib import Path
import importlib.util

def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    return Path(spec.origin).parent if spec and spec.origin else None

def ctx(path, pattern, n=3, maxhits=6):
    L = Path(path).read_text(errors='ignore').splitlines()
    hits = [i for i, l in enumerate(L) if re.search(pattern, l)]
    if not hits:
        return
    print(f'\n##### {path}  ({len(hits)} hits for /{pattern}/)')
    for i in hits[:maxhits]:
        for j in range(max(0, i - n), min(len(L), i + n + 1)):
            print(f'{j+1:5}: {L[j][:170]}')
        print('   ...')

for name in ('swegemma', 'adk_submission'):
    d = pkg_dir(name)
    print(f'\n======== {name}: {d}')
    for f in sorted(d.rglob('*.py')):
        t = f.read_text(errors='ignore')
        if re.search(r'SkillToolset|code_executor|load_skill\b|skills_dir|["\']skills["\']', t):
            ctx(f, r'SkillToolset\(|code_executor|load_skill\b|skills_dir|["\']skills["\']|_SKILL_TOOL_NAMES', 3, 6)

# The official HARNESS_README section on skills, if present
for p in (DATA_DIR / 'HARNESS_README.md',):
    if p.exists():
        t = p.read_text(errors='ignore')
        for m in re.finditer(r'(?is)#+[^\n]*skill[^\n]*\n.*?(?=\n#+ |\Z)', t):
            print('\n######## HARNESS_README skills section\n' + m.group(0)[:3000])
