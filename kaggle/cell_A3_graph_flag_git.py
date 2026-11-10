# =============================== CELL A3: what graph_available changes + how git is used (read-only) ===============================
import re
from pathlib import Path
import swegemma

root = Path(swegemma.__file__).parent

def lines(relpath, a, b):
    L = (root / relpath).read_text(errors='ignore').splitlines()
    print(f'\n##### {relpath} lines {a}-{b}')
    for j in range(a - 1, min(b, len(L))):
        print(f'{j+1:5}: {L[j]}')

def grep(relpath, pattern, ctx=2, maxhits=8):
    L = (root / relpath).read_text(errors='ignore').splitlines()
    hits = [i for i, l in enumerate(L) if re.search(pattern, l)]
    print(f'\n##### {relpath}  ({len(hits)} hits for /{pattern}/)')
    for i in hits[:maxhits]:
        for j in range(max(0, i - ctx), min(len(L), i + ctx + 1)):
            print(f'{j+1:5}: {L[j]}')
        print('   ...')

# 1. Everything that graph_available controls (prompt text? tool registration?)
lines('harness/agent_runner.py', 434, 470)
grep('harness/agent_runner.py', r'graph_available', 3, 10)

# 2. How the agent's patch is produced (git diff?) and how the snapshot is prepared
grep('tools/execution.py', r'git', 2, 8)
grep('harness/container_setup.py', r'git', 1, 18)

# 3. How the base commit is used (checkout? compared?)
grep('models/task.py', r'base_commit|git', 2, 4)
grep('harness/verification.py', r'git', 2, 4)

# 4. Which tool names the sample agent.yaml declares
print('\n##### sample_submission/agent.yaml tools section')
y = (DATA_DIR / 'sample_submission' / 'agent.yaml').read_text()
m = re.search(r'(?ms)^\s*tools:.*?(?=^\S|\Z)', y)
print(m.group(0) if m else y[:1500])
