# =============================== CELL D: does the sandbox install wheels at all? (read-only) ===============================
import inspect, re, subprocess, sys
from pathlib import Path
import swegemma
from swegemma.harness import container_setup as cs
from swegemma.sandbox import subprocess as sbx

root = Path(swegemma.__file__).parent

# 1. The wheel-install step the harness runs per task
for name in ('setup_container_wheels', 'install_test_dependencies'):
    fn = getattr(cs, name, None)
    if fn:
        print(f'##### container_setup.{name}\n' + inspect.getsource(fn)[:3500])

# 2. How exec() builds the environment of every sandbox command (PYTHONPATH etc.)
src = inspect.getsource(sbx).splitlines()
print('\n##### sandbox/subprocess.py 300-500')
for j in range(299, min(500, len(src))):
    print(f'{j+1:4}: {src[j]}')

# 3. Host python: is ensurepip / pip available? (explains "venv without pip")
for cmd in (['/usr/bin/python3', '-c', 'import ensurepip; print("ensurepip ok")'],
            ['/usr/bin/python3', '-m', 'pip', '--version'],
            [sys.executable, '-m', 'pip', '--version']):
    r = subprocess.run(cmd, capture_output=True, text=True)
    print('$', ' '.join(cmd[:3]), '->', (r.stdout or r.stderr).strip()[:160])
print('notebook sys.executable:', sys.executable)
