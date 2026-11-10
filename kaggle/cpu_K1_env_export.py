# =============================== K1: export the Kaggle notebook environment (CPU; after cells 1 and 2) ===============================
# Writes /kaggle/working/env_export/ (small text files) and prints sizes of what a Modal copy would need.
import importlib.metadata as md, json, os, platform, subprocess, sys
from pathlib import Path

OUT = WORKING_DIR / 'env_export'
OUT.mkdir(exist_ok=True)

# 1. Exact package list of the interpreter the sandbox venvs inherit (system_site_packages + _host_env.pth)
freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True).stdout
(OUT / 'freeze.txt').write_text(freeze)
print('python:', sys.version.split()[0], '| executable:', sys.executable, '| platform:', platform.platform())
print('packages in freeze:', len(freeze.splitlines()))

# 2. pytest plugins (entry-point group pytest11): pytest auto-loads these in EVERY task's test run
plugins = sorted({f'{ep.dist.name}=={ep.dist.version}' for ep in md.entry_points(group='pytest11')})
(OUT / 'pytest_plugins.txt').write_text('\n'.join(plugins) + '\n')
print('pytest11 plugins:', plugins)

# 3. Key versions (harness + libraries the control tasks touch)
keys = ['swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'litellm', 'vllm', 'torch', 'pytest',
        'fastapi', 'starlette', 'pydantic', 'pydantic-core', 'rich', 'pygments', 'ipython', 'httpx', 'anyio',
        'typing-extensions', 'markdown-it-py', 'python-multipart', 'jinja2', 'email-validator', 'orjson', 'ujson']
ver = {}
for k in keys:
    try:
        ver[k] = md.version(k)
    except md.PackageNotFoundError:
        ver[k] = None
(OUT / 'key_versions.json').write_text(json.dumps(ver, indent=2))
print('key versions:', ver)

# 4. Sizes of what Modal would need (nothing is copied yet)
def du(p):
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else (p.stat().st_size if p.exists() else 0)
import glob
wh = glob.glob('/kaggle/input/**/gemma-4-developer-agent-wheelhouse', recursive=True)
CONTROL = json.loads(open(glob.glob('/kaggle/input/**/gemma-paper-code/configs/control_tasks.json', recursive=True)[0]).read())['control_tasks']
snaps = [DATA_DIR / 'snapshots' / f'{i}.tgz' for i in CONTROL]
print('\nsizes (MB): wheelhouse', round(du(wh[0]) / 1e6, 1) if wh else 'not found',
      '| sample_submission', round(du(DATA_DIR / 'sample_submission') / 1e6, 2),
      '| sandbox/', round(du(DATA_DIR / 'sandbox') / 1e6, 2),
      '| 30 control snapshots', round(sum(du(s) for s in snaps) / 1e6, 1),
      '| missing snapshots:', [s.name for s in snaps if not s.exists()])
print('files written:', sorted(p.name for p in OUT.iterdir()))
