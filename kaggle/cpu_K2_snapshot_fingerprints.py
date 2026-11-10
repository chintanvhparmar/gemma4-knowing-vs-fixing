# =============================== K2: fingerprints of the 30 control snapshots (CPU; after cells 1-2) ===============================
# For each official snapshot: HEAD commit, HEAD^{tree} hash, uncommitted changes, untracked files.
# On Modal we clone the public repo at base_commit; identical tree hash + clean status = identical code.
import glob, json, subprocess, tarfile, tempfile
from pathlib import Path

CONTROL = json.loads(open(glob.glob('/kaggle/input/**/gemma-paper-code/configs/control_tasks.json', recursive=True)[0]).read())['control_tasks']
by_id = {t.instance_id: t for t in tasks}
res = {}
for i, tid in enumerate(CONTROL, 1):
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(DATA_DIR / 'snapshots' / f'{tid}.tgz') as tf:
            tf.extractall(td, filter='data')
        g = lambda *a: subprocess.run(['git', '-C', td, *a], capture_output=True, text=True).stdout.strip()
        res[tid] = dict(repo=by_id[tid].repo, base_commit=by_id[tid].base_commit,
                        head=g('rev-parse', 'HEAD'), tree=g('rev-parse', 'HEAD^{tree}'),
                        status=g('status', '--porcelain', '--untracked-files=all')[:2000],
                        n_files=len(g('ls-files').splitlines()))
    r = res[tid]
    print(f"[{i:2}/30] {tid:16} head==base: {r['head'] == r['base_commit']}  tree={r['tree'][:12]}  "
          f"files={r['n_files']}  dirty_lines={len(r['status'].splitlines())}", flush=True)
out = WORKING_DIR / 'env_export' / 'control_snapshot_fingerprints.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print('\nwritten', out, '| all head==base_commit:', all(r['head'] == r['base_commit'] for r in res.values()),
      '| all clean:', all(not r['status'] for r in res.values()))
