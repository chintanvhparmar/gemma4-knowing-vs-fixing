"""Runs at image build time. Reproduces the official notebook's cell 1 (pasted 9 Nov):
  1. delete *cutlass*.pth hooks in /usr/local/lib/python*/{dist,site}-packages
  2. put every wheelhouse wheel (skip names containing "cutlass") into /tmp/wheelhouse
     (cell 1 also renames 'cu128' -> '+cu128'; the manifest already has the restored names,
     taken from the notebook's own pip freeze)
  3. pip install -q --no-deps --force-reinstall <all wheels>
Every wheel is checked against the sha256 from the notebook's freeze before installing.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path("/opt/replica/wheelhouse_manifest.json")
LOCAL = Path("/opt/replica/wheelhouse")
TMP = Path("/tmp/wheelhouse")

for pat in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth", "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    for pth in glob.glob(pat):
        print("removing", pth)
        os.unlink(pth)

TMP.mkdir(parents=True, exist_ok=True)
items = json.loads(MANIFEST.read_text())
for it in items:
    if "cutlass" in it["file"].lower():
        continue
    dst = TMP / it["file"]
    if it["source"] == "local":
        shutil.copyfile(LOCAL / it["file"], dst)
    else:
        req = urllib.request.Request(it["url"], headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(req, timeout=300) as r, dst.open("wb") as f:
            shutil.copyfileobj(r, f)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    if h != it["sha256"]:
        sys.exit(f"sha256 mismatch for {it['file']}: {h}")
print(f"{len(items)} wheels verified in {TMP}")

wheels = sorted(str(w) for w in TMP.glob("*.whl"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels], check=True)
print("Wheelhouse installation complete.")
