"""Build wheelhouse_manifest.json: the 41 wheels the official notebook's cell 1 installs.

Input: a `pip freeze --all` exported from the real Kaggle notebook (kaggle/cpu_K1_env_export.py).
For every `<name> @ file:///tmp/wheelhouse/<file>#sha256=<hash>` line:
  * source "local": the file is in modal_replica/wheelhouse/ (downloaded from the Kaggle
    wheelhouse dataset; not on PyPI, or PyPI has a different build), sha256 must match;
  * source "pypi": PyPI has a file with the SAME name and SAME sha256 (checked via the PyPI
    JSON API); the URL is recorded.
Anything else is an error, so the image can only be built from byte-identical wheels.

    python -I modal_replica/make_manifest.py <freeze.txt>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE / "wheelhouse"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(freeze_path: str) -> None:
    items = []
    for line in Path(freeze_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\S+) @ file:///tmp/wheelhouse/(\S+?)#sha256=([0-9a-f]{64})$", line.strip())
        if m:
            items.append(m.groups())
    out, errors = [], []
    for name, fn, sha in items:
        lp = LOCAL / fn
        if lp.exists():
            if sha256(lp) != sha:
                errors.append(f"{fn}: local file sha256 differs")
            out.append(dict(name=name, file=fn, sha256=sha, source="local"))
            continue
        ver = fn.split("-")[1]
        req = urllib.request.Request(f"https://pypi.org/pypi/{name}/{ver}/json", headers={"User-Agent": "research-script"})
        try:
            files = {u["filename"]: u for u in json.load(urllib.request.urlopen(req, timeout=30))["urls"]}
        except Exception as e:
            errors.append(f"{fn}: PyPI lookup failed ({e})")
            continue
        u = files.get(fn)
        if not u or u["digests"]["sha256"] != sha:
            errors.append(f"{fn}: not on PyPI with the same sha256 -> download it from the wheelhouse dataset")
            continue
        out.append(dict(name=name, file=fn, sha256=sha, source="pypi", url=u["url"]))
    if errors:
        sys.exit("ERRORS:\n  " + "\n  ".join(errors))
    (HERE / "wheelhouse_manifest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} wheels: {sum(o['source'] == 'pypi' for o in out)} from PyPI, "
          f"{sum(o['source'] == 'local' for o in out)} local -> {HERE / 'wheelhouse_manifest.json'}")


if __name__ == "__main__":
    main(sys.argv[1])
