"""Step 1 of the Kaggle replica: is Kaggle's public CPU image `v170` the environment behind the
notebook setting "Pin 2026-07-01"?

Test: run `pip freeze --all` inside gcr.io/kaggle-images/python:<tag> on Modal (CPU only) and
compare with the freeze exported from the real notebook (kaggle/cpu_K1_env_export.py).
Expected if it is the right image: identical, except the 41 packages installed by the notebook's
cell 1 from the competition wheelhouse (and what they replaced).

    python -m modal run modal_replica/check_kaggle_image.py --tag v170 --k1-freeze <path to freeze.txt>

Why v170: tag list of gcr.io/kaggle-images/python (checked 9 Nov 2026): v169 uploaded 2026-05-25,
v170 uploaded 2026-06-29, next tagged image after that is none until August. UNVERIFIED until this runs.
"""
from __future__ import annotations

import pathlib
import re

import modal

app = modal.App("kaggle-image-check")


import os

# CPU image by default; set KAGGLE_IMAGE_REPO=kaggle-gpu-images/python for the GPU notebook image
# (only `pip freeze` runs, so no GPU is attached even for the GPU image).
REPO = os.environ.get("KAGGLE_IMAGE_REPO", "kaggle-images/python")


def _image(tag: str) -> modal.Image:
    # Modal needs `python` on PATH; only add an alias if the image lacks one (don't change the image otherwise).
    return modal.Image.from_registry(
        f"gcr.io/{REPO}:{tag}",
        setup_dockerfile_commands=["RUN command -v python >/dev/null 2>&1 || ln -sf \"$(command -v python3)\" /usr/local/bin/python"],
    ).entrypoint([])


TAG = "v170"
image = _image(TAG)


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def freeze() -> dict:
    import platform
    import subprocess
    import sys

    run = lambda *a: subprocess.run(list(a), capture_output=True, text=True).stdout
    return dict(
        python=sys.version.split()[0], executable=sys.executable,
        glibc=" ".join(platform.libc_ver()), os=run("cat", "/etc/os-release"),
        freeze=run(sys.executable, "-m", "pip", "freeze", "--all"),
    )


def _parse(text: str) -> dict[str, str]:
    """name (normalised) -> version-or-reference string."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name, ref = line.split(" @ ", 1)
            out[re.sub(r"[-_.]+", "-", name).lower()] = "@ " + ref.split("#sha256=")[0]
        elif "==" in line:
            name, ver = line.split("==", 1)
            out[re.sub(r"[-_.]+", "-", name).lower()] = ver
    return out


@app.local_entrypoint()
def main(k1_freeze: str, tag: str = TAG):
    import json

    assert tag == TAG, f"edit TAG in the file to {tag} (the image is defined at import time)"
    r = freeze.remote()
    out_dir = pathlib.Path(__file__).parent / "env"
    out_dir.mkdir(exist_ok=True)
    tag = f"{tag}_gpu" if "gpu" in REPO else tag  # file names only
    (out_dir / f"freeze_{tag}.txt").write_text(r["freeze"], encoding="utf-8")

    img = _parse(r["freeze"])
    nb = _parse(pathlib.Path(k1_freeze).read_text(encoding="utf-8"))
    wheelhouse = {k for k, v in nb.items() if v.startswith("@ file:///tmp/wheelhouse/")}
    same = [k for k in nb if k in img and nb[k] == img[k]]
    diff = {k: (img[k], nb[k]) for k in nb if k in img and nb[k] != img[k]}
    only_nb = sorted(k for k in nb if k not in img)
    only_img = sorted(k for k in img if k not in nb)

    print(f"image {tag}: python {r['python']} ({r['executable']}), glibc {r['glibc']}")
    print(re.search(r'PRETTY_NAME="([^"]+)"', r["os"]).group(1) if "PRETTY_NAME" in r["os"] else r["os"][:80])
    print(f"packages: image {len(img)} | notebook {len(nb)} | identical {len(same)}")
    print(f"different version: {len(diff)} (of which wheelhouse-installed: {len(set(diff) & wheelhouse)})")
    for k, (a, b) in sorted(diff.items()):
        if k not in wheelhouse:
            print(f"   NOT EXPLAINED  {k}: image {a} | notebook {b}")
    print(f"only in notebook: {len(only_nb)} (wheelhouse: {len(set(only_nb) & wheelhouse)})")
    print("   not from wheelhouse:", [k for k in only_nb if k not in wheelhouse])
    print(f"only in image: {len(only_img)}:", only_img[:40])
    summary = dict(tag=tag, python=r["python"], glibc=r["glibc"], n_image=len(img), n_notebook=len(nb),
                   identical=len(same), diff_unexplained={k: v for k, v in diff.items() if k not in wheelhouse},
                   only_notebook_not_wheelhouse=[k for k in only_nb if k not in wheelhouse], only_image=only_img)
    (out_dir / f"compare_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", out_dir / f"freeze_{tag}.txt", "and", out_dir / f"compare_{tag}.json")
