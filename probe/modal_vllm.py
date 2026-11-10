"""Run the knowledge probe against the competition checkpoint on a Modal L40S.

Design (privacy + cost):
  * The weights are downloaded ONCE by a CPU-only function into a Modal Volume (no GPU billed
    while downloading 21.7 GB).
  * The GPU function starts vLLM on 127.0.0.1 inside the container and runs
    probe/knowledge_probe.py in the SAME container. No public endpoint is exposed; only the
    result files are returned and written to probe/results/ on the laptop.
  * Serving flags mirror the official Kaggle notebook (gemma4 tool + reasoning parsers,
    max_model_len 32768, gpu_memory_utilization 0.90), with tensor parallel 1 (one L40S).

Commands (from the project folder):
  python -m modal run probe/modal_vllm.py::fetch             # CPU only, once (~21.7 GB) + sha256
  python -m modal run probe/modal_vllm.py::main --limit 1    # smoke test: 1 horizon + 1 probe
  python -m modal run probe/modal_vllm.py::main              # full probe
  (`::main` is required because the file has two local entrypoints; verified with --help)

Facts checked 9 Nov 2026: Modal client 1.6.3 API; image tag vllm/vllm-openai:v0.33.0 exists on
Docker Hub; vLLM 0.33.0 registers the `gemma4` tool and reasoning parsers; the HF repo
google/gemma-4-31B-it-qat-w4a16-ct is ungated, Apache-2.0, 21.7 GB.
UNVERIFIED until the smoke test: that the vLLM image works as a Modal base (python on PATH).
"""
from __future__ import annotations

import pathlib

import modal

MODEL_REPO = "google/gemma-4-31B-it-qat-w4a16-ct"
SERVED_NAME = "gemma-4-31b-it-qat-w4a16-ct"
VLLM_IMAGE = "vllm/vllm-openai:v0.33.0"
MODELS_DIR = "/models"
MODEL_DIR = f"{MODELS_DIR}/{MODEL_REPO.split('/')[-1]}"
PROBE_LOCAL = pathlib.Path(__file__).parent / "knowledge_probe.py"
RESULTS_LOCAL = pathlib.Path(__file__).parent / "results"

app = modal.App("gemma-knowledge-probe")
volume = modal.Volume.from_name("gemma4-weights", create_if_missing=True)

# Modal requires `python` (and pip) on PATH in registry images. The vLLM v0.33.0 runtime stage
# installs Python 3.12 only as /usr/bin/python3 (docker/Dockerfile line 792, checked 9 Nov), so the
# first build failed with "unable to determine the version of Python". Add the aliases first.
PY_ALIASES = [
    "RUN ln -sf /usr/bin/python3 /usr/local/bin/python"
    " && (command -v pip >/dev/null 2>&1"
    " || { printf '#!/bin/sh\\nexec python3 -m pip \"$@\"\\n' > /usr/local/bin/pip && chmod +x /usr/local/bin/pip; })"
    " && python --version && pip --version",  # fail the build here if either is still missing
]

image = (
    modal.Image.from_registry(VLLM_IMAGE, setup_dockerfile_commands=PY_ALIASES)
    .entrypoint([])  # the image's default entrypoint starts a server; we start it ourselves
    .add_local_file(PROBE_LOCAL, "/root/knowledge_probe.py")
)


@app.function(image=image, volumes={MODELS_DIR: volume}, cpu=4, memory=8192, timeout=3600)
def download() -> dict:
    """CPU-only: fetch the checkpoint into the Volume and return file sizes + sha256."""
    import hashlib
    import os

    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
    volume.commit()
    out = {}
    for p in sorted(pathlib.Path(MODEL_DIR).rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            out[str(p.relative_to(MODEL_DIR))] = dict(bytes=os.path.getsize(p), sha256=h.hexdigest())
    return out


@app.function(image=image, volumes={MODELS_DIR: volume}, gpu="L40S", cpu=4, memory=32768, timeout=3600)
def probe(limit: int | None = None, max_tokens: int = 8192, modes: tuple = ("nothink", "think"),
          probe_set: str = "1") -> dict:
    import json
    import subprocess
    import sys
    import time
    import urllib.request

    if not pathlib.Path(MODEL_DIR, "config.json").exists():
        raise RuntimeError("weights missing: run `python -m modal run probe/modal_vllm.py::fetch` first")

    vllm_version = subprocess.run([sys.executable, "-c", "import vllm; print(vllm.__version__)"],
                                  capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"vLLM {vllm_version} on {gpu}", flush=True)

    cmd = ["vllm", "serve", MODEL_DIR, "--served-model-name", SERVED_NAME,
           "--host", "127.0.0.1", "--port", "8000",
           "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
           "--tensor-parallel-size", "1",
           "--enable-auto-tool-choice", "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4"]
    server = subprocess.Popen(cmd)
    t0 = time.time()
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {server.returncode} (see logs above)")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if time.time() - t0 > 1500:
                raise RuntimeError("vLLM not healthy after 25 min")
            time.sleep(5)
        load_s = round(time.time() - t0)
        print(f"vLLM ready after {load_s}s", flush=True)

        # Results go to the Volume as well, so a dropped laptop connection (run with --detach)
        # can't lose them. Download: python -m modal volume get gemma4-weights probe_results/<run> probe/results/
        run_id = time.strftime("%Y%m%d_%H%M%S")
        out_dir = f"{MODELS_DIR}/probe_results/{run_id}"
        print(f"results dir on volume: probe_results/{run_id}", flush=True)
        rcs, secs = {}, {}
        # Both modes in one container (model loads once). Thinking off = template default.
        for mode in modes:
            args = [sys.executable, "/root/knowledge_probe.py", "--base-url", "http://127.0.0.1:8000/v1",
                    "--key-env", "UNUSED_KEY", "--model", SERVED_NAME, "--out", out_dir,
                    "--max-tokens", str(max_tokens),
                    "--label", f"(vLLM {vllm_version}, {gpu}, Modal, QAT w4a16)"]
            if mode == "think":
                args.append("--thinking")
            if limit:
                args += ["--limit", str(limit)]
            args += ["--set", probe_set]
            t1 = time.time()
            rcs[mode] = subprocess.run(args).returncode
            secs[mode] = round(time.time() - t1)
            volume.commit()  # persist after each mode
        files = {p.name: p.read_text(encoding="utf-8") for p in pathlib.Path(out_dir).glob("*")} \
            if pathlib.Path(out_dir).exists() else {}
        return dict(rc=rcs, vllm_version=vllm_version, gpu=gpu, load_s=load_s, probe_s=secs,
                    volume_dir=f"probe_results/{run_id}", files=files)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


@app.local_entrypoint()
def fetch():
    """`python -m modal run probe/modal_vllm.py::fetch`: download once, save sha256 locally."""
    import json

    r = download.remote()
    out = pathlib.Path(__file__).parent.parent / "configs" / "model_sha256_hf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": MODEL_REPO, "files": r}, indent=2), encoding="utf-8")
    total = sum(v["bytes"] for v in r.values())
    print(f"{len(r)} files, {total / 1e9:.1f} GB -> {out}")


@app.local_entrypoint()
def main(limit: int = 0, max_tokens: int = 8192, modes: str = "nothink,think", probe_set: str = "1"):
    import json

    ms = tuple(m.strip() for m in modes.split(",") if m.strip())
    assert set(ms) <= {"nothink", "think"}, f"modes must be nothink and/or think, got {ms}"
    assert probe_set in ("1", "2", "3"), f"probe_set must be 1, 2 or 3, got {probe_set}"
    r = probe.remote(limit=limit or None, max_tokens=max_tokens, modes=ms, probe_set=probe_set)
    RESULTS_LOCAL.mkdir(parents=True, exist_ok=True)
    for name, text in r.pop("files").items():
        (RESULTS_LOCAL / name).write_text(text, encoding="utf-8")
        print(f"saved {RESULTS_LOCAL / name}")
    print(json.dumps(r, indent=2))
