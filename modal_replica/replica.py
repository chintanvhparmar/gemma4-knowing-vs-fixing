"""Modal replica of the Kaggle GPU notebook environment (for the web-search conditions, which may
not run in Kaggle L4 sessions: "No Internet - All L4 sessions must have internet disabled").

Replica = gcr.io/kaggle-gpu-images/python:v170 (= notebook setting "Pin 2026-07-01"; verified by
check_kaggle_image.py) + the official notebook's cell 1 (install_wheelhouse.py, 41 sha256-checked
wheels from wheelhouse_manifest.json).

Entry points (run from the project folder):
  python -m modal run modal_replica/replica.py::freeze_check --k1-freeze <freeze.txt> --gpu-image-freeze modal_replica/env/freeze_v170_gpu.txt
      Build the image; its pip freeze must equal the v170 GPU image's, except the 41 wheelhouse
      packages, which must equal the notebook's (from K1). CPU only.
"""
from __future__ import annotations

import pathlib
import re

import modal

HERE = pathlib.Path(__file__).resolve().parent
KAGGLE_GPU_IMAGE = "gcr.io/kaggle-gpu-images/python:v170"

app = modal.App("kaggle-replica")

image = (
    modal.Image.from_registry(
        KAGGLE_GPU_IMAGE,
        setup_dockerfile_commands=["RUN command -v python >/dev/null 2>&1 || ln -sf \"$(command -v python3)\" /usr/local/bin/python"],
    )
    .entrypoint([])
    .add_local_dir(HERE / "wheelhouse", "/opt/replica/wheelhouse", copy=True)
    .add_local_file(HERE / "wheelhouse_manifest.json", "/opt/replica/wheelhouse_manifest.json", copy=True)
    .add_local_file(HERE / "install_wheelhouse.py", "/opt/replica/install_wheelhouse.py", copy=True)
    .run_commands("python /opt/replica/install_wheelhouse.py")
)


def _parse(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if " @ " in line:
            name, ref = line.split(" @ ", 1)
            out[re.sub(r"[-_.]+", "-", name).lower()] = "@ " + ref.split("#sha256=")[0]
        elif "==" in line and not line.startswith("#"):
            name, ver = line.split("==", 1)
            out[re.sub(r"[-_.]+", "-", name).lower()] = ver
    return out


@app.function(image=image, cpu=2, memory=4096, timeout=1200)
def _freeze() -> str:
    import subprocess
    import sys

    return subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True).stdout


# ---------------------------------------------------------------- competition data (official bytes)
# Downloaded with the participant's own Kaggle API credentials (Modal secret "kaggle-api", created by
# the user in the Modal dashboard; never printed) into a private Volume. Rule 2.4.b: only the user's
# account can access the Volume; nothing is shared.
COMPETITION = "gemma-4-developer-agent"
DATA = "/data"
data_vol = modal.Volume.from_name("kaggle-replica-data", create_if_missing=True)
kaggle_secret = modal.Secret.from_name("kaggle-api")


@app.function(image=image, secrets=[kaggle_secret], cpu=1, memory=2048, timeout=600)
def _list_files() -> str:
    import subprocess

    r = subprocess.run(["kaggle", "competitions", "files", COMPETITION, "--page-size", "200"],
                       capture_output=True, text=True)
    return r.stdout + ("\nSTDERR: " + r.stderr[-1500:] if r.returncode else "")


@app.function(image=image, secrets=[kaggle_secret], cpu=1, memory=2048, timeout=900)
def _all_files() -> list:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out, token = [], None
    while True:
        r = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        out += [(f.name, int(f.total_bytes or 0)) for f in r.files]
        token = r.next_page_token
        if not token:
            return out


@app.local_entrypoint()
def data_list():
    """List ALL competition files (paged) -> modal_replica/env/competition_files.json + a summary."""
    import collections
    import json

    files = _all_files.remote()
    (HERE / "env").mkdir(exist_ok=True)
    (HERE / "env" / "competition_files.json").write_text(json.dumps(files, indent=0), encoding="utf-8")
    by_dir = collections.defaultdict(lambda: [0, 0])
    for name, size in files:
        parts = name.split("/")
        key = "/".join(parts[:2]) + "/" if len(parts) > 2 else (parts[0] + "/" if len(parts) > 1 else name)
        by_dir[key][0] += 1
        by_dir[key][1] += size
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e9:.2f} GB")
    for k, (n, s) in sorted(by_dir.items()):
        print(f"  {k:45} {n:5} files {s / 1e6:10.1f} MB")


SMALL_PREFIXES = ("tasks.jsonl", "HARNESS_README.md", "docker/", "sample_submission/", "sandbox/", "wheels/")


@app.function(image=image, secrets=[kaggle_secret], volumes={DATA: data_vol}, cpu=2, memory=4096, timeout=3600)
def _fetch(wanted: list, fingerprints: dict) -> dict:
    """Download `wanted` [(name, size)] into /data/<name>; unzip if Kaggle served a zip; check size.
    Then for each snapshot with a K2 fingerprint: extract, compare HEAD^{tree} and git status."""
    import os
    import shutil
    import subprocess
    import tarfile
    import tempfile
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    def one(item):
        name, size = item
        dst = Path(DATA) / name
        if dst.exists() and dst.stat().st_size == size:
            return name, "cached"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            api.competition_download_file(COMPETITION, name, path=td, force=True, quiet=True)
            got = [Path(td) / f for f in os.listdir(td)]
            if len(got) != 1:
                return name, f"ERROR: {len(got)} files downloaded"
            f = got[0]
            if f.name == dst.name + ".zip" or (f.suffix == ".zip" and not name.endswith(".zip")):
                with zipfile.ZipFile(f) as z:
                    members = z.namelist()
                    if len(members) != 1:
                        return name, f"ERROR: zip with {len(members)} members"
                    z.extract(members[0], td)
                    f = Path(td) / members[0]
            shutil.move(str(f), dst)
        return name, "ok" if dst.stat().st_size == size else f"SIZE MISMATCH {dst.stat().st_size} != {size}"

    with ThreadPoolExecutor(4) as pool:
        status = dict(pool.map(one, wanted))
    data_vol.commit()

    checks = {}
    for name, _ in wanted:
        tid = Path(name).name.removesuffix(".tgz")
        if not name.startswith("snapshots/") or tid not in fingerprints:
            continue
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(Path(DATA) / name) as tf:
                tf.extractall(td, filter="data")
            g = lambda *a: subprocess.run(["git", "-c", f"safe.directory={td}", "-C", td, *a],
                                          capture_output=True, text=True).stdout.strip()
            fp = fingerprints[tid]
            checks[tid] = dict(head=g("rev-parse", "HEAD") == fp["head"], tree=g("rev-parse", "HEAD^{tree}") == fp["tree"],
                               clean=g("status", "--porcelain", "--untracked-files=all") == fp["status"])
    return dict(status=status, checks=checks)


@app.local_entrypoint()
def data_fetch(limit: int = 0):
    """Download the small official files + the 30 control snapshots to the Volume and verify them
    against the listing (sizes) and K2 (HEAD, tree hash, clean status). --limit N: only N snapshots."""
    import json

    files = json.loads((HERE / "env" / "competition_files.json").read_text(encoding="utf-8"))
    control = json.loads((HERE.parent / "configs" / "control_tasks.json").read_text(encoding="utf-8"))["control_tasks"]
    fps = json.loads((HERE / "env" / "control_snapshot_fingerprints.json").read_text(encoding="utf-8"))
    assert set(control) == set(fps), "K2 fingerprints do not cover exactly the control tasks"
    small = [(n, s) for n, s in files if n.startswith(SMALL_PREFIXES)]
    snaps = [(n, s) for n, s in files if n in {f"snapshots/{t}.tgz" for t in control}]
    assert len(snaps) == len(control), f"found {len(snaps)} control snapshots in the listing"
    if limit:
        snaps = snaps[:limit]
    print(f"fetching {len(small)} small files + {len(snaps)} snapshots "
          f"({sum(s for _, s in small + snaps) / 1e9:.2f} GB)")
    r = _fetch.remote(small + snaps, fps)
    bad = {k: v for k, v in r["status"].items() if v not in ("ok", "cached")}
    print(f"downloads: {len(r['status'])} | ok {sum(v == 'ok' for v in r['status'].values())} | "
          f"cached {sum(v == 'cached' for v in r['status'].values())} | problems: {bad}")
    allok = all(all(c.values()) for c in r["checks"].values())
    for tid, c in sorted(r["checks"].items()):
        print(f"  {tid:16} head {c['head']}  tree {c['tree']}  clean {c['clean']}")
    print(f"SNAPSHOTS IDENTICAL TO KAGGLE (K2): {allok} ({len(r['checks'])} checked)")


# ---------------------------------------------------------------- gold check (same code as on Kaggle)
ROOT = HERE.parent
CODE = "/code"
_skip = lambda p: "__pycache__" in str(p)
code_image = (
    image
    .add_local_dir(ROOT / "harness", f"{CODE}/harness", ignore=_skip)
    .add_local_file(ROOT / "configs" / "control_tasks.json", f"{CODE}/configs/control_tasks.json")
    .add_local_file(ROOT / "tasks_postcutoff" / "out" / "tasks.jsonl", f"{CODE}/tasks_postcutoff/out/tasks.jsonl")
    .add_local_dir(ROOT / "tasks_postcutoff" / "out" / "snapshots", f"{CODE}/tasks_postcutoff/out/snapshots")
    .add_local_dir(ROOT / "tasks_postcutoff" / "out" / "wheels", f"{CODE}/tasks_postcutoff/out/wheels")
)

# Environment variables set by the official notebook's cell 1 (same values)
CELL1_ENV = {
    "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_NO_USAGE_STATS": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENGINE_READY_TIMEOUT_S": "1200",
    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "NCCL_DEBUG": "WARN", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_TELEMETRY": "False", "LITELLM_LOG": "WARNING", "OPENAI_API_KEY": "EMPTY", "LOCAL_API_KEY": "EMPTY",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "OTEL_SDK_DISABLED": "true", "OTEL_PYTHON_DISABLED": "true",
}


@app.function(image=code_image, volumes={DATA: data_vol}, cpu=4, memory=16384, timeout=4 * 3600)
def _gold(limit: int, run_id: str) -> str:
    import os
    import sys

    os.environ.update(CELL1_ENV)
    sys.path.insert(0, f"{CODE}/harness")
    import gold_check

    out = pathlib.Path(DATA) / "gold_check" / f"modal_{run_id}.jsonl"
    gold_check.run(DATA, "/tmp/work", CODE, out, platform="modal-replica-v170-gpu-image", limit=limit)
    data_vol.commit()
    return out.read_text()


@app.local_entrypoint()
def gold_check(limit: int = 0):
    """Grade gold + empty patches for the 30 control + 12 post-cutoff tasks in the replica (CPU only).
    --limit N: first N tasks of each group (smoke test). Output also kept on the Volume."""
    import time

    run_id = time.strftime("%Y%m%d_%H%M%S") + (f"_limit{limit}" if limit else "")
    text = _gold.remote(limit, run_id)
    dst = HERE / "env" / f"gold_check_modal_{run_id}.jsonl"
    dst.write_text(text, encoding="utf-8")
    print("saved", dst)


# ---------------------------------------------------------------- agent runs on 4x L4 (= Kaggle L4x4, tp=4)
# Same hardware type and tensor parallelism as the Kaggle notebook, so generation speed (and therefore
# what fits in the 15-minute per-task budget) matches. Decided 9 Nov (L4 $0.000222/s on modal.com/pricing).
weights_vol = modal.Volume.from_name("gemma4-weights")
MODELS = "/models"
HF_MODEL_DIR = f"{MODELS}/gemma-4-31B-it-qat-w4a16-ct"   # Hugging Face copy (sha256 in configs/model_sha256_hf.json)
tavily_secret = modal.Secret.from_name("tavily-api")
RUNS = f"{DATA}/runs"

# K4 (9 Nov): Kaggle model V2 vs the Hugging Face copy: model.safetensors, config.json, generation_config.json,
# processor_config.json, tokenizer.json IDENTICAL; chat_template.jinja, tokenizer_config.json, README.md DIFFER
# (HF has a newer template, "Published 2026-07-09"; it renders agent tool-call conversations differently).
# So the model folder used here = HF weights + Kaggle's 3 text files (uploaded to the weights Volume),
# checked file-by-file against Kaggle's sha256 (modal_replica/env/model_sha256_kaggle.json).
KAGGLE_OVERRIDES = f"{MODELS}/kaggle_v2_overrides"
KAGGLE_MODEL_DIR = "/tmp/model_kaggle_v2"
KAGGLE_SHA = "/opt/replica/model_sha256_kaggle.json"
code_image = code_image.add_local_file(HERE / "env" / "model_sha256_kaggle.json", KAGGLE_SHA)
for _sub in ("out", "out_vague", "out_nospec"):  # silent-change prototypes (10 Nov), separate from the frozen set
    if (ROOT / "tasks_silent" / _sub / "tasks.jsonl").exists():
        code_image = code_image.add_local_dir(ROOT / "tasks_silent" / _sub, f"{CODE}/tasks_silent/{_sub}")


def _compose_kaggle_model(full_hash: bool = False) -> str:
    """Build /tmp/model_kaggle_v2 = symlinks to the HF files + Kaggle's chat_template / tokenizer_config / README.
    Checks every file against Kaggle's sha256 (weights: size only unless full_hash, ~2 min for 23 GB)."""
    import hashlib
    import json
    import os
    from pathlib import Path

    ref = json.loads(Path(KAGGLE_SHA).read_text())["files"]
    dst = Path(KAGGLE_MODEL_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ref:
        src = Path(KAGGLE_OVERRIDES) / name if (Path(KAGGLE_OVERRIDES) / name).exists() else Path(HF_MODEL_DIR) / name
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(src, link)
    extra = sorted(p.name for p in dst.iterdir() if p.name not in ref)
    assert not extra, f"unexpected files in composed model dir: {extra}"
    for name, info in ref.items():
        p = dst / name
        assert p.stat().st_size == info["bytes"], f"{name}: size {p.stat().st_size} != Kaggle {info['bytes']}"
        if full_hash or info["bytes"] < 1e9:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            assert h.hexdigest() == info["sha256"], f"{name}: sha256 differs from Kaggle"
    print(f"model dir {dst}: {len(ref)} files match Kaggle V2 "
          f"({'all sha256' if full_hash else 'sha256 for small files, size for weights'})", flush=True)
    return str(dst)


@app.function(image=code_image, cpu=4, memory=8192, timeout=1800, volumes={MODELS: weights_vol})
def _model_check() -> str:
    return _compose_kaggle_model(full_hash=True)


@app.local_entrypoint()
def model_check():
    """Full sha256 of the composed model dir vs Kaggle (K4). CPU only, ~3-5 min."""
    print(_model_check.remote())


def _start_vllm_like_cell4(agent_dir: pathlib.Path, model_path: str):
    """The official notebook's section 4, verbatim except MODEL_PATH (Kaggle: /kaggle/input/models/google/
    gemma-4/other/<model>/2; here the Hugging Face copy). Same VllmConfig, same adapters (the sample agent's
    dummy LoRAs -> --enable-lora, as on Kaggle), tp from the GPU count."""
    import os
    import re

    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    litellm.drop_params = True
    allowed = {"gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct",
               "gemma-4-e2b-it-qat-w4a16-ct"}
    extra = {"gemma-4-12b-it-qat-w4a16-ct": ("--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
    declared = discover_declared_models(agent_dir)
    assert len(declared) == 1
    declared_model = next(iter(declared))
    assert declared_model in allowed
    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, "adapters", None) or {}
    for name in adapter_map:
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
    ranks = [r for info in adapter_map.values() if (r := getattr(info, "rank", None)) is not None and r > 0]
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    tp = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
    cfg = VllmConfig(
        model=model_path, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
        max_model_len=32768,
        dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True, enable_lora=bool(adapter_map),
        max_loras=max(1, len(adapter_map)), max_lora_rank=max(ranks) if ranks else 128,
        scheduler_reserve_full_isl=False, tensor_parallel_size=tp, startup_timeout=60 * 20,
        extra_args=extra.get(declared_model, ()),
    )
    server = VllmServer(cfg, adapter_manifest=adapters)
    print("vLLM command:", " ".join(server.build_cmd()), flush=True)
    server.start()
    print(f"vLLM server started on {server.base_url} (tp={tp}, gpus={gpu_count})", flush=True)
    models = server.create_model_registry(aliases=[declared_model], model_prefix="openai/",
                                          api_key=os.environ.get("LOCAL_API_KEY", "EMPTY"))
    return server, models, declared_model


SEARCH_PROBE = (
    "\n\n## Mandatory first step (mechanics test)\n"
    "Before anything else: call `load_skill` with skill_name `web-search`, then call `run_skill_script` with "
    "skill_name `web-search`, file_path `scripts/web_search.py` and the arguments `marshmallow 4 validates data_key`. "
    "Quote the first result URL in your final message.\n")


# GPU: 4x L4 was chosen first (= Kaggle hardware) but Modal had no L4:4 capacity for 5 h+ (9 Nov 18:00-23:15);
# user chose L40S (tp=1, computed from the GPU count by the cell-4 code). The C0 calibration on both platforms
# measures the effect; the paper reports the hardware difference. Override: env MODAL_AGENT_GPU=L4:4.
import os as _os
AGENT_GPU = _os.environ.get("MODAL_AGENT_GPU", "L40S")


@app.function(image=code_image, gpu=AGENT_GPU, cpu=8, memory=65536, timeout=24 * 3600,
              volumes={DATA: data_vol, MODELS: weights_vol}, secrets=[tavily_secret])
def _agent_runs(conds: list, group: str, task_ids: list, repeats: int, run_name: str, concurrency: int = 3,
                max_hours: float = 20.0, search_budget: int = 7200, probe: str = "",
                task_dir: str = "tasks_postcutoff/out") -> dict:
    import json
    import os
    import shutil
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    os.environ.update(CELL1_ENV)
    sys.path.insert(0, f"{CODE}/harness")
    import build_conditions
    import pc_overlay
    import run_experiment
    from swegemma.models import load_tasks

    out = Path(RUNS) / run_name / group
    out.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)

    # web-search relay (key stays in this process tree, never in the sandbox env)
    relay = None
    if any("web-search" in build_conditions.CONDITIONS[c]["skills"] for c in conds):
        relay = subprocess.Popen([sys.executable, f"{CODE}/harness/search_proxy.py", "--port", "8765",
                                  "--log", str(out / "search_log.jsonl"), "--budget", str(search_budget)])
        time.sleep(2)
        assert relay.poll() is None, "search relay failed to start"

    # official cell 2: AGENT_DIR = copy of sample_submission (31B: unchanged, adapters kept)
    agent_dir = work / "sample_submission"
    shutil.rmtree(agent_dir, ignore_errors=True)
    shutil.copytree(f"{DATA}/sample_submission", agent_dir)
    server, models, model_name = _start_vllm_like_cell4(agent_dir, _compose_kaggle_model())

    stop = threading.Event()
    def _commit_loop():
        while not stop.wait(120):
            try:
                data_vol.commit()
            except Exception as e:
                print("volume commit failed:", e, flush=True)
    threading.Thread(target=_commit_loop, daemon=True).start()

    try:
        ag = build_conditions.build_all(Path(f"{DATA}/sample_submission"), work / "agents", model=model_name,
                                        conditions=conds)
        if probe:  # forced mechanics test copy (results not used in the paper)
            pdir = work / "agents" / f"{conds[0]}_{probe}"
            shutil.rmtree(pdir, ignore_errors=True)
            shutil.copytree(ag[conds[0]], pdir)
            sm = pdir / "prompts" / "system.md"
            sm.write_text(sm.read_text() + SEARCH_PROBE)
            ag = {f"{conds[0]}{probe}": pdir}
        if group == "control":
            tasks = load_tasks(Path(DATA) / "tasks.jsonl")
            run_experiment.run_batch(ag, task_ids, tasks, repeats=repeats, models=models, data_dir=Path(DATA),
                                     out_dir=out, concurrency=concurrency, max_hours=max_hours, platform_name="modal")
        else:
            pc = Path(CODE) / task_dir  # tasks_postcutoff/out (frozen set) or tasks_silent/out (prototypes)
            # prototypes reuse the frozen set's bundle (same pinned new versions, checked 10 Nov)
            wheels = pc / "wheels" if (pc / "wheels").exists() else Path(CODE) / "tasks_postcutoff" / "out" / "wheels"
            ov = pc_overlay.build_overlay(wheels, work / "pc_overlay")
            pc_overlay.enable(ov)
            try:
                run_experiment.run_batch(ag, task_ids, load_tasks(pc / "tasks.jsonl"), repeats=repeats, models=models,
                                         data_dir=Path(DATA), out_dir=out, concurrency=concurrency, max_hours=max_hours,
                                         platform_name="modal", tasks_path=pc / "tasks.jsonl",
                                         snapshots_dir=pc / "snapshots", wheels_dir=wheels)
            finally:
                pc_overlay.disable()
    finally:
        stop.set()
        server.stop()
        if relay:
            relay.terminate()
        data_vol.commit()
    runs = out / "runs.jsonl"
    return dict(out=str(out), runs=runs.read_text() if runs.exists() else "",
                search_log=(out / "search_log.jsonl").read_text() if (out / "search_log.jsonl").exists() else "")


@app.local_entrypoint()
def agent_runs(conds: str, group: str, tasks: str, repeats: int = 1, run_name: str = "", concurrency: int = 3,
               max_hours: float = 20.0, search_budget: int = 7200, probe: str = "", task_dir: str = "tasks_postcutoff/out"):
    """e.g. --conds C1 --group postcutoff --tasks pc_marshmallow_validates --probe searchprobe   (smoke)
            --conds C1,C2,C4 --group control --tasks all --repeats 2 --run-name main_c1"""
    import json
    import time

    cl = [c.strip() for c in conds.split(",") if c.strip()]
    if tasks == "all":
        if group == "control":
            ids = json.loads((ROOT / "configs" / "control_tasks.json").read_text())["control_tasks"]
        else:
            ids = [json.loads(l)["instance_id"] for l in (ROOT / task_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        ids = [t.strip() for t in tasks.split(",") if t.strip()]
    name = run_name or time.strftime("%Y%m%d_%H%M%S")
    r = _agent_runs.remote(cl, group, ids, repeats, name, concurrency, max_hours, search_budget, probe, task_dir)
    dst = HERE / "env" / "runs" / name / group
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "runs.jsonl").write_text(r["runs"], encoding="utf-8")
    (dst / "search_log.jsonl").write_text(r["search_log"], encoding="utf-8")
    print("volume:", r["out"], "| local copy:", dst)
    for l in r["runs"].splitlines():
        x = json.loads(l)
        print({k: x.get(k) for k in ("condition", "task", "repeat", "resolved", "tool_calls", "search_calls",
                                     "docs_calls", "duration_s", "error")})


# ---------------------------------------------------------------- MAIN RUNS (10 Nov): one job per condition
# configs/main_tasks.json lists the batches (task groups); one container per condition loads the model once
# and runs every batch (resumable: run_batch skips runs already in runs.jsonl). Started from a DEPLOYED app
# with modal_replica/spawn_main.py, so the jobs do not depend on the laptop process (a pilot was lost on
# 10 Nov 02:00 when the local watcher was killed).
code_image = code_image.add_local_file(ROOT / "configs" / "main_tasks.json", f"{CODE}/configs/main_tasks.json")


@app.function(image=code_image, gpu=AGENT_GPU, cpu=6, memory=49152, timeout=24 * 3600,
              volumes={DATA: data_vol, MODELS: weights_vol}, secrets=[tavily_secret])
def _condition_runs(cond: str, run_name: str, repeats: int = 2, concurrency: int = 3,
                    search_budget: int = 1440, max_hours: float = 20.0, batches: str = "",
                    drop_offline: bool = False) -> dict:
    """batches: comma list of batch names to run (default all). drop_offline (EXPLORATORY, not
    pre-registered, added 10 Nov after the main runs): rewrite the one harness task-message line
    "Environment is offline (no network/PyPI access). ..." to keep only the no-install rule."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    os.environ.update(CELL1_ENV)
    sys.path.insert(0, f"{CODE}/harness")
    import build_conditions
    import pc_overlay
    import run_experiment
    from swegemma.models import load_tasks

    plan = json.loads(Path(f"{CODE}/configs/main_tasks.json").read_text())
    if batches:
        keep = {x.strip() for x in batches.split(",")}
        plan["batches"] = [b for b in plan["batches"] if b["name"] in keep]
        assert {b["name"] for b in plan["batches"]} == keep, f"unknown batch in {keep}"
    root = Path(RUNS) / run_name / cond
    root.mkdir(parents=True, exist_ok=True)
    if drop_offline:
        from swegemma.harness import agent_runner
        old_line = ("- Environment is offline (no network/PyPI access). All repository and test dependencies "
                    "are ALREADY pre-installed. Do NOT attempt to run pip install or download packages.")
        new_line = ("- All repository and test dependencies are ALREADY pre-installed. "
                    "Do NOT attempt to run pip install or download packages.")
        _orig = agent_runner.build_agent_prompt

        def _patched(*a, **k):
            p = _orig(*a, **k)
            assert old_line in p, "offline line not found in task prompt"
            return p.replace(old_line, new_line)
        agent_runner.build_agent_prompt = _patched
    (root / "variant.json").write_text(json.dumps(dict(batches=batches, drop_offline=drop_offline)))
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    relay = None
    if "web-search" in build_conditions.CONDITIONS[cond]["skills"]:
        relay = subprocess.Popen([sys.executable, f"{CODE}/harness/search_proxy.py", "--port", "8765",
                                  "--log", str(root / "search_log.jsonl"), "--budget", str(search_budget)])
        time.sleep(2)
        assert relay.poll() is None, "search relay failed to start"
    agent_dir = work / "sample_submission"
    shutil.rmtree(agent_dir, ignore_errors=True)
    shutil.copytree(f"{DATA}/sample_submission", agent_dir)
    server, models, model_name = _start_vllm_like_cell4(agent_dir, _compose_kaggle_model())
    stop = threading.Event()

    def _commit_loop():
        while not stop.wait(60):
            try:
                data_vol.commit()
            except Exception as e:
                print("volume commit failed:", e, flush=True)
    threading.Thread(target=_commit_loop, daemon=True).start()
    done = {}
    try:
        ag = build_conditions.build_all(Path(f"{DATA}/sample_submission"), work / "agents", model=model_name,
                                        conditions=[cond])
        for b in plan["batches"]:
            out = root / b["name"]
            print(f"=== {cond} batch {b['name']} ({len(b['tasks'])} tasks x {repeats})", flush=True)
            if b["group"] == "control":
                run_experiment.run_batch(ag, b["tasks"], load_tasks(Path(DATA) / "tasks.jsonl"), repeats=repeats,
                                         models=models, data_dir=Path(DATA), out_dir=out, concurrency=concurrency,
                                         max_hours=max_hours, platform_name="modal")
            else:
                pc = Path(CODE) / b["task_dir"]
                wheels = pc / "wheels" if (pc / "wheels").exists() else Path(CODE) / "tasks_postcutoff" / "out" / "wheels"
                ov = pc_overlay.build_overlay(wheels, work / f"ov_{b['name']}")
                pc_overlay.enable(ov)
                try:
                    run_experiment.run_batch(ag, b["tasks"], load_tasks(pc / "tasks.jsonl"), repeats=repeats,
                                             models=models, data_dir=Path(DATA), out_dir=out, concurrency=concurrency,
                                             max_hours=max_hours, platform_name="modal", tasks_path=pc / "tasks.jsonl",
                                             snapshots_dir=pc / "snapshots", wheels_dir=wheels)
                finally:
                    pc_overlay.disable()
            data_vol.commit()
            p = out / "runs.jsonl"
            done[b["name"]] = len(p.read_text().splitlines()) if p.exists() else 0
    finally:
        stop.set()
        server.stop()
        if relay:
            relay.terminate()
        data_vol.commit()
    return dict(cond=cond, runs_per_batch=done)


@app.function(image=image, cpu=2, memory=8192, timeout=900, volumes={MODELS: weights_vol})
def _template_check(templates: dict) -> dict:
    """Render Kaggle's vs Hugging Face's chat template with transformers (the version in the replica = Kaggle)
    for (a) knowledge-probe style prompts (no tools) and (b) an agent conversation with tool calls."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    probe = [{"role": "user", "content": "What changed in pandas 3.0?"}]
    probe_sys = [{"role": "system", "content": "You are a helpful assistant."}] + probe
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    agent = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix bug"},
             {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "print(1)"},
             {"role": "assistant", "content": "I will edit it."},
             {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function",
                 "function": {"name": "read_file", "arguments": {"path": "b.py"}}}]},
             {"role": "tool", "tool_call_id": "c2", "content": "x=2"}]
    out = {}
    cases = {"probe": (probe, None, {}), "probe_sys": (probe_sys, None, {}),
             "probe_think_off": (probe, None, {"enable_thinking": False}),
             "probe_think_on": (probe, None, {"enable_thinking": True}),
             "agent_tools": (agent, tools, {})}
    for name, (msgs, tl, kw) in cases.items():
        r = {}
        for which, tmpl in templates.items():
            try:
                r[which] = tok.apply_chat_template(msgs, tools=tl, chat_template=tmpl, tokenize=False,
                                                   add_generation_prompt=True, **kw)
            except Exception as e:
                r[which] = f"ERROR {type(e).__name__}: {e}"
        out[name] = r
    import transformers
    out["_versions"] = {"transformers": transformers.__version__}
    return out


@app.local_entrypoint()
def template_check(kaggle_dir: str, hf_dir: str):
    import difflib

    t = {w: (pathlib.Path(p) / "chat_template.jinja").read_text(encoding="utf-8") for w, p in (("kaggle", kaggle_dir), ("hf", hf_dir))}
    r = _template_check.remote(t)
    print("versions:", r.pop("_versions"))
    for name, v in r.items():
        same = v["kaggle"] == v["hf"]
        print(f"{name:16} {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            for l in difflib.unified_diff(v["kaggle"].splitlines(), v["hf"].splitlines(), "kaggle", "hf", n=0, lineterm=""):
                print("    ", l[:200])


@app.function(image=code_image, cpu=1, memory=2048, timeout=600)
def _runtime_paths() -> dict:
    """Which distributions does the *function process* see that `pip freeze` (a subprocess) does not,
    and where do they live? And what sys.path does a fresh subprocess get (what the sandbox inherits)?"""
    import importlib.metadata as md
    import os
    import subprocess
    import sys

    def dists(paths):
        out = {}
        for d in md.distributions(path=paths):
            out.setdefault(d.metadata["Name"].lower(), []).append(str(getattr(d, "_path", "?")))
        return out
    sub_path = subprocess.run([sys.executable, "-c", "import sys, json; print(json.dumps(sys.path))"],
                              capture_output=True, text=True).stdout
    import json
    sub_path = json.loads(sub_path)
    here, sub = dists(sys.path), dists(sub_path)
    extra = {k: v for k, v in here.items() if k not in sub}
    dup = {k: v for k, v in here.items() if len(v) > 1}
    return dict(sys_path=sys.path, subprocess_sys_path=sub_path, pythonpath=os.environ.get("PYTHONPATH"),
                n_here=sum(len(v) for v in here.values()), n_sub=sum(len(v) for v in sub.values()),
                extra_in_function_process=extra, duplicates=dup)


@app.local_entrypoint()
def runtime_paths():
    import json
    print(json.dumps(_runtime_paths.remote(), indent=1)[:6000])


@app.local_entrypoint()
def freeze_check(k1_freeze: str, gpu_image_freeze: str):
    import json

    got = _parse(_freeze.remote())
    nb = _parse(pathlib.Path(k1_freeze).read_text(encoding="utf-8"))
    base = _parse(pathlib.Path(gpu_image_freeze).read_text(encoding="utf-8"))
    wheelhouse = {k: v for k, v in nb.items() if v.startswith("@ file:///tmp/wheelhouse/")}
    expected = {**base, **wheelhouse}
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    wrong = {k: (got[k], expected[k]) for k in expected if k in got and got[k] != expected[k]}
    print(f"replica {len(got)} packages | expected {len(expected)} (v170 GPU image {len(base)} + "
          f"{len(wheelhouse)} wheelhouse overrides)")
    print(f"missing {len(missing)}: {missing}\nextra {len(extra)}: {extra}")
    print(f"different {len(wrong)}:")
    for k, (a, b) in sorted(wrong.items()):
        print(f"   {k}: replica {a} | expected {b}")
    ok = not (missing or extra or wrong)
    print("REPLICA FREEZE IDENTICAL TO EXPECTED:", ok)
    (HERE / "env").mkdir(exist_ok=True)
    (HERE / "env" / "replica_freeze_check.json").write_text(
        json.dumps(dict(ok=ok, missing=missing, extra=extra, different=wrong), indent=2), encoding="utf-8")
