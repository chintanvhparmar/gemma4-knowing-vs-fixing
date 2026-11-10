# =============================== K4: are Kaggle's model files identical to the Hugging Face copy on Modal? ===============================
# CPU notebook (no GPU quota). Attach the model input first: Add Input -> Models -> "google/gemma-4",
# variation gemma-4-31b-it-qat-w4a16-ct, VERSION 2 (the official cell 4 loads .../other/<model>/2).
# Compares sha256 of every file with configs/model_sha256_hf.json (recorded when Modal downloaded
# google/gemma-4-31B-it-qat-w4a16-ct from Hugging Face). ~21.7 GB to read: a few minutes.
import glob, hashlib, json, os, time
from pathlib import Path

MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
assert MODEL_PATH.exists(), f'{MODEL_PATH} missing: attach the model input (version 2)'
ref = json.loads(open(glob.glob('/kaggle/input/**/gemma-paper-code/configs/model_sha256_hf.json', recursive=True)[0]).read())
hf = ref['files']

t0, kag = time.time(), {}
for p in sorted(MODEL_PATH.rglob('*')):
    if p.is_file():
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        kag[str(p.relative_to(MODEL_PATH))] = dict(bytes=p.stat().st_size, sha256=h.hexdigest())
        print(f'{kag[str(p.relative_to(MODEL_PATH))]["bytes"] / 1e9:7.2f} GB  {p.name}', flush=True)

same = [k for k in kag if k in hf and kag[k]['sha256'] == hf[k]['sha256']]
diff = [k for k in kag if k in hf and kag[k]['sha256'] != hf[k]['sha256']]
only_kaggle = sorted(set(kag) - set(hf))
only_hf = sorted(set(hf) - set(kag))
print(f'\n{time.time() - t0:.0f}s | Kaggle files {len(kag)} | HF files {len(hf)} | identical {len(same)} | '
      f'different {diff} | only on Kaggle {only_kaggle} | only on HF {only_hf}')
weights = [k for k in kag if k.endswith('.safetensors')]
print('ALL WEIGHT FILES IDENTICAL:', bool(weights) and all(k in same for k in weights), f'({len(weights)} .safetensors)')
out = WORKING_DIR / 'env_export' / 'model_sha256_kaggle.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(dict(path=str(MODEL_PATH), files=kag), indent=1))
print('written', out)
