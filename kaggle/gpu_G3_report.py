# =============================== G3: report on the dry run (no GPU work) ===============================
import json
from pathlib import Path

rows = []
for grp in ('control', 'postcutoff'):
    p = WORKING_DIR / 'dry' / grp / 'runs.jsonl'
    rows += [dict(json.loads(l), group=grp) for l in p.read_text().splitlines()] if p.exists() else []
print(f'{len(rows)} runs')
for r in rows:
    print(f"{r['group']:10} {r['condition']} {r['task']:26} resolved={str(r['resolved']):5} exit={r.get('test_exit_code')} "
          f"tools={r.get('tool_calls')} steps={r.get('steps')} tok_in={r.get('prompt_tokens')} cached={r.get('cached_tokens')} "
          f"compactions={r.get('compactions')} skills={r.get('skill_calls')} {r.get('duration_s')}s err={str(r.get('error'))[:80]}")
    print(f"{'':12}tool_hist={r.get('tool_hist')} patch_files={r.get('patch_files')}")

# Real TaskResult field names (from one saved run)
one = next((p for p in (WORKING_DIR / 'dry').rglob('rep1/*.json') if not p.name.endswith('.trace.json')), None)
if one:
    print('\nresult_fields keys:', sorted(json.loads(one.read_text()).get('result_fields', {})))

# Prompt checks + skill calls from the traces
for tr in sorted((WORKING_DIR / 'dry').rglob('*.trace.json')):
    t = json.loads(tr.read_text())
    steps = t.get('steps', [])
    sysmsg = next((s.get('message') or '' for s in steps if (s.get('extra') or {}).get('event_type') == 'system_instruction'), '')
    allmsg = json.dumps(steps[:3])
    print(f"\n== {tr.parent.parent.name}/{tr.name}: graph-tool section in prompt: {'Code Intelligence Tools' in allmsg} | "
          f"scratch rule: {'Scratch files' in sysmsg} | docs note: {'library-docs' in sysmsg}")
    for s in steps:
        for tc in s.get('tool_calls') or []:
            if tc.get('function_name') in ('run_skill_script', 'load_skill_resource'):
                obs = json.dumps(s.get('observation'))[:400]
                print(f"   SKILL CALL {tc['function_name']} args={json.dumps(tc.get('arguments'))[:300]}\n     -> {obs}")
