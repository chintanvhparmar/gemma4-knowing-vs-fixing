# =============================== G1b: prompt mentions of graph tools + how skills are wired (read-only) ===============================
import re
from pathlib import Path
import swegemma, google.adk as adk

def ctx(path, pattern, n=2, maxhits=8):
    L = Path(path).read_text(errors='ignore').splitlines()
    hits = [i for i, l in enumerate(L) if re.search(pattern, l)]
    print(f'\n##### {path}  ({len(hits)} hits)')
    for i in hits[:maxhits]:
        for j in range(max(0, i - n), min(len(L), i + n + 1)):
            print(f'{j+1:4}: {L[j][:180]}')
        print('   ...')

# 1. Exact prompt lines that name graph tools (official sample agent)
ctx(DATA_DIR / 'sample_submission' / 'prompts' / 'system.md', r'search_similar_code|get_code_neighbors|get_code_subgraph|graph', 2)
ctx(DATA_DIR / 'sample_submission' / 'prompts' / 'analyzer.md', r'search_similar_code|get_code_neighbors|get_code_subgraph|graph', 1)

# 2. How swegemma builds the skill toolset from the tools list / skills dir
ar = Path(swegemma.__file__).parent / 'harness' / 'agent_runner.py'
ctx(ar, r'_SKILL_TOOL_NAMES|SkillToolset|skills_dir|skills/', 4, 10)

# 3. ADK skill toolset: the instructions given to the model, and how run_skill_script executes
st = Path(adk.__file__).parent / 'tools' / 'skill_toolset.py'
L = st.read_text(errors='ignore').splitlines()
print(f'\n##### {st} lines 60-95 (model instructions)')
for j in range(59, min(95, len(L))):
    print(f'{j+1:4}: {L[j][:180]}')
ctx(st, r'subprocess|sys\.executable|exec\(|code_executor|sandbox|python', 2, 12)
