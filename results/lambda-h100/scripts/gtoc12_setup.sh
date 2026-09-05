#!/bin/bash
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export UV_LINK_MODE=copy
root="$HOME/spacepdhcg/gtoc12"
cd "$root"
log "== gtoc12 clone: $(git rev-parse --short HEAD) $(git branch --show-current) dirty=$(git status --porcelain=v1 | wc -l)"
log "== venv"
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q -p .venv/bin/python --upgrade pip "cmake==4.4.3" "ninja==1.13.2" "ruff==0.16.5" "pytest==9.1.1" jsonschema 2>&1 | tail -2
PATH="$root/.venv/bin:$PATH" .venv/bin/python -m pip install -q -e '.[dev]' 2>&1 | tail -3
.venv/bin/python -c "import spacepdhcg; from spacepdhcg.native import native_available; print('spacepdhcg', spacepdhcg.__version__, 'native', native_available())"
log "== fetch pinned data"
PYTHONPATH=src .venv/bin/python scripts/gtoc12/fetch_gtoc12_data.py 2>&1 | tail -15
ls -la benchmarks/gtoc12/data | head -20
log "== official verifier (extract pinned Linux binary) + example solution"
PYTHONPATH=src .venv/bin/python - <<'PY'
from spacepdhcg.gtoc12.data import official_verifier_binary, official_example_solution
from spacepdhcg.gtoc12.official import official_verifier_available, run_official_verifier
b = official_verifier_binary(extract=True)
print("binary:", b, b.stat().st_size, "bytes")
print("available:", official_verifier_available())
r = run_official_verifier(official_example_solution())
print("example:", r.summary())
PY
log "== archived runs present (fleet-master sources)"
ls results/gtoc12/runs
for r in results/gtoc12/runs/*; do printf '%s route_summaries=%s\n' "$r" "$(find $r -name route_summary.json | wc -l)"; done
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("results/gtoc12/runs/cluster_fleet_v6/run_report.json")
if p.exists():
    d = json.loads(p.read_text())
    print("v6 report keys:", list(d)[:30])
    for k in ("memory_total_pss_peak_mb", "workers", "families_priced", "budget_marks", "elapsed_seconds", "wall_seconds"):
        if k in d: print(k, "=", str(d[k])[:300])
    args = d.get("arguments") or d.get("args") or d.get("configuration")
    print("args:", json.dumps(args)[:1200] if args else None)
PY
log "== quick tests (gtoc12 subset)"
PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gtoc12_pipeline.py tests/test_gtoc12_cooperative.py -x -q 2>&1 | tail -3
log "== done"
