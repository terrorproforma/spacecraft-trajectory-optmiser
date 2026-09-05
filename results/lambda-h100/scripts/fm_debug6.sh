#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/gtoc12"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=""
git stash push -q -- src/spacepdhcg/gtoc12/cooperative.py; echo "stash rc=$?"
cp tests/test_gtoc12_cooperative.py /tmp/test_backup.py
python - <<'PY'
from pathlib import Path
p = Path("tests/test_gtoc12_cooperative.py"); t = p.read_text()
old = "    sys.setrecursionlimit(lowered)\n    try:\n        result = solve_fleet_master(columns, lp_bound=False)\n"
new = """    sys.setrecursionlimit(lowered)
    _orig_set = sys.setrecursionlimit
    def _spy(n):
        import traceback
        print('DBG setrecursionlimit', n); traceback.print_stack(limit=6)
        _orig_set(n)
    monkeypatch.setattr(sys, 'setrecursionlimit', _spy)
    try:
        result = solve_fleet_master(columns, lp_bound=False)
"""
assert old in t; p.write_text(t.replace(old, new))
PY
python -m pytest -p no:cacheprovider tests/test_gtoc12_cooperative.py -k widens_the_recursion_limit -s -q 2>&1 | grep -v '^$' | head -40
cp /tmp/test_backup.py tests/test_gtoc12_cooperative.py
git stash pop -q; echo "pop rc=$?"; git status --short
