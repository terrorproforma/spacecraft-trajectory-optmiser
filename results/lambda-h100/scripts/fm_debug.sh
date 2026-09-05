#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/gtoc12"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=""
git stash push -q -- src/spacepdhcg/gtoc12/cooperative.py; echo "stash rc=$?"
echo "required_depth present after stash: $(grep -c required_depth src/spacepdhcg/gtoc12/cooperative.py)"
python - <<'PY'
import sys
sys.path.insert(0, "tests")
import spacepdhcg.gtoc12.cooperative as cooperative
print("module:", cooperative.__file__)
from test_gtoc12_cooperative import _column
from spacepdhcg.gtoc12.cooperative import solve_fleet_master
columns = [_column(i, {100 + i: 100.0}, {100 + i: 3000.0}, 10.0 + i) for i in range(60)]
observed = []
orig = cooperative.ship_count
def rec(selected):
    observed.append(sys.getrecursionlimit()); return orig(selected)
cooperative.ship_count = rec
prev = sys.getrecursionlimit(); print("prev limit", prev)
sys.setrecursionlimit(300)
r = solve_fleet_master(columns, lp_bound=False)
print("observed n", len(observed), "max", max(observed) if observed else None, "limit now", sys.getrecursionlimit(), "selected", len(r.selected))
PY
git stash pop -q; echo "pop rc=$?"
echo "required_depth present after pop: $(grep -c required_depth src/spacepdhcg/gtoc12/cooperative.py)"
