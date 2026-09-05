#!/bin/bash
set -euo pipefail
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/gtoc12"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=""
test -z "$(pgrep -f 'gtoc12 fleet-master|gtoc12 cluster-fleet')" || { echo "gtoc12 job still running: refuse"; exit 3; }
git status --short
git diff --quiet -- src/spacepdhcg/gtoc12/cooperative.py && python3 "$HOME/s/patch_fleet_master.py"
grep -q '_stack_depth' tests/test_gtoc12_cooperative.py && python3 "$HOME/s/patch_fleet_master2.py"
grep -q 'peak = max(observed)' tests/test_gtoc12_cooperative.py || python3 "$HOME/s/patch_fleet_master3.py"
rm -rf tests/__pycache__
sed -n 1,30p tests/test_gtoc12_cooperative.py | grep -n '^import\|^from'
git diff --stat -- src tests
ruff check --fix --select I src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py
ruff check src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py
ruff format --check src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py || ruff format src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py
ruff check src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py
# Prove the new test fails without the fix (stash only the source change), then run it with the fix.
git stash push -q -- src/spacepdhcg/gtoc12/cooperative.py
if python -m pytest -q -p no:cacheprovider tests/test_gtoc12_cooperative.py -k widens_the_recursion_limit > /tmp/depth_without_fix.log 2>&1; then
  git stash pop -q; echo "regression test passes WITHOUT the fix - not a valid test"; exit 5
else
  tail -3 /tmp/depth_without_fix.log
  git stash pop -q
fi
python -m pytest -q -p no:cacheprovider tests/test_gtoc12_cooperative.py -k widens_the_recursion_limit
python -m pytest -q -p no:cacheprovider tests/test_gtoc12_cooperative.py -k 'not requires_data and not catalogue' -x 2>&1 | tail -2
export GIT_AUTHOR_NAME=SpacePDHCG-Integration GIT_AUTHOR_EMAIL=integration@spacepdhcg.local
export GIT_COMMITTER_NAME=SpacePDHCG-Integration GIT_COMMITTER_EMAIL=integration@spacepdhcg.local
git add src/spacepdhcg/gtoc12/cooperative.py tests/test_gtoc12_cooperative.py
git commit -q -m "fix(gtoc12): fleet-master search must not exceed the interpreter recursion limit

solve_fleet_master's depth-first search recurses once per usable column (the
skip branch walks index to len(usable)). The archive-wide fleet master on the
H100 host (12 archived sources + cluster_fleet_h100_v1, 805 recertified groups)
offered more columns than CPython's default 1000-frame limit and died with
RecursionError after the 32-minute recertification stage
(results/gtoc12/runs/fleet_master_h100_v1, logs/gtoc12-fleet_master_h100_v1.log).

Widen the recursion limit to 2 * columns + 200 for the search and restore it
afterwards; the regression test lowers the limit to just above the current
depth and packs 60 compatible columns."
git log --oneline -2
git status --short
