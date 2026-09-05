#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/gtoc12"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=""
git stash push -q -- src/spacepdhcg/gtoc12/cooperative.py; echo "stash rc=$?"
echo "required_depth present after stash: $(grep -c required_depth src/spacepdhcg/gtoc12/cooperative.py)"
sed -n "$(grep -n 'def test_master_widens' tests/test_gtoc12_cooperative.py | cut -d: -f1),+28p" tests/test_gtoc12_cooperative.py
python -m pytest -p no:cacheprovider tests/test_gtoc12_cooperative.py -k widens_the_recursion_limit -vv -s --tb=long 2>&1 | tail -25
git stash pop -q; echo "pop rc=$?"
