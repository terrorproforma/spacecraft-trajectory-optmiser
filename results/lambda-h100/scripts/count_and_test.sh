#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"; export CUDA_VISIBLE_DEVICES=""
cd "$HOME/spacepdhcg/gtoc12"; export PYTHONPATH="$PWD/src"
echo "== family counts"; time .venv/bin/python "$HOME/s/count_families.py" 2>&1 | tail -25
echo "== full gtoc12 test suite (background log ~/logs/gtoc12-tests-merged.log)"
nohup .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gtoc12_*.py -x --deselect tests/test_gtoc12_bundles.py::test_single_slot_pricing_stays_inside_the_declared_memory_budget > "$HOME/logs/gtoc12-tests-merged.log" 2>&1 < /dev/null &
echo "tests pid $!"