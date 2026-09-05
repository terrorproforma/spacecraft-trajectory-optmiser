#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/gtoc12"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=""
git stash push -q -- src/spacepdhcg/gtoc12/cooperative.py; echo "stash rc=$?"
cp "$HOME/s/dbg_test.py" /tmp/dbg_test.py
python -m pytest -p no:cacheprovider /tmp/dbg_test.py -s -q --rootdir . 2>&1 | tail -12
echo "== and with the repo conftest (copy into tests/)"
cp "$HOME/s/dbg_test.py" tests/test_zz_dbg.py
python -m pytest -p no:cacheprovider tests/test_zz_dbg.py -s -q 2>&1 | tail -12
rm -f tests/test_zz_dbg.py
git stash pop -q; echo "pop rc=$?"; git status --short
