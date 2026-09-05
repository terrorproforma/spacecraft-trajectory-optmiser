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
old = "    assert observed and max(observed) >= 2 * len(columns) + 200\n"
new = "    print('DBG previous', previous, 'lowered', lowered, 'n', len(observed), 'max', max(observed) if observed else None, 'limit', sys.getrecursionlimit(), 'cols', len(columns))\n" + old
assert old in t; p.write_text(t.replace(old, new))
PY
python -m pytest -p no:cacheprovider tests/test_gtoc12_cooperative.py -k widens_the_recursion_limit -s -q 2>&1 | grep -E 'DBG|passed|failed'
cp /tmp/test_backup.py tests/test_gtoc12_cooperative.py
git stash pop -q; echo "pop rc=$?"; git status --short
