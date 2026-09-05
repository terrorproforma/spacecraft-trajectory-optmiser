#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/gtoc12"
export PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/src" CUDA_VISIBLE_DEVICES=""
git stash push -q -- src/spacepdhcg/gtoc12/cooperative.py; echo "stash rc=$?"
python - <<'PY'
import sys, importlib.util, inspect
from pytest import MonkeyPatch
spec = importlib.util.spec_from_file_location("t", "tests/test_gtoc12_cooperative.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
fn = mod.test_master_widens_the_recursion_limit_for_the_column_count
print(inspect.getsource(fn)[-400:])
mp = MonkeyPatch()
try:
    fn(mp); print("RESULT: passed (unexpected without fix)")
except AssertionError as e:
    print("RESULT: AssertionError (expected without fix)", str(e)[:200])
finally:
    mp.undo()
PY
find tests -name '__pycache__' -type d | head -2; ls -la tests/__pycache__ 2>/dev/null | grep cooperative
git stash pop -q; echo "pop rc=$?"
