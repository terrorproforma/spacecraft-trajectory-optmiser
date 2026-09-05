#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-9e75b47-h100/seals"
export PYTHONPATH="${root}/src"

.venv/bin/python -m pytest -q \
    tests/test_benchmark_manifests.py \
    tests/test_campaign_scope.py \
    tests/test_g4_execution_contract.py \
    tests/test_g4_executor_contract.py \
    tests/test_g4_qualification.py \
    >"${out}/schema-scope-validation.log" 2>&1

.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("results/gpu/current-head-9e75b47-h100")
paths = [
    root / "current-head-summary.json",
    root / "g0/summary.json",
    root / "g1/summary.json",
    root / "g2/summary.json",
    root / "g3/summary.json",
]
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "9e75b470fd5378d9b20f6f13892ac909c43757cd"
    assert payload["local_only"] is True
    assert payload["immutable_uri"] is None
PY

test "$(git rev-parse HEAD)" = "9e75b470fd5378d9b20f6f13892ac909c43757cd"
test -z "$(git status --porcelain=v1)"
