#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
out="results/gpu/current-head-8cb3759-rtx5090/seals"
export PYTHONPATH="${root}/src"

.venv-current-head/bin/python -m pytest -q -p no:cacheprovider \
    tests/test_benchmark_manifests.py \
    tests/test_campaign_scope.py \
    tests/test_g4_execution_contract.py \
    tests/test_g4_executor_contract.py \
    tests/test_g4_qualification.py \
    >"${out}/schema-scope-validation.log" 2>&1

.venv-current-head/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("results/gpu/current-head-8cb3759-rtx5090")
paths = [
    root / "current-head-summary.json",
    root / "g2/summary.json",
    root / "g3/summary.json",
]
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "8cb3759b29ea8c7d843322a940a7ebcabfd9ff21"
    assert payload["source_tree"] == "6d27f2552d882b4418d16e4342e6854a436a952d"
    assert payload["hardware_id"] == "local-rtx-5090"
    assert payload["cuda_architecture"] == 120
    assert payload["local_only"] is True
    assert payload["immutable_uri"] is None
summary = json.loads((root / "current-head-summary.json").read_text(encoding="utf-8"))
assert summary["gates"] == {"G2": "PASS", "G3": "PASS"}
for gate in ("g2", "g3"):
    status = dict(
        line.split("=", 1) for line in (root / gate / "status.txt").read_text().splitlines() if "=" in line
    )
    assert status["status"] == "PASS", (gate, status)
PY

test "$(git rev-parse HEAD)" = "8cb3759b29ea8c7d843322a940a7ebcabfd9ff21"
test -z "$(git status --porcelain=v1)"
