#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "${root}"
evidence="results/gpu/current-head-8cb3759-rtx5090"
seals="${evidence}/seals"
commit="8cb3759b29ea8c7d843322a940a7ebcabfd9ff21"
# archive_run.py must not log into the tree it is indexing (H100 reseal lesson).
logdir="${RESEAL_LOGDIR:-/home/angus/reseal8cb/logs}"

test "$(git rev-parse HEAD)" = "${commit}"
test -z "$(git status --porcelain=v1)"

for gate in g2 g3; do
    context="${evidence}/${gate}/context"
    mkdir -p "${context}"
    cp "${evidence}/preflight/source.txt" "${context}/source.txt"
    cp "${evidence}/preflight/system.txt" "${context}/system.txt"
    cp "${evidence}/preflight/gpu.txt" "${context}/gpu.txt"
    cp "${evidence}/preflight/toolchain.txt" "${context}/toolchain.txt"
    cp "${evidence}/preflight/dependencies.txt" "${context}/dependencies.txt"
    cp "${evidence}/preflight/qoco-build-manifest.txt" "${context}/qoco-build-manifest.txt"
    cp "${evidence}/preflight/source-policy-schema.sha256" \
        "${context}/source-policy-schema.sha256"
    cp "${seals}/schema-scope-validation.log" "${context}/schema-scope-validation.log"
    cp "${evidence}/current-head-summary.json" "${context}/current-head-summary.json"
    .venv-current-head/bin/python scripts/gpu/archive_run.py \
        "${evidence}/${gate}" \
        --repository . \
        --require-clean-repository \
        --archive "${seals}/${gate}-${commit:0:12}.tar.gz" \
        >"${seals}/${gate}-archive.log" 2>&1
    sha256sum "${seals}/${gate}-${commit:0:12}.tar.gz" \
        >"${seals}/${gate}-${commit:0:12}.tar.gz.sha256"
done

.venv-current-head/bin/python - "${evidence}" "${commit}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
commit = sys.argv[2]
archives = []
for gate in ("g2", "g3"):
    path = root / "seals" / f"{gate}-{commit[:12]}.tar.gz"
    archives.append(
        {
            "gate": gate.upper(),
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evidence_index": f"{gate}/evidence-index.json",
            "local_only": True,
            "immutable_uri": None,
        }
    )
payload = {
    "schema_version": "current-head-archives-1.0.0",
    "source_commit": commit,
    "archives": archives,
    "local_only": True,
    "immutable_uri": None,
}
(root / "seals/archives.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
summary_path = root / "current-head-summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["archives"] = archives
summary_path.write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

rm -rf "${seals}/__pycache__"
.venv-current-head/bin/python scripts/gpu/archive_run.py \
    "${evidence}" \
    --repository . \
    --require-clean-repository \
    >"${logdir}/root-index.log" 2>&1
sha256sum "${evidence}/evidence-index.json" \
    >"${evidence}/evidence-index.json.sha256"
test -z "$(git status --porcelain=v1)"
