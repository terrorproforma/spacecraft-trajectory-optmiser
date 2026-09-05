#!/usr/bin/env bash
# Completion procedure for the H100 H5/H6 claim core: validate + aggregate + decide, then seal.
# Usage: g4-finish.sh [--preview]
#   --preview : allowed on a partial ledger; writes PREVIEW-* files only, never seals.
# Exit 2 from the decision step means the amendment is invalid (a 600 s twin qualified where the
# 120 s core was censored) and no H5/H6 decision was issued.
set -euo pipefail
source /home/ubuntu/s/g4env-h100.sh
amendment="${root}/benchmarks/g4_claim_core_amendment_v1_2.json"
seal_root="${root}/results/gpu/g4/claim-core-${head7}-h100"
cd "$root"
test "$(git rev-parse HEAD)" = "$head_sha"
test -z "$(git status --porcelain=v1)"
if [ "${1:-}" = "--preview" ]; then
  "$py" scripts/gpu/decide_g4_claim_core.py --repository "$root" --campaign "${campaign}" \
    --output "${campaign}/decision" --capabilities "${capability}" --amendment "${amendment}" --allow-incomplete
  exit 0
fi
if pgrep -f "run_g4_campaign.py run --claim-core" >/dev/null; then
  echo "worker still running; refusing to seal" >&2; exit 3
fi
"$py" scripts/gpu/decide_g4_claim_core.py --repository "$root" --campaign "${campaign}" \
  --output "${campaign}/decision" --capabilities "${capability}" --amendment "${amendment}"
mkdir -p "${seal_root}"
"$py" scripts/gpu/archive_run.py "${campaign}" --repository "$root" \
  --archive "${seal_root}/g4-claim-core-${head7}-h100.tar.gz" --require-clean-repository | tee "${seal_root}/seal.txt"
sha256sum "${seal_root}/g4-claim-core-${head7}-h100.tar.gz" > "${seal_root}/g4-claim-core-${head7}-h100.tar.gz.sha256"
cp "${campaign}/evidence-index.json" "${seal_root}/evidence-index.json"
sha256sum "${seal_root}/evidence-index.json" > "${seal_root}/evidence-index.json.sha256"
mkdir -p "${seal_root}/decision"
cp "${campaign}/decision/decision.json" "${campaign}/decision/h5_coordinates.jsonl" \
   "${campaign}/decision/h6_coordinates.jsonl" "${seal_root}/decision/"
cp -r "${campaign}/decision/publication" "${seal_root}/decision/"
cp "${campaign}/inputs.sha256" "${campaign}/environment.txt" "${campaign}/hardware.txt" "${seal_root}/" 2>/dev/null || true
echo "sealed: ${seal_root} (local-only; no immutable URI)"
