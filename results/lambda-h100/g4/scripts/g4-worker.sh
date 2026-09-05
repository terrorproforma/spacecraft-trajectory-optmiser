#!/usr/bin/env bash
# Durable H100 claim-core worker under amendment single-gpu-v1.2 (also the restart command).
# Run-and-flag contamination: never waits for GPU idle; resumes an interrupted group first.
# Runs from ~/spacepdhcg/v1 pinned at the campaign's source commit 1dbcae0 (branch g4/h100-1dbcae0),
# pinned to CPU cores 0-3; the host-context contamination channel is the native pmon stand-in.
set -euo pipefail
source /home/ubuntu/s/g4env-h100.sh
test "$(git rev-parse HEAD)" = "$head_sha"
test -z "$(git status --porcelain=v1)"
# IPM preflight: the QOCO library the capability probed must be the one this worker dlopens.
test -n "${SPACEPDHCG_QOCO_LIBRARY:-}" -a -f "${SPACEPDHCG_QOCO_LIBRARY}"
pinned=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ipm_library"]["sha256"])' "$capability")
test "$(sha256sum "${SPACEPDHCG_QOCO_LIBRARY}" | cut -d' ' -f1)" = "${pinned}"
test -x /home/ubuntu/s/host-pmon-linux.sh
cd "$root"
exec taskset -c 0-3 "${py}" scripts/gpu/run_g4_campaign.py run --claim-core \
  --amendment "${root}/benchmarks/g4_claim_core_amendment_v1_2.json" \
  --repository "$root" \
  --campaign "$campaign" \
  --executable "${exe}" \
  --capabilities "$capability" \
  --host-nvidia-smi /home/ubuntu/s/host-pmon-linux.sh
