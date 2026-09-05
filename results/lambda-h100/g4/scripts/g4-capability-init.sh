#!/usr/bin/env bash
# Stage 3 of the H100 G4 resume: regenerate the executor capability from 1dbcae0 on this host (real
# --g4-session IPM probe) and initialise a fresh claim-core checkpoint under amendment single-gpu-v1.2.
# No RTX 5090 rows are imported; the 5090 IPM diagnostic stratum is cited by metadata only.
set -euo pipefail
source /home/ubuntu/s/g4env-h100.sh
mkdir -p /home/ubuntu/g4 "$g4logs"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
test "$(git rev-parse HEAD)" = "$head_sha"
test -z "$(git status --porcelain=v1)"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo "GPU busy, refuse"; exit 3; }
test -f "$exe" && test -f "$qoco"
"$exe" --g4-capabilities | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d["compiled_source_commit"]=="'"$head_sha"'", d["compiled_source_commit"]; print("compiled_source_commit ok")'

# --- QOCO pin record ---------------------------------------------------------------------------
qsrc="$root/_upstream/qoco-g4"
{
  echo "recorded_utc=$(ts)"
  echo "host=$SPACEPDHCG_HARDWARE_ID sm_90 driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader) cuda=12.8"
  echo "qoco_library=$qoco"
  echo "qoco_library_sha256=$(sha256sum "$qoco" | cut -d' ' -f1)"
  echo "qoco_source=$qsrc"
  echo "qoco_commit=$(git -C "$qsrc" rev-parse HEAD)"
  echo "qoco_tree=$(git -C "$qsrc" rev-parse 'HEAD^{tree}')"
  echo "qoco_patch=scripts/gpu/qoco_absolute_kkt_stopping.patch sha256=$(sha256sum "$root/scripts/gpu/qoco_absolute_kkt_stopping.patch" | cut -d' ' -f1)"
  echo "qoco_patch_applied=$(git -C "$qsrc" apply --reverse --check "$root/scripts/gpu/qoco_absolute_kkt_stopping.patch" 2>/dev/null && echo yes || echo no)"
  echo "qoco_working_tree_status=$(git -C "$qsrc" status --porcelain=v1 | wc -l) changed paths (the patch)"
  echo "cudss=$("$py" -c 'from importlib.metadata import version; print(version("nvidia-cudss-cu12"))')"
  echo "same_library_as_reseal_g1_g3=$(grep -c "$(sha256sum "$qoco" | cut -d' ' -f1)" "$root/results/gpu/current-head-9e75b47-h100/g3/manifest.txt" 2>/dev/null || echo 0)"
} | tee /home/ubuntu/g4/qoco-pin-${head7}-h100.txt

# --- capability (real --g4-session probe: IPM probe must create QOCO workspaces) ------------------
rm -f "$capability"
echo "[$(ts)] generating capability"
t0=$SECONDS
"$py" scripts/gpu/generate_g4_executor_capability.py --repository "$root" --executable "$exe" --output "$capability" 2>&1 | tee "$g4logs/capability.log"
echo "capability generation wall: $((SECONDS-t0)) s" | tee -a "$g4logs/capability.log"
"$py" scripts/gpu/generate_g4_executor_capability.py --repository "$root" --executable "$exe" --output "$capability" --check
python3 - "$capability" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["session_probe"]
print("capability_sha256", d["capability_sha256"])
print("source_commit", d["source_commit"], "compiled", d["compiled_source_commit"])
print("executable_sha256", d["executable_sha256"])
print("ipm_library", d["ipm_library"])
print("probe kind", p["kind"], "attempts", p["attempt_count"], "same_process/context/workspace", p["same_process"], p["same_context"], p["same_workspace"])
ipm = p["pure_gpu_ipm_probe"]
print("ipm workspace creations", ipm["qoco_workspace_creations"], "status codes", ipm["qoco_status_codes"], "dispositions", sorted(set(ipm["dispositions"])))
assert all(c >= 1 for c in ipm["qoco_workspace_creations"]), "IPM probe did not create QOCO workspaces"
assert d["policy_amendments_supported"] == ["single-gpu-v1.1", "single-gpu-v1.2"]
print("hardware_id_env", "lambda-h100-80gb-hbm3")
PY

# --- fresh checkpoint / ledger ------------------------------------------------------------------
test ! -e "$campaign" || { echo "campaign dir exists: $campaign (refuse to overwrite)"; exit 4; }
mkdir -p "$campaign"
stratum_src=/home/ubuntu/bundles/rtx5090-strata/g4-claim-core-ccd5596
echo "[$(ts)] init checkpoint"
"$py" scripts/gpu/run_g4_campaign.py init --claim-core \
  --amendment "$root/benchmarks/g4_claim_core_amendment_v1_2.json" \
  --repository "$root" --campaign "$campaign" \
  --cite-diagnostic-stratum "ipm_no_equilibration_v1_1=$stratum_src" | tee "$g4logs/init.log"
# hardware / environment note for the ledger directory (the tooling records nvidia-smi per group).
{
  echo "hardware_id=$SPACEPDHCG_HARDWARE_ID"
  echo "gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader)"
  echo "cuda=12.8 ($(/usr/local/cuda-12.8/bin/nvcc --version | tail -1))"
  echo "cuda_architecture=90"
  echo "source_commit=$head_sha branch=$(git branch --show-current)"
  echo "executable=$exe sha256=$(sha256sum "$exe" | cut -d' ' -f1)"
  echo "capability=$capability sha256=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["capability_sha256"])' "$capability")"
  echo "acceptance_criteria=single-gpu-v1.2 (benchmarks/g4_claim_core_amendment_v1_2.json $(cut -d' ' -f1 "$root/benchmarks/g4_claim_core_amendment_v1_2.sha256"))"
  echo "rtx5090_rows_imported=none (stratum ipm_no_equilibration_v1_1 cited from $stratum_src by metadata only; the 4db5047 RTX 5090 ledger is a separate hardware stratum)"
  echo "qoco_pin=/home/ubuntu/g4/qoco-pin-${head7}-h100.txt"
  echo "initialised_utc=$(ts)"
  uname -a
} | tee "$campaign/hardware.txt"
echo "[$(ts)] CAPABILITY+INIT COMPLETE"
