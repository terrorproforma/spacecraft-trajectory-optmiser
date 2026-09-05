#!/bin/bash
# Supplementary discriminators for the v2 deferred sweep failures (run after v2_deferred.sh).
# 1. memcheck on pd3 using the CLI-normalised request (radians) instead of the raw degrees example.
# 2. Magnitude of the pd6_fft device/CPU parity (gdb on the debug test; the test aborts before printing).
# 3. CPU-only planner baseline on this host (no GPU) to separate sm_90 effects from candidate logic.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export V2="$HOME/spacepdhcg/v2"
cd "$V2"
export PATH="$V2/.venv/bin:/usr/local/cuda-12.8/bin:$HOME/.local/node/bin:$PATH"
export PYTHONPATH="$V2/src"
export SPACEPDHCG_NATIVE_LIBRARY="$V2/build-v2-relwithdebinfo/libspacepdhcg.so"
export SPACEPDHCG_QOCO_LIBRARY="$HOME/spacepdhcg/v1/build-current-head-qoco/libqoco.so"
export LD_LIBRARY_PATH="$HOME/spacepdhcg/v1/build-current-head-qoco-cudss-lib:$V2/.venv/lib/python3.12/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out="$V2/results/gpu/h100-deferred-3373988/supplement"
mkdir -p "$out"
test -z "$(pgrep -f 'v2_deferred.sh')" || { log "sweep still running: refuse"; exit 3; }
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { log "GPU busy: refuse"; exit 3; }
: > "$out/items.tsv"
item() {
  local id=$1 expected=$2; shift 2
  { printf '%s\t' "$id"; printf '%q ' "$@"; printf '\n'; } >> "$out/commands.txt"
  log "STEP $id"; local t0=$SECONDS
  "$@" > "$out/$id.log" 2>&1; local rc=$?
  local verdict=PASS; [ "$rc" -eq "$expected" ] || verdict=FAIL
  printf '%s\t%s\texit=%s\texpected=%s\t%ss\n' "$id" "$verdict" "$rc" "$expected" "$((SECONDS-t0))" >> "$out/items.tsv"
  log "$verdict  $id exit=$rc (expected $expected) $((SECONDS-t0))s"
}
export CUDA_VISIBLE_DEVICES=0
# 1. pd3 memcheck with the normalised (radians) request the CLI actually sends to the executable.
req=build-v2-gpu-deferred/plan-powered_descent_3dof/native-request.json
item pd3-memcheck-normalised-request 0 compute-sanitizer --tool memcheck --leak-check full build-v2-cuda-release/cuda-tools/spacepdhcg_plan "$req" --output "$out/memcheck-pd3-normalised.json" --quiet
item pd3-memcheck-normalised-status 0 python -c "import json; d=json.load(open('$out/memcheck-pd3-normalised.json')); print(d['status']['code'], d['certificate']['certified'], d['certificate'].get('failed_gates'), d.get('objective'))"
# 2. pd6_fft parity magnitude via gdb on the debug test (break where the failing require is evaluated).
cat > "$out/parity.gdb" <<'GDB'
set pagination off
break device_time_dilated_test.cu:372
run
print *parity
print pd6_one
print pd6_four
print pd3_one
print pd3_four
continue
quit
GDB
item pd6-parity-magnitude-gdb 0 gdb -batch -x "$out/parity.gdb" build-v2-cuda-debug/cuda-tests/device_time_dilated_test
# 3. CPU-only planner baseline (no device): the same examples through the Python CPU reference path.
export CUDA_VISIBLE_DEVICES=""
item cpu-planner-pytest 0 python -m pytest -q -p no:cacheprovider tests/test_planner_cpu_reference.py tests/test_planner_schema.py tests/test_planner_viewer_export.py
for f in hcw_rendezvous powered_descent_3dof powered_descent_6dof low_thrust; do
  item "cpu-plan-$f" 0 spacepdhcg plan "examples/planner/$f.json" --backend cpu_reference --output "$out/cpu-plan-$f"
done
for f in hcw_rendezvous powered_descent_3dof powered_descent_6dof low_thrust; do
  item "cpu-plan-$f-status" 0 python -c "import json; d=json.load(open('$out/cpu-plan-$f/plan-result.json')); print(d['status']['code'], d['certificate']['certified'], d['certificate'].get('failed_gates'), d.get('objective'))"
done
log "== done"; cat "$out/items.tsv"
