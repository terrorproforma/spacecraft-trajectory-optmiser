#!/bin/bash
# Re-run the two literature GPU legs that the self-refusing preflight blocked, on the fixed v2 HEAD.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export V2="$HOME/spacepdhcg/v2"
cd "$V2"
export PATH="$V2/.venv/bin:/usr/local/cuda-12.8/bin:$HOME/.local/node/bin:$PATH"
export PYTHONPATH="$V2/src"
export CUDA_VISIBLE_DEVICES=0
export SPACEPDHCG_NATIVE_LIBRARY="$V2/build-v2-relwithdebinfo/libspacepdhcg.so"
export SPACEPDHCG_QOCO_LIBRARY="$HOME/spacepdhcg/v1/build-current-head-qoco/libqoco.so"
export LD_LIBRARY_PATH="$HOME/spacepdhcg/v1/build-current-head-qoco-cudss-lib:$V2/.venv/lib/python3.12/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out="$V2/results/gpu/h100-deferred-3373988/literature-rerun-$(git rev-parse --short HEAD)"
mkdir -p "$out"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { log "GPU busy: refuse"; exit 3; }
printf 'status=RUNNING\nsource_commit=%s\nstarted_utc=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)" > "$out/status.txt"
log "preflight"; spacepdhcg literature gpu-preflight > "$out/gpu-preflight.log" 2>&1; echo "preflight exit=$?" | tee -a "$out/gpu-preflight.log"
log "gpu-run blackmore + chari (expected exit 2: chari stays gap by design)"
t0=$SECONDS
spacepdhcg literature gpu-run blackmore-2010-pd3-case1 chari-2024-pd6-monte-carlo > "$out/gpu-run.log" 2>&1
rc=$?
log "gpu-run exit=$rc after $((SECONDS-t0))s"
cp results/literature/acikmese-ploen-2007-pd3.json results/literature/blackmore-2010-pd3-case1.json results/literature/chari-2024-pd6-monte-carlo.json "$out/"
git status --porcelain -- results/literature docs/REFERENCE_REPRODUCTION_REPORT.md benchmarks/literature/reference_reproduction.json > "$out/git-status.txt"
git diff --stat -- docs/REFERENCE_REPRODUCTION_REPORT.md benchmarks/literature/reference_reproduction.json >> "$out/git-status.txt"
python3 "$HOME/s/lit_peek2.py" > "$out/summary.txt" 2>&1
python3 "$HOME/s/lit_peek3.py" >> "$out/summary.txt" 2>&1
verdict=FAIL; [ "$rc" -eq 2 ] && verdict=PASS
printf 'status=%s\ngpu_run_exit=%s\nexpected_exit=2\ncompleted_utc=%s\n' "$verdict" "$rc" "$(date -u +%FT%TZ)" > "$out/status.txt"
cat "$out/status.txt"; cat "$out/summary.txt"
