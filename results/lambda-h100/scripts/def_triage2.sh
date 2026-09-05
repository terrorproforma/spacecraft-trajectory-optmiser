#!/bin/bash
source "$HOME/spacepdhcg/env.sh"
cd "$HOME/spacepdhcg/v2" || exit 1
export LD_LIBRARY_PATH="$HOME/spacepdhcg/v1/build-current-head-qoco-cudss-lib:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
echo "== v2 QOCO variant refs"; grep -rn '09f0495\|abstol' docs/GPU_DEFERRED_VALIDATION_V2.md benchmarks/gpu_deferred_validation_v2.json scripts/gpu/*.sh 2>/dev/null | cut -c1-260 | head -12
echo "== v1 qoco build provenance"; ls "$HOME/spacepdhcg/v1/build-current-head-qoco/" | head; cat "$HOME/spacepdhcg/v1/results/gpu/current-head-9e75b47-h100/preflight/qoco.txt" 2>/dev/null | head -20; ls "$HOME/spacepdhcg/v1/results/gpu/current-head-9e75b47-h100/preflight/"
echo "== upstream qoco HEAD"; (cd "$HOME/spacepdhcg/v1/_upstream" 2>/dev/null && ls; for d in "$HOME"/spacepdhcg/v1/_upstream/qoco*; do echo "$d: $(git -C "$d" rev-parse --short HEAD 2>/dev/null) $(git -C "$d" status --porcelain 2>/dev/null | wc -l) dirty"; done)
echo "== device_time_dilated_test direct (release)"; CUDA_VISIBLE_DEVICES=0 build-v2-cuda-release/cuda-tests/device_time_dilated_test 2>&1 | tail -5
echo "== plan output dir"; ls build-v2-gpu-deferred/plan-powered_descent_3dof/ build-v2-gpu-deferred/plan-hcw_rendezvous/
echo "== hcw plan status"; python3 -c "
import json; d=json.load(open('build-v2-gpu-deferred/plan-hcw_rendezvous/plan-result.json')) if __import__('os').path.exists('build-v2-gpu-deferred/plan-hcw_rendezvous/plan-result.json') else None
print(json.dumps({k:d[k] for k in d if k in ('status','certificate','backend','iterations')}, indent=1)[:2500] if d else 'no plan-result.json')
"
echo "== pd3 plan status"; python3 -c "
import json,os; p='build-v2-gpu-deferred/plan-powered_descent_3dof/plan-result.json'
d=json.load(open(p)) if os.path.exists(p) else None
print(json.dumps({k:d[k] for k in d if k in ('status','certificate','backend')}, indent=1)[:2500] if d else 'no plan-result.json')
"
