#!/bin/bash
set -uo pipefail
source "$HOME/spacepdhcg/env.sh"; export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$HOME/spacepdhcg/gtoc12"; export PYTHONPATH="$PWD/src"; runs=results/gtoc12/runs
echo "[$(date -u +%FT%TZ)] joint_itinerary_h100_v8 over cluster_fleet_v8 (4 workers)"
nice -n 5 .venv/bin/python -u -m spacepdhcg gtoc12 joint-itinerary --run-id joint_itinerary_h100_v8 --output "$runs/joint_itinerary_h100_v8" \
  --source "$runs/cluster_fleet_v8" --top 100000 --min-collected-kg 450 --workers 4 --per-ship-seconds 600 --budget-seconds 7200 --insert-trials 4 \
  > "$HOME/logs/gtoc12-joint_itinerary_h100_v8.log" 2>&1
echo "[$(date -u +%FT%TZ)] exit=$?"