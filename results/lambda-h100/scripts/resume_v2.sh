#!/bin/bash
# Resume the v2 pipeline after cluster_fleet_h100_v2 (and joint v1) have finished or died: joint v2 -> fleet master -> verify.
# Usage: setsid nohup bash ~/s/resume_v2.sh > ~/logs/resume_v2.sh.log 2>&1 < /dev/null &
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"; export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root="$HOME/spacepdhcg/gtoc12"; cd "$root"; export PYTHONPATH="$root/src"; py="$root/.venv/bin/python"; runs=results/gtoc12/runs
RESULT="$HOME/logs/gtoc12-v2-RESULT"; NICE=5
pgrep -f "spacepdhcg gtoc12 (cluster-fleet|joint-itinerary|fleet-master)" >/dev/null && { log "gtoc12 processes still running; refusing to resume"; exit 2; }
mapfile -t LOCAL_SOURCES < <("$py" -c "import json; print('\n'.join(json.load(open('$runs/fleet_master_v7/run_report.json'))['sources']))")
[ -d "$runs/cluster_fleet_v8" ] && LOCAL_SOURCES+=("$runs/cluster_fleet_v8")
src_args() { for s in "$@"; do printf -- "--source %s " "$s"; done; }
if [ -d "$runs/cluster_fleet_h100_v2/clusters" ] && [ ! -f "$runs/joint_itinerary_h100_v2/run_report.json" ]; then
  echo "status=RUNNING stage=joint-v2 (resume)" > "$RESULT"
  log "== joint_itinerary_h100_v2 (22 workers)"
  nice -n $NICE "$py" -u -m spacepdhcg gtoc12 joint-itinerary --run-id joint_itinerary_h100_v2 --output "$runs/joint_itinerary_h100_v2" \
    --source "$runs/cluster_fleet_h100_v2" --top 100000 --min-collected-kg 450 --workers 22 --per-ship-seconds 600 --budget-seconds 10800 --insert-trials 4 \
    > "$HOME/logs/gtoc12-joint_itinerary_h100_v2.log" 2>&1; log "joint v2 exit=$?"
fi
MASTER_SOURCES=("${LOCAL_SOURCES[@]}" "$runs/cluster_fleet_h100_v1")
[ -d "$runs/joint_itinerary_h100_v1/ships" ] && MASTER_SOURCES+=("$runs/joint_itinerary_h100_v1")
[ -d "$runs/cluster_fleet_h100_v2/clusters" ] && MASTER_SOURCES+=("$runs/cluster_fleet_h100_v2")
[ -d "$runs/joint_itinerary_h100_v2/ships" ] && MASTER_SOURCES+=("$runs/joint_itinerary_h100_v2")
echo "status=RUNNING stage=fleet-master (resume)" > "$RESULT"
log "== fleet_master_h100_v2 (${#MASTER_SOURCES[@]} sources)"
nice -n $NICE "$py" -u -m spacepdhcg gtoc12 fleet-master --run-id fleet_master_h100_v2 --output "$runs/fleet_master_h100_v2" $(src_args "${MASTER_SOURCES[@]}") --workers 22 > "$HOME/logs/gtoc12-fleet_master_h100_v2.log" 2>&1
rc=$?; log "fleet-master exit=$rc"; [ $rc -eq 0 ] || { echo "status=FAIL stage=fleet-master exit=$rc" > "$RESULT"; exit 1; }
echo "status=RUNNING stage=verify (resume)" > "$RESULT"
"$py" "$HOME/s/verify_runs.py" cluster_fleet_h100_v2 joint_itinerary_h100_v1 joint_itinerary_h100_v2 fleet_master_h100_v2 > "$HOME/logs/gtoc12-verify_h100_v2.log" 2>&1; vrc=$?
"$py" -m spacepdhcg gtoc12 verify "$runs/fleet_master_h100_v2/fleet/Result.txt" --official > "$runs/fleet_master_h100_v2/independent_verify.txt" 2>&1
"$py" -m spacepdhcg gtoc12 leg-stats --solution "$runs/fleet_master_h100_v2/fleet/Result.txt" --output results/gtoc12/leg_stats/after_h100_v2.json > "$HOME/logs/gtoc12-legstats_h100_v2.log" 2>&1
"$py" "$HOME/s/chain_stats.py" --json "$runs/fleet_master_h100_v2/chain_stats.json" "${MASTER_SOURCES[@]}" > "$HOME/logs/gtoc12-chainstats_h100_v2.log" 2>&1
echo "status=$([ $vrc -eq 0 ] && echo PASS || echo VERIFY_FAIL) stage=done verify_exit=$vrc finished_utc=$(date -u +%FT%TZ)" > "$RESULT"
log "== done"