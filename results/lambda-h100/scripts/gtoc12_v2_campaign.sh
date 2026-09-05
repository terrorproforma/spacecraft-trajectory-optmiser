#!/bin/bash
# GTOC12 H100 campaign v2 (CPU only): cluster_fleet_h100_v2 (22 workers, 8 h) || joint_itinerary_h100_v1 (4 workers, existing archives)
#   -> joint_itinerary_h100_v2 (new v2 chains) -> fleet_master_h100_v2 (archive-wide, LP bound) -> official + independent verification -> stats.
set -uo pipefail
log() { echo "[$(date -u +%FT%TZ)] $*"; }
source "$HOME/spacepdhcg/env.sh"
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
root="$HOME/spacepdhcg/gtoc12"; cd "$root"
export PYTHONPATH="$root/src"
py="$root/.venv/bin/python"
runs=results/gtoc12/runs
RESULT="$HOME/logs/gtoc12-v2-RESULT"
NICE=5
log "source $(git rev-parse HEAD) branch $(git branch --show-current) dirty=$(git status --porcelain=v1 | wc -l); cpus=$(nproc); load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "status=RUNNING stage=cluster-fleet+joint-v1 started_utc=$(date -u +%FT%TZ)" > "$RESULT"

# sources of the archive-wide passes: fleet_master_v7's sixteen + the H100 v1 archive (+ cluster_fleet_v8 when it has been copied)
mapfile -t FM7_SOURCES < <("$py" -c "import json; print('\n'.join(json.load(open('$runs/fleet_master_v7/run_report.json'))['sources']))")
LOCAL_SOURCES=("${FM7_SOURCES[@]}")
[ -d "$runs/cluster_fleet_v8" ] && [ -n "$(find $runs/cluster_fleet_v8 -name route_summary.json -print -quit)" ] && LOCAL_SOURCES+=("$runs/cluster_fleet_v8")
src_args() { for s in "$@"; do printf -- "--source %s " "$s"; done; }

# -- stage 0 (background, 4 workers): joint re-optimisation of every existing chain >= 450 kg, fleet_master_v7 ships first
log "== joint-itinerary joint_itinerary_h100_v1 (4 workers, ${#LOCAL_SOURCES[@]} local sources + cluster_fleet_h100_v1, fleet_master_v7 ships first)"
nice -n $NICE "$py" -u -m spacepdhcg gtoc12 joint-itinerary \
  --run-id joint_itinerary_h100_v1 --output "$runs/joint_itinerary_h100_v1" \
  $(src_args "${LOCAL_SOURCES[@]}") --source "$runs/cluster_fleet_h100_v1" \
  --fleet-report "$runs/fleet_master_v7/run_report.json" \
  --top 100000 --min-collected-kg 450 --workers 4 --per-ship-seconds 600 --budget-seconds 27000 --insert-trials 4 \
  > "$HOME/logs/gtoc12-joint_itinerary_h100_v1.log" 2>&1 < /dev/null &
JOINT1_PID=$!
log "joint v1 pid $JOINT1_PID"

# -- stage 1 (foreground, 22 workers, 8 h): wide family search
log "== cluster-fleet cluster_fleet_h100_v2 (22 workers, nice $NICE, 28800 s budget, radii 1.75,1.6 x collect+phasing bands, >=20 members, 5 ships, beam 32)"
nice -n $NICE "$py" -u -m spacepdhcg gtoc12 cluster-fleet \
  --run-id cluster_fleet_h100_v2 --output "$runs/cluster_fleet_h100_v2" \
  --workers 22 --ships-per-cluster 5 --cluster-radius 1.75,1.6 --all-family-bands --collect-epoch-families --min-members 20 \
  --beam-width 32 --refine-top 3 --max-deploys 10 --seed 0 \
  --collector-harvest --collect-dp-inflation-fit results/gtoc12/hop_inflation_fit.json --collect-dp-step-days 15 \
  --earth-prescreen-ratio 0.7 --substitution-budget-seconds 180 --return-sweep-budget-seconds 240 \
  --cluster-budget-seconds 6600 --retime-budget-seconds 900 \
  --budget-seconds 28800 --max-clusters 400 \
  > "$HOME/logs/gtoc12-cluster_fleet_h100_v2.log" 2>&1
rc=$?
log "cluster-fleet exit=$rc"
if [ "$rc" -ne 0 ]; then echo "status=FAIL stage=cluster-fleet exit=$rc" > "$RESULT"; tail -30 "$HOME/logs/gtoc12-cluster_fleet_h100_v2.log"; fi
"$py" "$HOME/s/chain_stats.py" --json "$runs/cluster_fleet_h100_v2/chain_stats.json" "$runs/cluster_fleet_h100_v2" "$runs/cluster_fleet_h100_v1" "${LOCAL_SOURCES[@]}" 2>&1 | tail -40

echo "status=RUNNING stage=wait-joint-v1" > "$RESULT"
log "== waiting for joint v1 (pid $JOINT1_PID)"; wait $JOINT1_PID; log "joint v1 exit=$?"
tail -5 "$HOME/logs/gtoc12-joint_itinerary_h100_v1.log"

# -- stage 2: joint re-optimisation of the new v2 chains (machine is free: 22 workers)
echo "status=RUNNING stage=joint-v2" > "$RESULT"
if [ -d "$runs/cluster_fleet_h100_v2/clusters" ]; then
  log "== joint-itinerary joint_itinerary_h100_v2 over cluster_fleet_h100_v2 (22 workers)"
  nice -n $NICE "$py" -u -m spacepdhcg gtoc12 joint-itinerary \
    --run-id joint_itinerary_h100_v2 --output "$runs/joint_itinerary_h100_v2" \
    --source "$runs/cluster_fleet_h100_v2" \
    --top 100000 --min-collected-kg 450 --workers 22 --per-ship-seconds 600 --budget-seconds 10800 --insert-trials 4 \
    > "$HOME/logs/gtoc12-joint_itinerary_h100_v2.log" 2>&1
  log "joint v2 exit=$?"; tail -5 "$HOME/logs/gtoc12-joint_itinerary_h100_v2.log"
fi

# -- stage 3: archive-wide master with the LP bound
echo "status=RUNNING stage=fleet-master" > "$RESULT"
MASTER_SOURCES=("${LOCAL_SOURCES[@]}" "$runs/cluster_fleet_h100_v1" "$runs/joint_itinerary_h100_v1")
[ -d "$runs/cluster_fleet_h100_v2/clusters" ] && MASTER_SOURCES+=("$runs/cluster_fleet_h100_v2")
[ -d "$runs/joint_itinerary_h100_v2/ships" ] && MASTER_SOURCES+=("$runs/joint_itinerary_h100_v2")
[ -d "$runs/cluster_fleet_v8" ] && [ -n "$(find $runs/cluster_fleet_v8 -name route_summary.json -print -quit)" ] && [[ ! " ${MASTER_SOURCES[*]} " =~ " $runs/cluster_fleet_v8 " ]] && MASTER_SOURCES+=("$runs/cluster_fleet_v8")
log "== fleet-master fleet_master_h100_v2 (${#MASTER_SOURCES[@]} sources, 22 workers): ${MASTER_SOURCES[*]}"
nice -n $NICE "$py" -u -m spacepdhcg gtoc12 fleet-master \
  --run-id fleet_master_h100_v2 --output "$runs/fleet_master_h100_v2" \
  $(src_args "${MASTER_SOURCES[@]}") --workers 22 \
  > "$HOME/logs/gtoc12-fleet_master_h100_v2.log" 2>&1
rc=$?
log "fleet-master exit=$rc"
[ "$rc" -eq 0 ] || { echo "status=FAIL stage=fleet-master exit=$rc" > "$RESULT"; tail -30 "$HOME/logs/gtoc12-fleet_master_h100_v2.log"; exit 1; }

# -- stage 4: verification (official binary + independent verifier), leg stats, chain stats
echo "status=RUNNING stage=verify" > "$RESULT"
log "== official GTOC12_Verify over every Result.txt of the new runs + independent verifier on the fleet"
"$py" "$HOME/s/verify_runs.py" cluster_fleet_h100_v2 joint_itinerary_h100_v1 joint_itinerary_h100_v2 fleet_master_h100_v2 > "$HOME/logs/gtoc12-verify_h100_v2.log" 2>&1
vrc=$?; tail -12 "$HOME/logs/gtoc12-verify_h100_v2.log"
"$py" -m spacepdhcg gtoc12 verify "$runs/fleet_master_h100_v2/fleet/Result.txt" --official > "$runs/fleet_master_h100_v2/independent_verify.txt" 2>&1; tail -4 "$runs/fleet_master_h100_v2/independent_verify.txt"
"$py" -m spacepdhcg gtoc12 leg-stats --solution "$runs/fleet_master_h100_v2/fleet/Result.txt" --output results/gtoc12/leg_stats/after_h100_v2.json > "$HOME/logs/gtoc12-legstats_h100_v2.log" 2>&1; tail -3 "$HOME/logs/gtoc12-legstats_h100_v2.log"
"$py" "$HOME/s/chain_stats.py" --json "$runs/fleet_master_h100_v2/chain_stats.json" "${MASTER_SOURCES[@]}" 2>&1 | tail -30
echo "status=$([ $vrc -eq 0 ] && echo PASS || echo VERIFY_FAIL) stage=done verify_exit=$vrc finished_utc=$(date -u +%FT%TZ)" > "$RESULT"
log "== done"