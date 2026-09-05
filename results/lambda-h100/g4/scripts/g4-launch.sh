#!/usr/bin/env bash
# Stage 4/5: exclusivity + affinity checks, then launch the single serialized worker and the observer
# under nohup with logs in ~/logs/g4-h100/.
set -uo pipefail
source /home/ubuntu/s/g4env-h100.sh
mkdir -p "$g4logs"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
launchlog="$g4logs/launch.log"
{
echo "[$(ts)] pre-launch checks"
# 1. no worker/executor already
if pgrep -f 'run_g4_campaign.py run --claim-core' >/dev/null; then echo "a worker is already running; refuse"; exit 3; fi
# 2. GPU exclusivity: wait (up to 30 min) for any foreign compute process to end, noting it.
for i in $(seq 1 180); do
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  [ -z "$apps" ] && break
  echo "[$(ts)] foreign compute process present, waiting: $apps"
  for p in $(echo "$apps" | cut -d, -f1); do echo "   pid $p cwd=$(readlink /proc/$p/cwd 2>/dev/null) cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200)"; done
  sleep 10
done
apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
[ -z "$apps" ] || { echo "GPU still busy after 30 min: $apps"; exit 3; }
echo "[$(ts)] nvidia-smi compute apps: none (exclusive)"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
# 3. CPU affinity: our GTOC12 workers (if any) must not overlap cores 0-3.
moved=0
for p in $(pgrep -f 'spacepdhcg gtoc12|gtoc12 cluster-fleet|gtoc12 fleet-master|cluster_fleet_h100' 2>/dev/null); do
  aff=$(taskset -cp "$p" 2>/dev/null | awk -F: '{gsub(/ /,"",$2); print $2}')
  case "$aff" in
    4-25|4-*|1[0-9]-25) ;;  # already clear of 0-3
    *) echo "[$(ts)] gtoc12 pid $p affinity $aff overlaps cores 0-3 -> taskset -a -cp 4-25"; taskset -a -cp 4-25 "$p" >/dev/null 2>&1 && moved=$((moved+1)) ;;
  esac
done
echo "[$(ts)] gtoc12 processes found: $(pgrep -fc 'spacepdhcg gtoc12|gtoc12 cluster-fleet|gtoc12 fleet-master|cluster_fleet_h100' || true); moved to 4-25: $moved"
echo "[$(ts)] load: $(cut -d' ' -f1-3 /proc/loadavg); top cpu consumers:"; ps -eo pid,psr,pcpu,comm --sort=-pcpu | head -5
# 4. launch observer then worker (worker pinned to cores 0-3 inside g4-worker.sh)
nohup bash /home/ubuntu/s/g4-observer.sh > "$g4logs/observer.nohup.log" 2>&1 < /dev/null &
obs=$!
nohup bash /home/ubuntu/s/g4-worker.sh >> "$g4logs/worker.log" 2>> "$g4logs/worker.err" < /dev/null &
wk=$!
echo "[$(ts)] LAUNCHED worker_pid=$wk observer_pid=$obs"
echo "launched_utc=$(ts) worker_pid=$wk observer_pid=$obs" > "$g4logs/launch.txt"
sleep 45
echo "[$(ts)] worker alive: $(kill -0 $wk 2>/dev/null && echo yes || echo NO); affinity: $(taskset -cp $wk 2>/dev/null | cut -d: -f2)"
echo "executor: $(pgrep -f 'device_scvx_integration_test --g4-session' | head -1 || echo none)  affinity: $(taskset -cp $(pgrep -f 'device_scvx_integration_test --g4-session' | head -1) 2>/dev/null | cut -d: -f2)"
echo "nvidia-smi compute apps now: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
tail -5 "$g4logs/worker.log"; tail -3 "$g4logs/worker.err"
} 2>&1 | tee -a "$launchlog"
