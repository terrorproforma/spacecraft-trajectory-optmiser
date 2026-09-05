#!/usr/bin/env bash
# Lightweight external observer: executor PID/threads/affinity, GPU state, every compute context
# (native pmon) and the shared GPU lock payload every 60 s, so a silent generation or a foreign
# process can be localised afterwards.
source /home/ubuntu/s/g4env-h100.sh
log="$g4logs/observer.log"
while true; do
  pid=$(pgrep -f "device_scvx_integration_test --g4-session" | head -1)
  if [ -n "$pid" ]; then
    tids=$(ls /proc/$pid/task 2>/dev/null | wc -l)
    etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    aff=$(taskset -cp "$pid" 2>/dev/null | awk -F: '{gsub(/ /,"",$2); print $2}')
  else
    tids=""; etime=""; aff=""
  fi
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  apps=$(nvidia-smi pmon -c 1 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && $3=="C" {printf "%s:%s:sm%s,", $2, $10, $4}')
  # SHARED_GPU_LOCK_FILE is hard-coded to /home/angus/.spacepdhcg-gpu.lock in run_g4_campaign.py;
  # /home/angus exists on this host only as an ubuntu-owned directory for that advisory lock.
  lock=$(tr -d '\n' < /home/angus/.spacepdhcg-gpu.lock 2>/dev/null | cut -c1-160)
  results=$(find "$campaign/runs" -name result.json 2>/dev/null | wc -l)
  load=$(cut -d' ' -f1-3 /proc/loadavg)
  echo "$(date -u +%FT%TZ) exec_pid=${pid:-none} etime=${etime} tids=${tids} aff=${aff} gpu=${gpu} compute=${apps} results=${results} load=${load} lock=${lock}" >> "$log"
  sleep 60
done
