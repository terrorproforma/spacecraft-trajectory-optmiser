#!/usr/bin/env bash
# Checkpoint status (counts) plus the last group_finished events from the worker log.
# Usage: g4-status.sh [N]   (N = number of recent group_finished events to show, default 5)
source /home/ubuntu/s/g4env-h100.sh
cd "$root"
"${py}" scripts/gpu/run_g4_campaign.py status --claim-core \
  --amendment "${root}/benchmarks/g4_claim_core_amendment_v1_2.json" --repository "$root" --campaign "$campaign"
echo "worker: $(pgrep -f 'run_g4_campaign.py run --claim-core' | head -1 || echo none)  executor: $(pgrep -f 'device_scvx_integration_test --g4-session' | head -1 || echo none)  observer: $(pgrep -f 'g4-observer.sh' | head -1 || echo none)"
w=$(pgrep -f 'run_g4_campaign.py run --claim-core' | head -1); [ -n "$w" ] && echo "worker affinity: $(taskset -cp "$w" 2>/dev/null | cut -d: -f2)"
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)  compute apps: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
echo "results: $(find "$campaign/runs" -name result.json 2>/dev/null | wc -l)  contaminated attempts flagged: $(grep -l '"contaminated": true' "$campaign"/runs/*/attempts/*.json 2>/dev/null | wc -l)"
grep -h '"event": "group_finished"' "$g4logs/worker.err" 2>/dev/null | tail -n "${1:-5}"
