#!/bin/bash
echo "== lit rerun"; tail -3 "$HOME/logs/lit_rerun.sh.log"; cat "$HOME"/spacepdhcg/v2/results/gpu/h100-deferred-3373988/literature-rerun-*/status.txt 2>/dev/null
echo "== gtoc12"; cat "$HOME/logs/gtoc12-RESULT"; grep -o '"done": [0-9]*' "$HOME/logs/gtoc12-fleet_master_h100_v1.log" | tail -1; tail -3 "$HOME/logs/gtoc12_campaign.sh.log" | cut -c1-300
echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime
