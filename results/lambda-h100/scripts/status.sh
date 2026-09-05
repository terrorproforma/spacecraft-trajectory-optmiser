#!/bin/bash
echo "== $(date -u +%FT%TZ) load: $(cut -d' ' -f1-3 /proc/loadavg) mem: $(free -g | awk '/Mem/{print $3"G/"$2"G"}')"
echo "== reseal"; tail -4 ~/logs/reseal_all.sh.log; cat ~/logs/reseal-*/RESULT 2>/dev/null
ev=$(ls -d ~/spacepdhcg/v1/results/gpu/current-head-*-h100 2>/dev/null | head -1)
for g in g0 g1 g2 g3; do
  if [ -f "$ev/$g/status.txt" ]; then printf '%s: %s | last: %s\n' "$g" "$(head -1 $ev/$g/status.txt)" "$(tail -1 ~/logs/reseal-*/$g.log 2>/dev/null)"; fi
done
echo "== v2 build"; tail -3 ~/logs/v2_build.sh.log
echo "== gtoc12"; cat ~/logs/gtoc12-RESULT 2>/dev/null
echo "families priced: $(ls ~/spacepdhcg/gtoc12/results/gtoc12/runs/cluster_fleet_h100_v1/clusters 2>/dev/null | wc -l); fleets: $(ls ~/spacepdhcg/gtoc12/results/gtoc12/runs/cluster_fleet_h100_v1/fleets 2>/dev/null | wc -l); last incumbent: $(grep -o '"incumbent_kg": [0-9.]*' ~/logs/gtoc12-cluster_fleet_h100_v1.log 2>/dev/null | tail -1); elapsed: $(grep -o '"elapsed_minutes": [0-9.]*' ~/logs/gtoc12-cluster_fleet_h100_v1.log 2>/dev/null | tail -1)"
tail -2 ~/logs/gtoc12_campaign.sh.log 2>/dev/null
echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
