#!/bin/bash
cd "$HOME/spacepdhcg/gtoc12" || exit 1
echo "== cluster-fleet log tail"; tail -c 2500 "$HOME/logs/gtoc12-cluster_fleet_h100_v1.log"; echo
echo "== run dir"; ls results/gtoc12/runs/cluster_fleet_h100_v1/ | head -30
for f in results/gtoc12/runs/cluster_fleet_h100_v1/*.json; do echo "--- $f"; head -c 1500 "$f"; echo; done 2>/dev/null | head -120
echo "== campaign log"; tail -6 "$HOME/logs/gtoc12_campaign.sh.log"
echo "== fleet-master log tail"; tail -c 1200 "$HOME"/logs/gtoc12-fleet_master_h100_v1.log 2>/dev/null; echo
echo "== reference (WSL v6) numbers in docs"; grep -rn 'kg' docs/GTOC12*.md 2>/dev/null | grep -i 'v6\|fleet' | head -8
