set -eu
cd /home/angus/worktrees/spacepdhcg-gtoc12/results/gtoc12/runs
python3 - <<'PY'
import json
d = json.load(open("cluster_fleet_v8/run_report.json"))
print("v8 status", d.get("status"), "wall", round(d.get("wall_seconds_total", 0)), "bundles", len(d.get("bundles", [])), "best", (d.get("best") or {}).get("score_kg"))
raise SystemExit(0 if d.get("status") else 3)
PY
find cluster_fleet_v8 \( -name run_report.json -o -name bundle.json -o -name route_summary.json -o -name fleet.json -o -path 'cluster_fleet_v8/fleet/*' \) -not -path '*/viewer/trajectories.json' -print0 | tar czf /mnt/c/Users/Angus/h100work/cluster_fleet_v8.tgz --null -T -
ls -la /mnt/c/Users/Angus/h100work/cluster_fleet_v8.tgz; tar tzf /mnt/c/Users/Angus/h100work/cluster_fleet_v8.tgz | wc -l