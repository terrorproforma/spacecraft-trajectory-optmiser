bash ~/s/v2_status.sh 2>&1 | tail -9 | cut -c1-300
cd ~/spacepdhcg/gtoc12 && python3 ~/s/chain_stats.py results/gtoc12/runs/cluster_fleet_h100_v2 results/gtoc12/runs/cluster_fleet_h100_v1 results/gtoc12/runs/cluster_fleet_v8 results/gtoc12/runs/cluster_fleet_v7 results/gtoc12/runs/joint_itinerary_v2 results/gtoc12/runs/joint_itinerary_h100_v1 results/gtoc12/runs/joint_itinerary_h100_v8 2>&1 | tail -9
echo "-- joint v8:"; python3 - <<'PY'
import json, pathlib
q = pathlib.Path("results/gtoc12/runs/joint_itinerary_h100_v8/ships.jsonl")
if q.exists():
    rows = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
    print(len(rows), "done of", rows[-1].get("total"), "improved", sum(r["gain_kg"] > 1e-6 for r in rows), "gain", round(sum(r["gain_kg"] for r in rows), 1), ">=600 before", sum(r["before_kg"] >= 600 for r in rows), "after", sum(r["after_kg"] >= 600 for r in rows))
PY
echo "families logged: $(grep -c '"family"' ~/logs/gtoc12-cluster_fleet_h100_v2.log)"