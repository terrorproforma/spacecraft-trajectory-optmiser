#!/bin/bash
cd ~/spacepdhcg/gtoc12
python3 - <<'PY'
import json
d = json.load(open("results/gtoc12/runs/cluster_fleet_v6/run_report.json"))
tl = d.get("timeline")
print("timeline type:", type(tl).__name__, "len", len(tl) if hasattr(tl, "__len__") else "")
if isinstance(tl, list):
    for row in tl[:8]:
        print(" ", json.dumps(row)[:200])
elif isinstance(tl, dict):
    for k in list(tl)[:12]:
        print(" ", k, "=", json.dumps(tl[k])[:160])
print("settings:", json.dumps(d.get("settings"))[:800])
print("wall_seconds_total:", d.get("wall_seconds_total"), "status:", d.get("status"))
PY
echo "== our run"; ls -la results/gtoc12/runs/cluster_fleet_h100_v1/; cat ~/logs/gtoc12-cluster_fleet_h100_v1.log | tail -5
echo "== worker cpu"; ps -eo pid,pcpu,rss,etime,args --sort=-pcpu | grep cluster-fleet | grep -v grep | wc -l
