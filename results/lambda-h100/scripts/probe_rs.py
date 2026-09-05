import json, os
p = "results/gtoc12/runs/cluster_fleet_h100_v1/clusters/family_0001/ship_01/route_summary.json"
d = json.load(open(p))
print(list(d.keys()))
print({k: d[k] for k in d if not isinstance(d[k], (list, dict))})
for k in d:
    if isinstance(d[k], dict): print(k, "->", list(d[k].keys())[:15])
print(os.listdir(os.path.dirname(p)))
print(os.listdir("results/gtoc12/runs/fleet_master_h100_v1"))