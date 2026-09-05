import json
d = json.load(open("results/gtoc12/runs/fleet_master_v7/run_report.json"))
print("keys:", list(d.keys()))
for k in ("sources", "source_directories", "archives", "settings"):
    if k in d: print(k, json.dumps(d[k])[:1200])
m = d.get("master", {})
print("master keys:", list(m.keys()))
print("lp:", {k: m.get(k) for k in ("lp_bound_kg","lp_gap_kg","exhaustive","columns","collected_kg","gap_kg","ships")})
j = json.load(open("results/gtoc12/runs/joint_itinerary_v2/run_report.json"))
print("joint keys:", list(j.keys()))
print("joint settings:", json.dumps(j.get("settings"))[:800])
print("joint summary:", json.dumps({k: j[k] for k in j if k not in ("ships","settings","tasks","sources")})[:1200])
import itertools
rows = [json.loads(l) for l in open("results/gtoc12/runs/joint_itinerary_v2/ships.jsonl")]
print("ship record keys:", list(rows[0].keys()))
print("row0:", json.dumps(rows[0])[:700])