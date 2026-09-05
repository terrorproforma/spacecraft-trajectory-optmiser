import json, re, collections
from pathlib import Path

root = Path.home() / "spacepdhcg/gtoc12/results/gtoc12/runs"
v = json.loads((root / "fleet_master_h100_v1/official_verification.json").read_text())
rows = v["rows"]
kinds = collections.Counter()
fails = collections.Counter()
msgs = collections.Counter()
for r in rows:
    p = r["path"]
    if p.endswith("/fleet/Result.txt"):
        kind = "fleet-solution:" + p.split("/")[3]
    elif "/fleets/" in p:
        kind = "cluster-fleet-candidate"
    elif "/clusters/" in p or "/columns/" in p:
        kind = "per-ship-diagnostic"
    else:
        kind = "other:" + p
    kinds[kind] += 1
    if not r["ok"]:
        fails[kind] += 1
        msgs[re.sub(r"line\d+", "lineN", r["message"])[:120]] += 1
print("passed", v["passed"], "of", v["total"])
for k in sorted(kinds):
    print(f"  {k:40s} total {kinds[k]:4d} failed {fails[k]:4d}")
print("failure messages:")
for m, n in msgs.most_common():
    print(f"  {n:4d}  {m}")
print("fleet solutions:")
for r in rows:
    if r["path"].endswith("/fleet/Result.txt"):
        print("  ", json.dumps(r))
# the failing per-ship files: are they collector (slot>1) ships of cooperative clusters?
slots = collections.Counter()
for r in rows:
    if not r["ok"]:
        m = re.search(r"ship_(\d+)", r["path"])
        slots[m.group(1) if m else "?"] += 1
print("failing per-ship slots:", dict(slots))
# how did the archived WSL fleet masters look?
for prev in sorted(root.glob("fleet_master_v*/official_verification.json")) + sorted(root.glob("fleet10_master_v1/official_verification.json")):
    d = json.loads(prev.read_text())
    print(prev.parent.name, "passed", d.get("passed"), "of", d.get("total"))
# the cluster-fleet run report's own verification of its fleet
cf = json.loads((root / "cluster_fleet_h100_v1/run_report.json").read_text())
print("cluster_fleet_h100_v1 best/final verification:", json.dumps(cf.get("final_fleet", {}).get("verification") or cf.get("best", {}).get("official") or cf.get("official"))[:400])
print("cluster_fleet report keys:", list(cf.keys()))
