import json, pathlib, statistics
for run in ("joint_itinerary_h100_v1", "joint_itinerary_h100_v2"):
    q = pathlib.Path("results/gtoc12/runs", run, "ships.jsonl")
    if not q.exists(): continue
    rows = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
    b = [r["before_kg"] for r in rows]; a = [r["after_kg"] for r in rows]; g = [r["gain_kg"] for r in rows]
    print(run, "ships", len(rows), "improved", sum(x > 1e-6 for x in g), "gain total", round(sum(g), 1), "mean gain", round(statistics.mean(g), 2), "median", round(statistics.median(g), 2), "max", round(max(g), 1))
    for th in (550, 575, 590, 600, 610, 625, 650):
        print(f"  >= {th}: before {sum(x >= th for x in b):3d} after {sum(x >= th for x in a):3d}")
    fleet = [r for r in rows if r.get("in_fleet")]
    print("  fleet ships", len(fleet), "before avg", round(statistics.mean(r["before_kg"] for r in fleet), 2) if fleet else None, "after avg", round(statistics.mean(r["after_kg"] for r in fleet), 2) if fleet else None)
    ws = [r["wall_seconds"] for r in rows]; print("  wall per ship min/med/max", round(min(ws)), round(statistics.median(ws)), round(max(ws)), "certs", sum(r.get("certifications", 0) for r in rows), "inserted", sum(1 for r in rows if r.get("inserted")))
    top = sorted(rows, key=lambda r: -r["after_kg"])[:8]
    for r in top: print(f"   {r['after_kg']:.1f} (was {r['before_kg']:.1f}, +{r['gain_kg']:.1f}) {r['group']} slot {r['slot']} in_fleet={r.get('in_fleet')}")