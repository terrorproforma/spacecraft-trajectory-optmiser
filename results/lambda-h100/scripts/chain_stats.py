"""Chain-mass distribution of route archives: python chain_stats.py [--json out] <run dir>...

One chain per (run, asteroid set): the best certified variant's total_collected_kg.  Prints the
count of chains >= 600 / 650 / 700 kg per run and over the union of runs (unique asteroid sets).
"""
import json, pathlib, sys
args = sys.argv[1:]
out = None
if args and args[0] == "--json":
    out = args[1]; args = args[2:]
report = {}
union: dict[frozenset, float] = {}
for run in args:
    root = pathlib.Path(run)
    chains: dict[frozenset, float] = {}
    n_files = 0
    for path in sorted(root.rglob("route_summary.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if not d.get("certified", True):
            continue
        n_files += 1
        key = frozenset(int(a) for a in d.get("asteroids", []))
        kg = float(d.get("total_collected_kg", 0.0))
        chains[key] = max(chains.get(key, 0.0), kg)
        union[key] = max(union.get(key, 0.0), kg)
    masses = sorted(chains.values(), reverse=True)
    row = {
        "routes": n_files, "chains": len(masses),
        "ge_550": sum(m >= 550 for m in masses), "ge_600": sum(m >= 600 for m in masses),
        "ge_650": sum(m >= 650 for m in masses), "ge_700": sum(m >= 700 for m in masses),
        "max_kg": masses[0] if masses else None,
        "top5": [round(m, 1) for m in masses[:5]],
    }
    report[root.name] = row
    print(f"{root.name:38s} routes {n_files:4d} chains {len(masses):4d} >=550 {row['ge_550']:3d} >=600 {row['ge_600']:3d} >=650 {row['ge_650']:3d} >=700 {row['ge_700']:3d} max {row['max_kg'] if masses else 0:.1f}")
masses = sorted(union.values(), reverse=True)
row = {"chains": len(masses), "ge_550": sum(m >= 550 for m in masses), "ge_600": sum(m >= 600 for m in masses),
       "ge_650": sum(m >= 650 for m in masses), "ge_700": sum(m >= 700 for m in masses),
       "max_kg": masses[0] if masses else None, "top10": [round(m, 1) for m in masses[:10]]}
report["UNION"] = row
print(f"{'UNION (unique asteroid sets)':38s} {'':11s} chains {len(masses):4d} >=550 {row['ge_550']:3d} >=600 {row['ge_600']:3d} >=650 {row['ge_650']:3d} >=700 {row['ge_700']:3d} max {row['max_kg'] if masses else 0:.1f}")
print("top10:", row["top10"])
if out:
    pathlib.Path(out).write_text(json.dumps(report, indent=1) + "\n")