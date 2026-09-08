"""Bound incumbent + persisted primary pilot routes without any new flight solves."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
bonus_path = OUT / "bonus_coefficients.txt"
weights = [float(line.split()[0]) for line in bonus_path.read_text().splitlines() if line.strip()]
assert len(weights) == 60000 and all(0 < w <= 1 for w in weights)


def read(path):
    return json.loads((ROOT / path).read_text())


def score(route):
    return sum(m * weights[int(a) - 1] for a, m in route["collected_mass_kg"].items())


inc_dir = ROOT / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
inc = [json.loads(p.read_text()) for p in sorted(inc_dir.glob("ship-*.json"))]
pilot_dir = "build/performance/family-gpu-v588/output/clusters/family_0032"
new = [read(f"{pilot_dir}/ship_{i:02d}/route_summary.json") for i in (1, 2, 3)]
assert len(inc) == 23 and all(not r["plan"]["foreign_deploy_epochs"] for r in inc)

# Pilot ship 1 can always be replaced by retained ship 6: same asteroid footprint,
# no dependencies, greater physical mass and greater fixed-bonus objective.
assert set(new[0]["plan"]["deploy_epochs"]) == set(inc[5]["plan"]["deploy_epochs"])
assert set(new[0]["plan"]["collect_epochs"]) == set(inc[5]["plan"]["collect_epochs"])
assert not new[0]["plan"]["foreign_deploy_epochs"] and not new[0]["plan"]["orphaned"]
assert new[0]["total_collected_kg"] < inc[5]["total_collected_kg"] and score(new[0]) < score(inc[5])
inc_asteroids = {a for r in inc for a in r["plan"]["deploy_epochs"]}
assert all(not inc_asteroids.intersection(r["plan"]["deploy_epochs"]) for r in new[1:])
foreign = new[2]["plan"]["foreign_deploy_epochs"]
assert set(map(int, foreign)) == set(new[1]["plan"]["orphaned"]) == {49895, 12043}
assert all(abs(t - new[1]["plan"]["deploy_epochs"][a]) < 1e-9 for a, t in foreign.items())

base_mass = sum(r["total_collected_kg"] for r in inc)
base_score = sum(map(score, inc))
pair_mass = sum(r["total_collected_kg"] for r in new[1:])
pair_score = sum(map(score, new[1:]))
mass_desc = sorted((r["total_collected_kg"] for r in inc), reverse=True)
score_desc = sorted(map(score, inc), reverse=True)
rows = []
# Uncollected deployments are allowed. Ship 2 alone is admissible; ship 3 alone
# is not because its foreign miners require ship 2 at matching deploy epochs.
for included in ([new[1]], new[1:]):
    added = len(included)
    for count in range(added, 24 + added):
        retained = count - added
        maximum_mass = sum(r["total_collected_kg"] for r in included) + sum(mass_desc[:retained])
        required_mass = count * math.log(count / 2) / 0.004
        objective_bound = sum(map(score, included)) + sum(score_desc[:retained])
        rows.append({"new_ships": [2] if added == 1 else [2, 3], "ships": count, "retained_incumbent_ships": retained,
                     "physical_mass_upper_bound_kg": maximum_mass,
                     "physical_mass_required_kg": required_mass,
                     "weighted_objective_upper_bound_kg": objective_bound,
                     "excluded_by": "ship_count_rule" if maximum_mass < required_mass - 1e-8 else "weighted_objective_bound" if objective_bound < base_score - 1e-8 else "UNRESOLVED"})
assert all(r["excluded_by"] != "UNRESOLVED" for r in rows)

old = read("results/gtoc12/runs/cluster_fleet_v10_control/clusters/family_0032/ship_02/route_summary.json")
epoch_rows = []
for asteroid in old["plan"]["collect_epochs"]:
    old_stay = old["plan"]["collect_epochs"][asteroid] - old["plan"]["deploy_epochs"][asteroid]
    new_stay = new[1]["plan"]["collect_epochs"][asteroid] - new[1]["plan"]["deploy_epochs"][asteroid]
    epoch_rows.append({"asteroid": int(asteroid), "old_mining_days": old_stay,
                       "new_mining_days": new_stay, "change_days": new_stay - old_stay,
                       "raw_mass_change_kg": new[1]["collected_mass_kg"][asteroid] - old["collected_mass_kg"][asteroid]})

certificate = {
    "status": "all_persisted_primary_additions_excluded_from_improving_the_retained_fleet",
    "scope": "23 retained incumbent routes plus 3 primary routes persisted by either v588 or v589; no unseen historical alternatives or lost in-memory variants are covered",
    "method": "Exact dependence/dominance checks followed by conservative cardinality bounds; no GPU work, SCvx, MILP, or trajectory integration",
    "bonus_sha256": hashlib.sha256(bonus_path.read_bytes()).hexdigest(),
    "incumbent_raw_kg": base_mass, "incumbent_weighted_kg": base_score,
    "incumbent_rule_headroom_raw_kg": base_mass - 23 * math.log(23 / 2) / 0.004,
    "pilot_per_ship_raw_kg": [r["total_collected_kg"] for r in new],
    "pilot_per_ship_weighted_kg": list(map(score, new)),
    "dominated_pilot_ship_1": {"pilot_raw_kg": new[0]["total_collected_kg"], "pilot_weighted_kg": score(new[0]), "incumbent_ship": 6, "incumbent_raw_kg": inc[5]["total_collected_kg"], "incumbent_weighted_kg": score(inc[5])},
    "new_cooperative_pair": {"raw_kg": pair_mass, "weighted_kg": pair_score, "foreign_asteroids": sorted(map(int, foreign))},
    "cardinality_bounds": rows,
    "ship_2_historical_vs_pilot": {"old_raw_kg": old["total_collected_kg"], "new_raw_kg": new[1]["total_collected_kg"], "old_weighted_kg": score(old), "new_weighted_kg": score(new[1]), "summed_mining_day_change": sum(r["change_days"] for r in epoch_rows), "per_asteroid": epoch_rows},
}
(OUT / "subset-certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
print(json.dumps({k: v for k, v in certificate.items() if k not in ("cardinality_bounds", "ship_2_historical_vs_pilot")}, indent=2))
print(json.dumps(rows[-4:], indent=2))
