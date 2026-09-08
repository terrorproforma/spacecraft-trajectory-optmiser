"""Read exact retained event pairs and summarize the archive/source cap finding."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def main():
    target = ROOT / "conclusion.json"
    if target.exists():
        raise FileExistsError("Preserve existing conclusion")
    pool = json.loads((ROOT / "report.json").read_text())
    path = REPO / "build/performance/coupled-fuel-search-v611/inputs/Result.txt"
    pairs = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        ship, event = map(int, parts[:2])
        if event > 0:
            pairs[(ship, event, float(parts[2]))].append(float(parts[-1]))
    ships = {i: {"ship": i, "deploys": [], "collects": [], "raw_kg": 0.0} for i in range(1, 24)}
    for (ship, asteroid, epoch), masses in pairs.items():
        assert len(masses) == 2
        delta = masses[1] - masses[0]
        if abs(delta + 40) < 1e-6:
            ships[ship]["deploys"].append(asteroid)
        elif delta > 1e-6:
            ships[ship]["collects"].append(asteroid)
            ships[ship]["raw_kg"] += delta
        else:
            raise AssertionError((ship, asteroid, epoch, delta))
    total_raw = sum(row["raw_kg"] for row in ships.values())
    assert abs(total_raw - 14051.854893908598) < 1e-7
    ten = [row for row in pool["all_route_records"] if row["deploys"] == 10]
    result = {
        "scope": "Read-only existing source/archives and exact Result events; no numerical solve",
        "GPU_calls": 0,
        "retained_Result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "actual_fleet_deploy_histogram": dict(sorted(Counter(len(r["deploys"]) for r in ships.values()).items())),
        "actual_fleet_collect_histogram": dict(sorted(Counter(len(r["collects"]) for r in ships.values()).items())),
        "actual_fleet_deployed_miners": sum(len(r["deploys"]) for r in ships.values()),
        "actual_fleet_collected_miners": sum(len(r["collects"]) for r in ships.values()),
        "actual_fleet_raw_kg": total_raw,
        "ten_deployer_records": len(ten),
        "ten_deployer_collect_histogram": dict(sorted(Counter(r["collects"] for r in ten).items())),
        "ten_deployer_closed_records": sum(r["closed_independent"] for r in ten),
        "source_findings": [
            {"file": "search.py", "lines": [58, 196, 1217],
             "finding": "Default generation and collect-DP cap are10; inclusive depth loop attempts through10"},
            {"file": "bundles.py", "lines": [226, 275, 657, 680],
             "finding": "Cluster depth cap propagates10; refinement deduplicates solely by Earth leg then keeps top2"},
            {"file": "bundles.py", "lines": [211, 213],
             "finding": "Joint asteroid insertion is disabled by default after earlier unsuccessful trial"},
            {"file": "search.py", "lines": [1796, 1806, 1809, 1812, 1822],
             "finding": "Beam prunes estimated reserve/return feasibility and diversity before deeper completion"},
            {"file": "cooperative.py", "lines": [950, 971, 1049],
             "finding": "Master limits fleet ship count, asteroid conflicts, dependency and raw ship rule; no per-column8-miner cap"},
        ],
        "conclusion": (
            "Nine-miner closed routes are already present. Ten-deployer routes exist but no certified "
            "independently closed10-miner route was found in the inspected published pool. Raising max_deploys "
            "from8to10 is not the fix: defaults and recorded family campaigns already use10."
        ),
        "recommendation": (
            "Next route-family expansion should retain depth diversity through the finite refinement shortlist: "
            "pair certified Earth legs with separate9/10-miner deploy-and-collect chains, carrying the complete "
            "collection/return cost into ranking. The existing one-chain-per-Earth-leg rule can exclude every "
            "larger alternative behind a higher-ranked shorter chain. Compare a bounded depth-stratified arm "
            "with current selection using identical generated candidates; rank true fixed-bonus contribution "
            "and preserve raw ship eligibility and both final fleet checkers. Do not simply repeat insertion "
            "or increase beam/deploy budgets without collecting depth/prune/refinement attrition."
        ),
        "limits": (
            "Archive scope excludes packed-only/newer outputs outside results/gtoc12/runs. Per-route certified "
            "flags and cargo are archive evidence, not a new physical or complete-fleet certification. "
            "Source shows a possible selection bottleneck; existing reports do not prove a specific rejected "
            "10-miner alternative would certify. No GPU experiment prepared or launched."
        ),
        "actual_fleet_ships": list(ships.values()),
    }
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "actual_fleet_ships"}, indent=2))


if __name__ == "__main__":
    main()
