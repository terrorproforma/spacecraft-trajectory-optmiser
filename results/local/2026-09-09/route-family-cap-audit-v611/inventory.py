"""Read retained route summaries and frozen source settings; no solver or GPU calls."""

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (ROOT / "report.json").exists():
        raise FileExistsError("Preserve existing audit")
    # rg is unavailable inside this WSL runtime; inspect only the named archive subtree.
    paths = sorted(str(path.relative_to(REPO)) for path in
                   (REPO / "results/gtoc12/runs").rglob("route_summary.json"))
    rows = []
    for relative in paths:
        path = REPO / relative
        entry = read(path)
        plan = entry.get("plan", entry)
        deploy, collect = plan.get("deploy_epochs", {}), plan.get("collect_epochs", {})
        if not deploy or not collect:
            continue
        closed = not plan.get("foreign_deploy_epochs") and set(deploy) == set(collect)
        cargo = entry.get("collected_mass_kg", plan.get("collected_mass_kg", {}))
        rows.append({
            "path": relative, "sha256": sha(path), "certified": bool(entry.get("certified")),
            "deploys": len(deploy), "collects": len(collect), "closed_independent": closed,
            "raw_kg": sum(cargo.values()),
            "plan_identity": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        })
    closed = [row for row in rows if row["closed_independent"] and row["certified"]]
    unique = {row["plan_identity"]: row for row in closed}
    incumbents = []
    for ship in range(1, 24):
        path = REPO / "build/performance/incident-window-substitution-v604/inputs" / f"ship-{ship:02d}.json"
        route = read(path)
        plan = route["plan"]
        incumbents.append({
            "ship": ship, "sha256": sha(path), "deploys": len(plan["deploy_epochs"]),
            "collects": len(plan["collect_epochs"]),
            "foreign_collects": len(plan.get("foreign_deploy_epochs", {})),
            "raw_kg": sum(route["collected_mass_kg"].values()),
        })
    hist = Counter(row["deploys"] for row in closed)
    summary = {
        "scope": "Existing results/gtoc12/runs route_summary.json files only; duplicates retained separately",
        "GPU_calls": 0, "route_files_found": len(paths), "route_files_with_event_tables": len(rows),
        "certified_closed_records": len(closed), "certified_closed_histogram": dict(sorted(hist.items())),
        "unique_certified_closed_plans": len(unique),
        "unique_certified_closed_histogram": dict(sorted(Counter(row["deploys"] for row in unique.values()).items())),
        "best_raw_closed_per_deploy_count": {
            str(count): max((row for row in closed if row["deploys"] == count), key=lambda row: row["raw_kg"])
            for count in sorted(hist)
        },
        "incumbent_ships": incumbents,
        "incumbent_deploy_histogram": dict(sorted(Counter(row["deploys"] for row in incumbents).items())),
        "incumbent_collect_histogram": dict(sorted(Counter(row["collects"] for row in incumbents).items())),
        "incumbent_total_deploys": sum(row["deploys"] for row in incumbents),
        "incumbent_total_collects": sum(row["collects"] for row in incumbents),
        "raw_mass_scope": "Archive cargo summaries, not a new complete-fleet certificate or weighted score",
        "all_route_records": rows,
    }
    (ROOT / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("all_route_records", "incumbent_ships")}, indent=2))


if __name__ == "__main__":
    main()
