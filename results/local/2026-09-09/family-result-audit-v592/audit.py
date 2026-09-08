"""Read-only, standard-library audit of richer-family outputs and retained routes."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def signature(route):
    plan = route["plan"]
    fields = ("launch_epoch", "earth_return_epoch", "deploy_epochs", "collect_epochs", "foreign_deploy_epochs", "orphaned")
    return json.dumps({key: plan.get(key) for key in fields}, sort_keys=True)


def footprint(route):
    p = route["plan"]
    return {int(a) for key in ("deploy_epochs", "collect_epochs") for a in p[key]}


def brief(path, route):
    p = route["plan"]
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "raw_kg": route["total_collected_kg"],
        "final_mass_kg": route["final_mass_kg"],
        "collects": sorted(map(int, p["collect_epochs"])),
        "deploys": sorted(map(int, p["deploy_epochs"])),
        "foreign": sorted(map(int, p.get("foreign_deploy_epochs", {}))),
        "orphaned": p.get("orphaned", []),
        "signature_sha256": hashlib.sha256(signature(route).encode()).hexdigest(),
    }


inc_dir = ROOT / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
incumbents = [(p, json.loads(p.read_text())) for p in sorted(inc_dir.glob("ship-*.json"))]
archives = []
for path in (ROOT / "results").rglob("route_summary.json"):
    if "2026-09-09" in path.parts:
        continue
    try:
        r = json.loads(path.read_text())
        if r.get("certified") and "plan" in r:
            archives.append((path, r))
    except (OSError, ValueError):
        pass

report = {"historical_route_files": len(archives), "incumbent": load("results/lambda/2026-09-06/fleet_master_v11/run_report.json")["best"], "pilots": []}
for name, directory in (
    ("local_v588", "build/performance/family-gpu-v588/output"),
    ("h100_v589", "results/lambda/2026-09-09/family-gpu-v589/output"),
):
    run = load(directory + "/run_report.json")
    pilot = {"name": name, "best": run["best"], "wall_seconds_total": run["wall_seconds_total"], "columns": run["master"]["columns"], "routes": []}
    for path in sorted((ROOT / directory).rglob("route_summary.json")):
        route = json.loads(path.read_text())
        p = route["plan"]
        current = brief(path, route)
        current["identical_schedule_history"] = [brief(q, other) for q, other in archives if signature(route) == signature(other)]
        current["same_collect_set_history"] = sorted((brief(q, other) for q, other in archives if set(p["collect_epochs"]) == set(other["plan"]["collect_epochs"])), key=lambda x: -x["raw_kg"])[:12]
        current["incumbent_conflicts"] = [brief(q, other) | {"overlap": sorted(footprint(route) & footprint(other))} for q, other in incumbents if footprint(route) & footprint(other)]
        pilot["routes"].append(current)
    report["pilots"].append(pilot)

hist = load("results/lambda-h100/gtoc12/cluster_fleet_v10_control/run_report.json")
report["historical_family32"] = next(b for b in hist["bundles"] if b["label"] == 32)
(OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
for pilot in report["pilots"]:
    print(pilot["name"], pilot["wall_seconds_total"], pilot["best"]["score_kg"], pilot["best"]["fleet"]["collected_kg_per_ship"])
    for route in pilot["routes"]:
        print("  ", route["raw_kg"], "identical_history", len(route["identical_schedule_history"]), "same_collect_history", [(r["raw_kg"], r["path"]) for r in route["same_collect_set_history"][:3]], "incumbent_conflicts", [(r["path"].split("/")[-1], r["raw_kg"], r["overlap"]) for r in route["incumbent_conflicts"]])
