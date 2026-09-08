"""Read-only local archive coverage audit; no optimizer, verifier, or GPU calls."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("v597_driver", HERE / "run.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
summaries = {i: json.loads((HERE / "inputs" / f"ship-{i:02d}.json").read_text()) for i in range(1, 24)}
cases = driver.make_cases(summaries)
signatures = {tuple(c["collect_order"]): c["case"] for c in cases}
selected = summaries[15]
target = set(selected["plan"]["deploy_epochs"])
paths = sorted((REPO / "results").rglob("route_summary.json"))
paths += sorted((REPO / "build/performance/orphan-recovery-v595/output-compatible").rglob("route_summary.json"))
report = {"scope": "locally unpacked route_summary.json files in results, plus the completed v595 route outputs",
          "files_examined": 0, "certified_routes": 0, "same_deploy_set": [],
          "exact_collect_order_matches": [], "unreadable": [],
          "candidate_orders": len(cases), "input_sha256": json.loads((HERE / "inputs/sha256.json").read_text()),
          "concurrent_scripts": {}, "limitations": [
              "This audit does not enumerate hidden or transient candidates inside earlier GPU or DP searches.",
              "Earlier Held-Karp collection pricing allowed order permutations under its then-current deploy epochs, model, and grid.",
              "The proposed search is new post-certification work around the promoted nine-collect v595 schedule; no claim all permutations were never implicitly considered.",
              "Packed archives not also unpacked locally are not inspected."]}
for path in paths:
    report["files_examined"] += 1
    try:
        route = json.loads(path.read_text())
        if not route.get("certified") or "plan" not in route:
            continue
        report["certified_routes"] += 1
        plan = route["plan"]
        if set(plan["deploy_epochs"]) != target:
            continue
        order = tuple(int(a) for a in sorted(plan["collect_epochs"], key=plan["collect_epochs"].get))
        item = {"path": path.relative_to(REPO).as_posix(), "raw_kg": route["total_collected_kg"],
                "collect_order": order, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        report["same_deploy_set"].append(item)
        if order in signatures:
            report["exact_collect_order_matches"].append(item | {"case": signatures[order]})
    except (OSError, ValueError, KeyError) as error:
        report["unreadable"].append({"path": path.relative_to(REPO).as_posix(), "error": repr(error)})
for name in ("run_joint_campaign_v636.py", "prepare_joint_campaign_v637.py", "retry_joint_campaign_v639.py"):
    path = REPO / "build/performance" / name
    report["concurrent_scripts"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "replay old v595 own-orphan campaign to compare native joint device selection; no new-incumbent interior order mutation"}
(HERE / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k in ("files_examined", "certified_routes", "candidate_orders")}
                 | {"same_deploy_set_routes": len(report["same_deploy_set"]),
                    "matching_candidate_orders": len(report["exact_collect_order_matches"]),
                    "unreadable_files": len(report["unreadable"])}, indent=2))
