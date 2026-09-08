"""Audit immutable truth-set evidence; no GPU/library loading or result mutation."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / "truth-set-v606"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ready = read(KIT / "ready-manifest.json")
assert (
    sha(KIT / "ready-manifest.json")
    == "7e87c5360efaee14ddaec94602ac4132b610073d782118d6e4cb122312c4d2cc"
)
for name, expected in ready["files"].items():
    assert sha(KIT / name) == expected, name
report = read(KIT / "output/report.json")
launch = read(KIT / "launch.json")
assert report["pid"] == launch["pid"] == 377
assert launch["supervisor_pid"] == 298
assert report["profile"] == launch["profile"] == "local"
assert (
    report["native_libraries"]
    == read(KIT / "profiles.json")["profiles"]["local"]["native_libraries"]
)
assert report["scvx_settings"] == read(KIT / "preparation.json")["scvx_settings"]
assert report["driver_sha256"] == sha(KIT / "run.py")
assert report["fixed_wrapper_sha256"] == sha(KIT / "fixed_refine.py")
assert launch["supervisor_sha256"] == sha(KIT / "launch.py")
assert report["input_sha256"] == sha(KIT / "inputs/Result.txt")
assert not report["cargo_resizing_allowed"] and not report["retiming_allowed"]
assert not report["physics_tolerances_changed"]
native_path = KIT / "output/native-solves.jsonl"
native = (
    []
    if not native_path.exists()
    else [json.loads(line) for line in native_path.read_text().splitlines()]
)
result = {
    "complete": report["complete"],
    "status": report["status"],
    "stage": report["stage"],
    "audited_prepared_files": len(ready["files"]),
    "native_calls": len(native),
    "whole_route_refinements": report["whole_route_refinements_started"],
    "promotions": len(report["promotions"]),
    "cases": [],
    "native_wrapper_wall_seconds": sum(row["seconds"] for row in native),
    "native_iterations": sum(row.get("iterations", 0) for row in native),
    "native_accepted_iterations": sum(row.get("accepted_iterations", 0) for row in native),
    "native_status_counts": dict(Counter(row.get("status", "exception") for row in native)),
}
plans = {case["id"]: case for case in read(KIT / "plan/plan.json")["cases"]}
counts = Counter(row["case"] for row in native)
controls = {}
for record in report["cases"]:
    prescribed = plans[record["id"]]
    assert record["prescription"] == prescribed
    directory = KIT / "output/cases" / record["id"]
    summary = {
        "id": record["id"],
        "ship": record["ship"],
        "kind": record["kind"],
        "status": record["status"],
        "native_calls": counts[record["id"]],
        "proxy_interpretation_allowed": record.get("proxy_interpretation_allowed", False),
    }
    if record["kind"] == "control":
        controls[record["id"]] = record["status"] == "control_passed"
        if controls[record["id"]]:
            checked = record["verification"]
            assert checked == read(directory / "verification.json")
            assert checked["ok"] and checked["independent"]["ok"] and checked["official"]["ok"]
            assert abs(checked["score_kg"] - report["baseline"]["score_kg"]) <= 1e-7
            assert abs(checked["total_mass_kg"] - report["baseline"]["total_mass_kg"]) <= 1e-7
    elif record["status"] != "pending":
        paired = controls.get(prescribed["requires_control"], False)
        if not paired:
            assert record["status"] == "skipped_corresponding_control_failed"
            assert counts[record["id"]] == 0 and not directory.exists()
            assert not record["proxy_interpretation_allowed"]
    refined_path = directory / "refinement.json"
    if refined_path.exists():
        refined = read(refined_path)
        assert refined["plan"] == prescribed["plan"]
        assert refined["collected_mass_kg"] == prescribed["prescribed_cargo_kg"]
        assert refined["passes"] == 1
        assert len(refined["legs"]) <= prescribed["maximum_leg_calls"]
        details = [read(path) for path in sorted(directory.glob("leg-*.json"))]
        summary.update(
            requested_legs=prescribed["maximum_leg_calls"],
            leg_certified=sum(row["certified"] for row in details),
            failure_details=[
                {
                    key: value
                    for key, value in failure.items()
                    if key not in ("solution", "solution_arrays")
                }
                for failure in refined["failures"]
            ],
            route_certified=refined["certified"],
            prescribed_raw_kg=sum(refined["collected_mass_kg"].values()),
            unattempted_leg_indices=record.get("unattempted_leg_indices", []),
            leg_statuses=[
                {
                    "leg": row["leg"],
                    "status": row["status"],
                    "certified": row["certified"],
                    "diagnostic": row["diagnostic"],
                    "certification": row["certification"],
                    "solution_status": None
                    if row["solution"] is None
                    else row["solution"]["status"],
                }
                for row in details
            ],
            certified_prefix_propellant_kg=sum(
                row["initial_mass_kg"] - row["certificate"]["final_mass_kg"]
                for row in details[:-1]
                if row["certified"]
            ),
        )
        if details and len(details) == prescribed["maximum_leg_calls"]:
            final = details[-1]
            summary["last_leg"] = {
                key: final[key]
                for key in (
                    "from",
                    "to",
                    "departure",
                    "arrival",
                    "initial_mass_kg",
                    "minimum_final_mass_kg",
                    "status",
                    "diagnostic",
                )
            }
            summary["last_leg"]["available_propellant_kg"] = (
                final["initial_mass_kg"] - final["minimum_final_mass_kg"]
            )
            if final["certified"]:
                summary["last_leg"]["certified_propellant_kg"] = (
                    final["initial_mass_kg"] - final["certificate"]["final_mass_kg"]
                )
            else:
                summary["last_leg"]["uncertified_solution_residual"] = final["solution"][
                    "max_defect"
                ]
        for row in details:
            if "solution_arrays" in row:
                path = directory / Path(row["solution_arrays"]["path"]).name
                assert sha(path) == row["solution_arrays"]["sha256"]
    result["cases"].append(summary)
if report["complete"]:
    assert len(native) == report["native_solves_started"] == report["native_solves_completed"] <= 66
    assert [row["number"] for row in native] == list(range(1, len(native) + 1))
    assert report["whole_route_refinements_started"] <= 4
    assert len(report["cases"]) == 4
    assert (
        report["baseline"]["ok"]
        and report["baseline"]["independent"]["ok"]
        and report["baseline"]["official"]["ok"]
    )
    for best in report["promotions"]:
        assert best["ok"] and best["independent"]["ok"] and best["official"]["ok"]
        assert best["score_kg"] > report["baseline"]["score_kg"]
        assert best["total_mass_kg"] >= 23 * math.log(23 / 2) / 0.004 - 1e-8
        case = plans[best["source"]]
        assert controls[case["requires_control"]]
        assert (
            sha(KIT / "output/best" / Path(best["solution"]).parent.name / "Result.txt")
            == best["solution_sha256"]
        )
    if not report["promotions"]:
        assert report["best"]["source"] == "retained_v595"
        assert report["best"]["solution_sha256"] == sha(KIT / "inputs/Result.txt")
    result.update(
        wall_seconds=report["seconds"],
        baseline_verification_seconds=report["baseline"]["verification_seconds"],
        retained_raw_kg=report["best"]["total_mass_kg"],
        retained_weighted_kg=report["best"]["score_kg"],
        returncode=launch.get("returncode"),
        report_sha256=sha(KIT / "output/report.json"),
        launch_sha256=sha(KIT / "launch.json"),
        auditor_sha256=sha(Path(__file__)),
        all_full_fleet_verification_seconds=report["baseline"]["verification_seconds"]
        + sum(
            case["verification"]["verification_seconds"]
            for case in report["cases"]
            if "verification" in case
        ),
        all_route_refinement_seconds=sum(case["refinement_seconds"] for case in report["cases"]),
    )
    qoco_reports = [
        item
        for path in (KIT / "output/cases").glob("*/leg-*.json")
        for details in [read(path)]
        if details["solution"] is not None
        for item in details["solution"]["solver_reports"]
    ]
    result["qoco"] = {
        "reported_subproblem_rows": len(qoco_reports),
        "reported_inner_iterations": sum(item["iterations"] for item in qoco_reports),
        "qualified_rows": sum(bool(item["qualified"]) for item in qoco_reports),
        "status_counts": dict(Counter(item["qoco_status"] for item in qoco_reports)),
        "timing_scope": (
            "native wrapper wall time includes host marshalling; not isolated CUDA kernel time"
        ),
    }
    cases_by_id = {case["id"]: case for case in result["cases"]}
    result["matched_pairs"] = []
    for case in result["cases"]:
        if case["kind"] != "probe":
            continue
        control = cases_by_id[plans[case["id"]]["requires_control"]]
        result["matched_pairs"].append(
            {
                "probe": case["id"],
                "control_passed": controls[control["id"]],
                "extra_certified_prefix_propellant_kg": case["certified_prefix_propellant_kg"]
                - control["certified_prefix_propellant_kg"],
                "same_return_geometry": all(
                    case["last_leg"][key] == control["last_leg"][key]
                    for key in ("from", "to", "departure", "arrival")
                ),
                "probe_return_available_propellant_kg": case["last_leg"]["available_propellant_kg"],
                "control_certified_return_propellant_kg": control["last_leg"][
                    "certified_propellant_kg"
                ],
                "interpretation": (
                    "All prior legs certified; changed earlier route consumed more fuel; "
                    "same scheduled return did not converge/certify. "
                    "This is not a proof of physical infeasibility or "
                    "a measured complete-route propellant value."
                ),
            }
        )
    (ROOT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            **{key: value for key, value in result.items() if key != "cases"},
            "cases": [
                {key: value for key, value in case.items() if key != "leg_statuses"}
                for case in result["cases"]
            ],
        },
        indent=2,
    )
)
