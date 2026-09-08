"""Read-only CPU audit of the completed v607b return-window rescue."""

import ctypes
import datetime
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
KIT = ROOT.parent / "return-window-rescue-v607b"
OUT = KIT / "output"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def policy(rows, count):
    candidates = sorted(
        (
            r
            for r in rows
            if r["geometry_feasible"]
            and all(
                r[k] is not None and math.isfinite(r[k])
                for k in ("predicted_propellant_kg", "authority_ratio", "departure", "arrival")
            )
        ),
        key=lambda r: (
            r["predicted_propellant_kg"],
            r["authority_ratio"],
            -r["tof_days"],
            r["departure"],
            r["arrival"],
        ),
    )
    selected, used = [], set()
    for diverse in (True, False):
        for row in candidates:
            if len(selected) == count:
                return selected
            pair = row["departure"], row["arrival"]
            if (
                pair in used
                or (diverse
                and any(
                    max(abs(row["departure"] - r["departure"]), abs(row["arrival"] - r["arrival"]))
                    < 20
                    for r in selected
                ))
            ):
                continue
            selected.append(row)
            used.add(pair)
    return selected


def main(destination=ROOT / "report.json"):
    report = read(OUT / "report.json")
    require(report.get("complete"), "Run is incomplete; audit does not poll or launch it")
    require(destination.parent.resolve() == ROOT.resolve(), "Audit output must remain in audit folder")
    require(not destination.exists(), "Audit output must be new")
    ready = read(KIT / "ready-manifest.json")
    require(
        sha(KIT / "ready-manifest.json")
        == "9414730e31bc2d896285761dc6436277cefb7d92af8a297291d9e3116a2c28ec",
        "Ready pin",
    )
    for name, expected in ready["files"].items():
        require(sha(KIT / name) == expected, f"Frozen kit changed: {name}")
    launch = read(KIT / "launch.json")
    require(
        launch["pid"] == report["pid"] == 1053 and launch["supervisor_pid"] == 1048, "PID identity"
    )
    require(launch.get("compute_process_observation") == "", "GPU prelaunch observation")
    require(launch["driver_sha256"] == sha(KIT / "run.py"), "Driver launch hash")
    require(launch["supervisor_sha256"] == sha(KIT / "launch.py"), "Supervisor launch hash")
    require(
        report["whole_route_reruns"] == 0
        and not report["cargo_scaling_allowed"]
        and not report["prefix_retiming_allowed"],
        "Execution scope",
    )
    native = [json.loads(line) for line in (OUT / "native-returns.jsonl").read_text().splitlines()]
    require(
        len(native) <= 5
        and len(native) == report["native_returns_started"] == report["native_returns_completed"],
        "Native call budget/accounting",
    )
    require([r["number"] for r in native] == list(range(1, len(native) + 1)), "Native sequence")
    require(len(report["cases"]) <= 5, "Case budget")
    require(native[0]["case"] == "control_return", "Positive control ordering")
    bycase = Counter(r["case"] for r in native)
    require(all(n == 1 for n in bycase.values()), "No return was retried")
    frozen = read(KIT / "preparation.json")
    require(report["scvx_settings"] == frozen["scvx_settings"], "Unchanged SCvx settings")
    base = report["baseline"]
    require(base["ok"] and base["official"]["ok"] and base["independent"]["ok"], "Baseline gates")
    cases, allsolver, emitted_fleets = [], [], []
    sys.path.insert(0, str(KIT / "source/src"))
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    from spacepdhcg.gtoc12.screening import (
        propellant_for_delta_v,
        return_inflation_model,
        thrust_authority_km_s,
    )
    from spacepdhcg.gtoc12.solution import Event, ShipTrajectory, Solution, format_solution

    baseline_solution = Solution.read(KIT / "inputs/Result.txt")
    original_control = Solution.read(KIT / "inputs/control/Result.txt").ships[0]

    def section(ship, prefix=False):
        items = [
            item
            for item in ship.items
            if not prefix or (item.epoch if isinstance(item, Event) else item.end) <= 69218
        ]
        return format_solution(Solution([ShipTrajectory(1, items)]))

    for row in report["cases"]:
        directory = OUT / "cases" / row["id"]
        outcome = read(directory / "outcome.json")
        returned = read(directory / "return.json")
        origin = "control" if row["kind"] == "control" else "probe"
        prescription = read(KIT / "inputs" / origin / "prescription.json")["plan"]
        cargo = prescription["collected_mass_kg"]
        after = (
            read(KIT / "inputs" / origin / "leg-15.json")["certificate"]["final_mass_kg"]
            + cargo["59653"]
        )
        minimum = 500 + sum(cargo.values())
        require(row["cargo_kg"] == cargo, "Immutable cargo")
        require(
            row["prefix_fingerprint"] == read(OUT / "prefix" / (origin + ".json"))["fingerprint"],
            "Prefix fingerprint",
        )
        require(row["native_return_calls"] == bycase[row["id"]] == 1, "One native return per case")
        require(
            outcome["initial_mass_kg"] == after and outcome["minimum_final_mass_kg"] == minimum,
            "Fixed mass budget",
        )
        require(
            (outcome["departure"], outcome["arrival"])
            == (row["window"]["departure"], row["window"]["arrival"]),
            "Exact prescribed return epochs",
        )
        require(69218 <= outcome["departure"] < outcome["arrival"] <= 69807, "Legal return window")
        require(
            sha(directory / "return-solution.npz") == returned["solution_arrays"]["sha256"],
            "Retained return arrays",
        )
        with np.load(directory / "return-solution.npz") as arrays:
            require(
                all(np.all(np.isfinite(arrays[name])) for name in arrays.files),
                "Finite retained arrays",
            )
        solved = outcome["solution"]
        allsolver.extend(solved.get("solver_reports", []))
        entry = {
            "id": row["id"],
            "status": row["status"],
            "departure": outcome["departure"],
            "arrival": outcome["arrival"],
            "fixed_initial_mass_kg": after,
            "fixed_minimum_final_mass_kg": minimum,
            "fixed_cargo_kg": sum(cargo.values()),
            "available_propellant_kg": after - minimum,
            "reported_propellant_kg": solved["propellant_kg"],
            "reported_max_defect": solved["max_defect"],
            "native_status": solved["status"],
            "SCvx_iterations": solved["iterations"],
            "SCvx_accepted_iterations": solved["accepted_iterations"],
            "certified": outcome["certified"],
            "diagnostic": outcome["diagnostic"],
            "refinement_seconds": row["refinement_seconds"],
            "arrays_sha256": sha(directory / "return-solution.npz"),
        }
        checked_path = directory / "verification.json"
        if not outcome["certified"]:
            require(not checked_path.exists(), "Uncertified return cannot become a checked fleet")
            require(
                not any(p["source"] == row["id"] for p in report["promotions"]),
                "Failed return promotion",
            )
        if checked_path.exists():
            checked = read(checked_path)
            require(checked == row["verification"], "Checker report identity")
            require(
                checked["ok"]
                and checked["official"]["ok"]
                and checked["independent"]["ok"]
                and checked["prescribed_cargo_verified"],
                "Both fullfleet gates and cargo",
            )
            fleet = Solution.read(directory / "fleet-Result.txt")
            require(fleet.ship_count == 23, "Fleet count")
            for ship, retained in zip(fleet.ships, baseline_solution.ships, strict=True):
                if ship.ship_id != 7:
                    require(section(ship) == section(retained), "Other ship changed")
            ship7 = fleet.ships[6]
            visits = ship7.asteroid_visits()
            expected_events = sorted(
                (int(k), value, -40.0) for k, value in prescription["deploy_epochs"].items()
            )
            expected_events += sorted(
                (int(k), value, cargo[k]) for k, value in prescription["collect_epochs"].items()
            )
            actual = sorted((e.event_id, e.epoch, e.after.mass - e.before.mass) for e in visits)
            require(len(actual) == len(expected_events) == 16, "No additional asteroid visits")
            for item, expected in zip(actual, sorted(expected_events), strict=True):
                require(
                    item[:2] == expected[:2] and abs(item[2] - expected[2]) < 1e-8,
                    "Exact event schedule/cargo",
                )
            if origin == "control":
                require(
                    section(ship7, True) == section(original_control, True),
                    "Control prefix exact emitted bytes",
                )
            entry["fullfleet"] = checked
            emitted_fleets.append(str(directory / "fleet-Result.txt"))
        cases.append(entry)
    require(report["control_passed"] == cases[0]["certified"], "Control certification agreement")
    screening_audit = None
    if "screening" in report:
        require(report["control_passed"], "No screening after failed control")
        rows, pairs = [], []
        mass, minimum = 1207.2044215842654, 1115.6605065023955
        for stage in ("coarse", "fine"):
            metadata = read(OUT / "screening" / (stage + ".json"))
            raw = OUT / "screening" / (stage + ".npz")
            require(sha(raw) == metadata["raw_sha256"], "Screening raw hash")
            with np.load(raw) as values:
                dv = values["departure_delta_v"] + values["arrival_delta_v"]
                ratio = dv / np.maximum(thrust_authority_km_s(mass, values["tof_days"], 1), 1e-12)
                inflation = return_inflation_model(values["tof_days"], ratio)
                propellant = propellant_for_delta_v(mass, inflation * dv)
                for name, expected in (
                    ("authority_ratio", ratio),
                    ("inflation", inflation),
                    ("predicted_propellant_kg", propellant),
                ):
                    require(
                        np.array_equal(values[name], expected, equal_nan=True),
                        "Independent ranking recomputation: " + name,
                    )
                for i, row in enumerate(metadata["rows"]):
                    pair = row["departure"], row["arrival"]
                    require(69218 <= pair[0] < pair[1] <= 69807, "Screened epoch domain")
                    require(
                        pair == (values["departure"][i], values["arrival"][i]),
                        "Raw row epoch identity",
                    )
                    if row["geometry_feasible"]:
                        require(
                            row["predicted_propellant_kg"] == propellant[i],
                            "JSON/raw ranking identity",
                        )
                        require(
                            row["predicted_reserve_kg"] == mass - minimum - propellant[i],
                            "Predicted reserve",
                        )
                    pairs.append(pair)
            rows.extend(metadata["rows"])
        screening = report["screening"]
        require(
            len(pairs) == len(set(pairs)) == screening["total_rows"] <= 9614,
            "Screen count/deduplication",
        )
        require(screening["coarse_rows"] == 7022 and screening["fine_rows"] <= 2592, "Grid budget")
        require(screening["logical_branch_requests"] == 2 * len(pairs) <= 19228, "Direction budget")
        require(policy(rows, 4) == screening["selected"], "Deterministic complete-domain selection")
        require(
            [c["window"] for c in report["cases"][1:]]
            == screening["selected"][: len(report["cases"]) - 1],
            "Selected-to-refined identity",
        )
        distances = [
            max(abs(a["departure"] - b["departure"]), abs(a["arrival"] - b["arrival"]))
            for i, a in enumerate(screening["selected"])
            for b in screening["selected"][i + 1 :]
        ]
        screening_audit = {
            "coarse_rows": screening["coarse_rows"],
            "fine_rows": screening["fine_rows"],
            "total_pairs": len(pairs),
            "direction_requests": 2 * len(pairs),
            "deterministic_ranking_recomputed": True,
            "minimum_selected_basin_separation_days": min(distances),
            "selected": screening["selected"],
            "all_selected_proxy_reserves_negative": all(
                r["predicted_reserve_kg"] < 0 for r in screening["selected"]
            ),
        }
    else:
        require(len(cases) == 1, "Failed control suppressed all candidates")
    for promotion in report["promotions"]:
        require(
            promotion["ok"] and promotion["official"]["ok"] and promotion["independent"]["ok"],
            "Promotion checkers",
        )
        require(
            promotion["score_kg"] > base["score_kg"]
            and promotion["total_mass_kg"] >= 23 * math.log(23 / 2) / 0.004,
            "Promotion objective/fleet rule",
        )
        require(
            sha(Path(promotion["solution"])) == promotion["solution_sha256"], "Promoted Result hash"
        )
    if not report["promotions"]:
        require(
            report["best"]["solution_sha256"] == sha(KIT / "inputs/Result.txt")
            and report["best"]["score_kg"] == base["score_kg"],
            "Incumbent retained",
        )
    files = {
        str(path.relative_to(KIT)): sha(path) for path in sorted(OUT.rglob("*")) if path.is_file()
    }
    result = {
        "audit_ok": True,
        "audit_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "GPU_calls": 0,
        "scope": (
            "Read-only artifact/accounting audit and CPU arithmetic replay; "
            "no new solve or fleet verifier invocation"
        ),
        "audit_source_sha256": sha(Path(__file__)),
        "reviewed_kit_files": len(ready["files"]),
        "ready_sha256": sha(KIT / "ready-manifest.json"),
        "output_report_sha256": sha(OUT / "report.json"),
        "output_files": files,
        "status": report["status"],
        "control_passed": report["control_passed"],
        "native_returns": len(native),
        "whole_route_reruns": 0,
        "cases": cases,
        "screening": screening_audit,
        "SCvx_iterations": sum(r["iterations"] for r in native),
        "SCvx_accepted_iterations": sum(r["accepted_iterations"] for r in native),
        "native_wrapper_wall_seconds": sum(r["wrapper_wall_seconds"] for r in native),
        "total_wall_seconds": report["seconds"],
        "fleet_verification_wall_seconds": base["verification_seconds"]
        + sum(c.get("fullfleet", {}).get("verification_seconds", 0) for c in cases),
        "qoco_rows": len(allsolver),
        "qoco_inner_iterations": sum(r["iterations"] for r in allsolver),
        "qoco_api_status_counts": dict(Counter(str(r["api_status"]) for r in allsolver)),
        "qoco_qualified_counts": dict(Counter(str(r["qualified"]) for r in allsolver)),
        "promotions": report["promotions"],
        "retained_raw_kg": report["best"]["total_mass_kg"],
        "retained_weighted_kg": report["best"]["score_kg"],
        "emitted_fullfleets": emitted_fleets,
        "interpretation_limit": (
            "Failed SCvx returns and proxy estimates do not prove physical infeasibility. "
            "Timings include host overhead; solver rows are not independent whole-route solutions."
        ),
    }
    result["prefix_extra_certified_propellant_kg"] = sum(
        read(KIT / "inputs/probe" / f"leg-{i:02d}.json")["solution"]["propellant_kg"]
        - read(KIT / "inputs/control" / f"leg-{i:02d}.json")["solution"]["propellant_kg"]
        for i in range(16)
    )
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "audit_ok",
                    "status",
                    "native_returns",
                    "retained_raw_kg",
                    "retained_weighted_kg",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch.object(
        ctypes, "CDLL", side_effect=AssertionError("Audit forbids native library loading")
    ):
        main(ROOT / sys.argv[1] if len(sys.argv) > 1 else ROOT / "report.json")
