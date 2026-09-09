"""Independent saved-output accounting and residual arithmetic; no project imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KIT = ROOT / "build/performance/fleet-budgeted-prefix-v630"
OUT = KIT / "output"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def request(plan):
    value = {"flights": [(int(x["from"]), int(x["to"]), float(x["t0"]), float(x["tf"])) for x in plan["legs"] if x["role"] != "camp"],
             "deploy": sorted((int(k), float(v)) for k, v in plan["deploy_epochs"].items()),
             "collect": sorted((int(k), float(v)) for k, v in plan["collect_epochs"].items()),
             "cargo": sorted((int(k), float(v)) for k, v in plan["collected_mass_kg"].items())}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError("Assertions required for evidence audit")
    ready = read(KIT / "ready.json")
    assert sha(KIT / "ready.json") == "a145167e71d52a709bd80bf3af87458fd824c13c14484e702eb326f27ae60643"
    for name, expected in ready["files"].items():
        assert sha(KIT / name) == expected, name
    for name, expected in ready["external_files"].items():
        assert sha(KIT / name) == expected, name
    report, launch, protocol = read(OUT / "report.json"), read(OUT / "launch-report.json"), read(KIT / "protocol.json")
    assert report["complete"] and launch["complete"] and launch["passed"] and launch["exit_code"] == 0
    assert launch["owned_descendant_cleanup"]["verified_empty"] and not launch["compute_processes_after_cleanup"]
    assert report["status"] == "no_new_certified_routes_incumbent_retained"
    assert report["full_fleet_CPU_checks"] == report["full_fleet_official_checks"] == report["fleet_selection_calls"] == report["incumbent_promotions"] == 0
    assert not list(OUT.rglob("Result.txt")) and not list(OUT.rglob("ship.txt"))
    # These exact constants are read from the frozen source; no ephemeris or dynamics evaluation.
    constant_source = (KIT / "host/spacepdhcg/gtoc12/constants.py").read_text()
    assert "AU_KM: Final = 1.49597870691e8" in constant_source
    assert "MU_SUN_KM3_S2: Final = 1.32712440018e11" in constant_source
    DU = 1.49597870691e8
    VU = DU / math.sqrt(DU**3 / 1.32712440018e11)
    rows, total = [], Counter()
    for expected in protocol["candidates"]:
        directory = OUT / expected["id"]
        original = read(KIT / "inputs" / expected["input_file"])
        prescribed = read(directory / "prescribed-plan.json")
        route = read(directory / "route-result.json")
        case = read(directory / "case-report.json")
        assert request(original) == request(prescribed) == request(route["plan"]) == expected["request_sha256"]
        assert {int(k): v for k, v in route["collected_mass_kg"].items()} == {int(k): v for k, v in original["collected_mass_kg"].items()}
        flown = [x for x in original["legs"] if x["role"] != "camp"]
        legs = sorted((directory / "legs").iterdir())
        assert [int(x.name) for x in legs] == list(range(case["native_solves_started"]))
        assert len(legs) == case["native_solves_returned"]
        case_iterations, certificates = 0, 0
        prior_mass = None
        entries = []
        for index, path in enumerate(legs):
            bounds = read(path / "boundary.json")["effective"]
            leg = read(path / "leg-result.json")
            raw = read(path / "native-raw.json")
            assert leg["leg"] == index
            assert [leg["from"], leg["to"], leg["departure"], leg["arrival"]] == [flown[index]["from"], flown[index]["to"], flown[index]["t0"], flown[index]["tf"]]
            if index == 0:
                assert bounds["initial_mass"] == 3000
                with np.load(path / "seed.npz") as passed, np.load(KIT / "inputs" / expected["first_seed"]) as saved:
                    for name in ("initial_state", "node_epochs_mjd", "thrust_n"):
                        assert np.array_equal(passed[name], saved[name])
                assert raw["seed_backend"] == "cuda_zoh_replay"
            else:
                previous = flown[index-1]
                body = str(previous["to"])
                mass = prior_mass
                assert mass is not None
                if original["deploy_epochs"].get(body) == previous["tf"]:
                    mass -= 40
                elif original["collect_epochs"].get(body) == previous["tf"]:
                    mass += original["collected_mass_kg"][body]
                if original["collect_epochs"].get(body) == flown[index]["t0"] and original["collect_epochs"].get(body) != previous["tf"]:
                    mass += original["collected_mass_kg"][body]
                assert bounds["initial_mass"] == mass
                assert raw["seed_backend"] == "cuda"
            assert raw["iterations"] <= 44
            case_iterations += raw["iterations"]
            with np.load(path / "native-raw.npz") as arrays:
                x = arrays["states_scaled"]
                final = np.r_[x[-1, :3] * DU, x[-1, 3:6] * VU, x[-1, 6] * bounds["initial_mass"]]
                residual_position = float(np.linalg.norm(final[:3] - bounds["arrival_position"]))
                residual_velocity = float(np.linalg.norm(final[3:6] - bounds["arrival_velocity"]))
                assert np.isfinite(x).all() and np.isfinite(arrays["thrust_n"]).all()
                maximum_thrust = float(np.max(np.linalg.norm(arrays["thrust_n"], axis=1)))
            if leg["certified"]:
                certificate_file = directory / "propagation" / f"flight-{index}.npz"
                with np.load(certificate_file) as propagation:
                    assert len(propagation["results"]) == 1
                    actual = propagation["results"][0]
                    assert actual["status"] == 0
                    assert float(actual["final_state"][6]) == leg["certificate"]["final_mass_kg"]
                    prior_mass = float(actual["final_state"][6])
                assert read(certificate_file.with_suffix(".json"))["API_status"] == 0
                certificates += 1
            else:
                assert index == len(legs) - 1 and leg["certificate"] is None
                assert not (directory / "propagation" / f"flight-{index}.npz").exists()
            solvers = raw["solver_reports"]
            entries.append({"leg": index, "flight": [leg[k] for k in ("from", "to", "departure", "arrival")],
                            "initial_mass_kg": bounds["initial_mass"], "minimum_final_mass_kg": bounds["minimum_final_mass"],
                            "raw_transcription_final_mass_kg": float(final[6]), "certified": leg["certified"],
                            "native_status": raw["status"], "diagnostic": raw["diagnostic"],
                            "iterations": raw["iterations"], "accepted_iterations": raw["accepted_iterations"],
                            "max_defect": raw["max_defect"], "virtual_inf": raw["virtual_inf"],
                            "raw_arrival_node_position_error_km": residual_position,
                            "raw_arrival_body_velocity_difference_km_s": residual_velocity,
                            "free_arrival_vinf": bounds["free_arrival_vinf"],
                            "maximum_raw_thrust_n": maximum_thrust,
                            "conic_reports": len(solvers), "qualified_conic_reports": sum(bool(r["qualified"]) for r in solvers),
                            "conic_status_counts": dict(Counter(str(r["qoco_status"]) for r in solvers)),
                            "last_qualified_conic": next((r for r in reversed(solvers) if r["qualified"]), None),
                            "last_conic": solvers[-1] if solvers else None,
                            "native_raw_sha256": sha(path / "native-raw.npz"),
                            "leg_record_sha256": sha(path / "leg-result.json")})
        assert case_iterations == case["native_iterations"] and certificates == case["flight_certificate_calls"]
        wait_files = list((directory / "propagation").glob("wait-*.npz"))
        assert len(wait_files) == case["wait_certificate_calls"]
        for file in wait_files:
            with np.load(file) as waiting:
                assert waiting["results"][0]["status"] == 0
                assert waiting["results"][0]["final_state"][6] == waiting["legs"][0]["initial"][6]
        telemetry = case["boundary_ephemeris"]
        assert telemetry["backend"] == "cuda" and telemetry["batches"] == 1 and telemetry["CPU_ephemeris_calls"] == 0
        rows.append({"id": expected["id"], "ship": expected["ship"], "request_sha256": expected["request_sha256"],
                     "source_member": expected["source_member"], "source_member_sha256": expected["source_member_sha256"],
                     "prescribed_raw_delta_kg": expected["raw_delta_kg"], "prescribed_weighted_delta_kg": expected["weighted_delta_kg"],
                     "seconds": case["seconds"], "boundary_ephemeris": telemetry, "legs": entries,
                     "first_failure": entries[-1], "fixed_prescription_unchanged": True,
                     "all_completed_mass_handoffs_exact": True, "historical_failed_prefixes": expected["historical_failures"]})
        for key in ("native_solves_started", "native_solves_returned", "native_iterations", "flight_certificate_calls", "wait_certificate_calls", "seeded_calls_started", "cold_calls_started"):
            total[key] += case[key]
    for key, count in total.items():
        assert report[key] == count
    result = {"passed": True, "scope": "Saved immutable inputs, JSON, NPZ readbacks, mass handoff arithmetic and provenance only; no trajectory propagation, ephemeris, Lambert, solver or verifier reruns",
              "report_sha256": sha(OUT / "report.json"), "launch_sha256": sha(OUT / "launch-report.json"),
              "ready_sha256": sha(KIT / "ready.json"), "counts": total, "cases": rows,
              "CUDA_boundary_batches": 3, "CUDA_boundary_state_requests": sum(r["boundary_ephemeris"]["state_requests"] for r in rows),
              "remaining_CPU_body_state_calls": report["runtime_CPU_body_state_calls"],
              "worker_seconds": report["worker_seconds"], "verified_new_fleet": False,
              "all_failed_routes_retained": True, "baseline_Result_sha256": protocol["baseline_Result_sha256"],
              "baseline_weighted_kg": protocol["baseline_weighted_kg"], "baseline_raw_kg": protocol["baseline_raw_kg"],
              "outputs": {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}}
    with (HERE / "report.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "counts": total, "failures": [{k: r["first_failure"][k] for k in ("leg", "flight", "native_status", "max_defect", "virtual_inf", "raw_arrival_node_position_error_km", "qualified_conic_reports", "conic_reports")} for r in rows], "report_sha256": sha(HERE / "report.json")}))


if __name__ == "__main__":
    main()
