"""Saved-output comparisons only; no project import, solver or trajectory propagation."""

import hashlib
import json
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
OUT = KIT / "output"


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


assert __debug__
ready = read(KIT / "ready.json")
assert sha(KIT / "ready.json") == "f053a57f9996244c815924d6b41e9cefa8f1fc7c47a652d50a4cd1d3d643fefe"
for name, expected in ready["files"].items():
    assert sha(KIT / name) == expected, name
for name, expected in ready["external_files"].items():
    assert sha(KIT / name) == expected, name
report, launch = read(OUT / "report.json"), read(OUT / "launch-report.json")
assert report["complete"] and report["status"] == "candidate_return_failed"
assert launch["complete"] and launch["passed"] and launch["exit_code"] == 0
assert launch["owned_descendant_cleanup"]["verified_empty"]
assert not launch["compute_processes_after_cleanup"] and not launch["failures"]
assert report["native_solves_started"] == report["native_solves_returned"] == 2
assert report["native_iterations"] == 24 and report["flight_certificate_calls"] == 1
assert report["wait_certificate_calls"] == report["cold_calls_started"] == 0
assert report["full_fleet_CPU_checks"] == report["full_fleet_official_checks"] == 0
assert report["incumbent_promoted"] is False and not (OUT / "candidate-fleet").exists()
source = arrays(KIT / "inputs/archived-return.npz")
protocol = read(KIT / "protocol.json")
cases = []
for name, key in (("original_mass_control", "control"), ("candidate_mass_return", "candidate")):
    directory = OUT / name
    seed = arrays(directory / "seed-upload.npz")
    for field in ("initial_state", "thrust_n", "node_epochs_mjd"):
        assert seed[field].tobytes() == source[field].tobytes(), (name, field)
    target = float(seed["target_initial_mass_kg"])
    assert target == protocol[key]["target_initial_mass_kg"]
    raw = arrays(directory / "legs/00/native-raw.npz")
    copied = arrays(directory / "legs/00/post-clamp.npz")
    details = read(directory / "legs/00/native-raw.json")
    leg = read(directory / "leg-result.json")
    boundary = read(directory / "boundary.json")["effective"]
    assert raw["epochs"].tobytes() == source["node_epochs_mjd"].tobytes()
    assert raw["states_scaled"].shape == (226, 7) and raw["thrust_n"].shape == (226, 3)
    assert all(np.isfinite(a).all() for a in raw.values())
    assert details["seed_backend"] == "cuda_zoh_mass_scaled_replay"
    assert details["outer_transfer_bytes"]["trajectory_upload_bytes"] == 5480
    assert leg["initial_mass_kg"] == target
    node_mass = float(raw["states_scaled"][-1, 6] * target)
    assert abs(node_mass - details["final_mass_kg"]) < 1e-10
    qualified = [r for r in details["solver_reports"] if r.get("qualified") == 1]
    priced = [r for r in details["history"] if "merit" in r]
    comparison = {
        "case": name, "initial_mass_kg": target, "iterations": details["iterations"],
        "accepted_iterations": details["accepted_iterations"], "status": details["status"],
        "diagnostic": details["diagnostic"], "native_seconds": details["solve_seconds"],
        "max_defect": details["max_defect"], "virtual_inf": details["virtual_inf"],
        "final_node_mass_kg": node_mass, "node_mass_is_independently_certified": False,
        "minimum_final_mass_kg": boundary["minimum_final_mass"],
        "node_mass_shortfall_kg": boundary["minimum_final_mass"] - node_mass,
        "qualified_conic_reports": len(qualified), "conic_reports": len(details["solver_reports"]),
        "all_priced_steps_rejected": all(row["accepted"] == 0 for row in priced),
        "priced_step_rows": priced,
        "raw_vs_post_pipeline_max_thrust_difference_n": float(np.max(np.abs(raw["thrust_n"] - copied["thrust_n"]))),
        "source_seed_arrays_uploaded_bit_exact": True, "trajectory_payload_bytes": 5480,
        "leg_result_sha256": sha(directory / "leg-result.json"),
        "raw_npz_sha256": sha(directory / "legs/00/native-raw.npz"),
    }
    if key == "control":
        assert leg["certified"] and leg["certificate"] is not None
        assert details["iterations"] == details["accepted_iterations"] == 2
        certificate = arrays(directory / "propagation/flight-0.npz")
        cert_metadata = read(directory / "propagation/flight-0.json")
        assert cert_metadata["API_status"] == 0 and certificate["results"][0]["status"] == 0
        final_mass = float(certificate["results"][0]["final_state"][6])
        assert final_mass == leg["certificate"]["final_mass_kg"]
        assert final_mass >= boundary["minimum_final_mass"] - 1e-9
        comparison["independent_certificate"] = leg["certificate"]
        comparison["certificate_readback_sha256"] = sha(directory / "propagation/flight-0.npz")
    else:
        assert not leg["certified"] and leg["certificate"] is None
        assert details["iterations"] == 22 and details["accepted_iterations"] == 0
        assert not (directory / "propagation").exists()
        for field in raw:
            assert raw[field].tobytes() == copied[field].tobytes()
        factor = target / float(source["initial_state"][6])
        delta = float(np.max(np.abs(raw["thrust_n"] - source["thrust_n"] * factor)))
        assert delta < 1e-12
        comparison["analytical_scaled_seed_thrust_max_difference_n"] = delta
        comparison["candidate_certificate"] = None
    cases.append(comparison)
audit = {
    "passed": True, "scope": "Static saved readback/input/counter comparison; no numerical rerun",
    "prepared_files_rehashed": len(ready["files"]), "external_files_rehashed": len(ready["external_files"]),
    "cases": cases, "worker_seconds": report["worker_seconds"],
    "native_calls": 2, "SCvx_iterations": 24, "new_flight_certificates": 1,
    "new_wait_certificates": 0, "full_fleet_checks": 0, "new_fleet_results": 0,
    "score_unchanged": True, "infeasibility_certificate": False,
    "actual_current_Result_sha256": protocol["baseline_Result_sha256"],
    "report_sha256": sha(OUT / "report.json"), "launch_report_sha256": sha(OUT / "launch-report.json"),
    "raw_output_files": {str(p.relative_to(OUT)): {"sha256": sha(p), "bytes": p.stat().st_size}
                         for p in sorted(OUT.rglob("*")) if p.is_file()},
}
with (KIT / "saved-audit.json").open("x") as stream:
    json.dump(audit, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"passed": True, "output_files": len(audit["raw_output_files"]),
                  "audit_sha256": sha(KIT / "saved-audit.json"), "cases": cases}, indent=2))
