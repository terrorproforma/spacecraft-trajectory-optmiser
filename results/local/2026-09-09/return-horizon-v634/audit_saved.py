"""Audit saved bytes, metadata and uploaded arrays only; no trajectory equations."""

import ast
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

KIT = "build/performance/return-horizon-v634"
READY = "ac6379bdaf8bdc28f4296509ab9aa3bddbac2903c53c9edd15759529879d5edf"
RESULT = "1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da"
BASELINE = "build/performance/mass-merit-return-v632/output/candidate_mass_return/legs/00/native-raw.json"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def npz(raw):
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            require(data[:6] == b"\x93NUMPY", "NPY signature")
            version = data[6]
            size = 2 if version == 1 else 4
            length = int.from_bytes(data[8:8 + size], "little")
            header = ast.literal_eval(data[8 + size:8 + size + length].decode().strip())
            require(not header["fortran_order"], "Expected C-order array")
            result[name.removesuffix(".npy")] = (header, data[8 + size + length:])
    return result


def audit(read):
    def load(name):
        return json.loads(read(name))

    def item(name):
        return load(f"{KIT}/{name}")

    ready_bytes = read(f"{KIT}/ready.json")
    require(sha(ready_bytes) == READY, "Reviewed ready identity")
    ready = json.loads(ready_bytes)
    for name, digest in ready["files"].items():
        require(sha(read(f"{KIT}/{name}")) == digest, f"Prepared input changed: {name}")
    report, launch = item("output/report.json"), item("output/launch-report.json")
    require(report["complete"] and report["status"] == "both_horizons_uncertified", "Terminal outcome")
    require(launch["complete"] and launch["passed"] and launch["exit_code"] == 0, "Supervisor outcome")
    require(launch["owned_descendant_cleanup"]["verified_empty"] and not launch["compute_processes_after_cleanup"], "Cleanup")
    require(launch["ready_sha256"] == READY, "Launched ready identity")
    require(report["native_solves_started"] == report["native_solves_returned"] == 2, "Native call budget")
    require(report["native_iterations"] == 76, "Iteration accounting")
    for name in ("flight_certificate_calls", "wait_certificate_calls", "full_fleet_CPU_checks",
                 "full_fleet_official_checks", "cold_calls_started", "standalone_inspection_calls",
                 "Lambert_calls", "fresh_prefix_propagations", "CPU_boundary_ephemeris_calls"):
        require(report[name] == 0, f"Unexpected work: {name}")
    require(report["ephemeris_batches_started"] == report["ephemeris_batches_returned"] == 1, "Ephemeris calls")
    require(report["ephemeris_state_requests"] == 2, "Ephemeris request budget")
    require(not report["incumbent_promoted"], "False promotion")
    require(item("fleet-binding.json")["result_sha256"] == RESULT, "Current baseline identity")
    original = item("inputs/candidate-plan.json")
    archive = npz(read(f"{KIT}/inputs/archived-return.npz"))
    earth = npz(read(f"{KIT}/output/earth-readback.npz"))
    requests = npz(read(f"{KIT}/output/earth-requests.npz"))
    require(earth["requests"] == requests["requests"], "Ephemeris request/readback binding")
    require(earth["requests"][0]["shape"] == (2,), "Request shape")
    require(len(earth["requests"][1]) == 32 and len(earth["results"][1]) == 112, "C API payload sizes")
    targets = item("output/earth-targets.json")
    require(sha(read(f"{KIT}/output/earth-elements.bin")) == targets["source_elements_sha256"], "Earth element source")
    require(item("output/earth-call.json")["status"] == 0, "Ephemeris API status")
    prior = load(BASELINE)
    cases = []
    for index, (arrival, nodes, updates, accepted) in enumerate(((69743, 241, 44, 7), (69773, 256, 32, 4))):
        base = f"output/arrival-{arrival}"
        plan = item(f"{base}/prescribed-plan.json")
        require(sorted(plan["asteroids"]) == sorted(original["asteroids"]), "Asteroid footprint")
        for name in original:
            if name not in ("asteroids", "earth_return_epoch", "legs"):
                require(plan[name] == original[name], f"Fixed prescription changed: {name}")
        require(plan["earth_return_epoch"] == arrival and plan["legs"][:-1] == original["legs"][:-1], "Fixed prefix")
        require(plan["legs"][-1] == original["legs"][-1] | {"tf": arrival}, "Only return horizon changes")
        payload = npz(read(f"{KIT}/{base}/seed-upload.npz"))
        require(payload["initial_state"] == archive["initial_state"], "Archived initial state changed")
        require(payload["target_initial_mass_kg"] == archive["target_initial_mass_kg"], "Mass factor changed")
        require(payload["thrust_n"][0]["shape"] == (nodes, 3), "Seed node count")
        require(payload["thrust_n"][1][:226 * 24] == archive["thrust_n"][1], "Archived controls changed")
        require(payload["thrust_n"][1][225 * 24:] == bytes((nodes - 225) * 24), "Tail is not exact zero coast")
        require(payload["node_epochs_mjd"][1][:226 * 8] == archive["node_epochs_mjd"][1], "Archived node epochs changed")
        times = struct.unpack(f"<{nodes}d", payload["node_epochs_mjd"][1])
        require(times == tuple(69263.0 + 2 * i for i in range(nodes)), "Target node grid")
        request = struct.unpack_from("<iid", earth["requests"][1], index * 16)
        result = struct.unpack_from("<6dii", earth["results"][1], index * 56)
        require(request == (0, 0, float(arrival)) and result[-2:] == (0, 0), "Earth request/row status")
        require(list(result[:3]) == targets["positions_km"][index] and list(result[3:6]) == targets["velocities_km_s"][index], "Earth target readback")
        boundary = item(f"{base}/boundary.json")["effective"]
        require(boundary["arrival_position"] == list(result[:3]) and boundary["arrival_velocity"] == list(result[3:6]), "GPU boundary target")
        require(boundary["initial_mass"] == 1438.8883038351994 and boundary["minimum_final_mass"] == 1177.2073921971253, "Fixed mass boundary")
        raw = item(f"{base}/legs/00/native-raw.json")
        leg = item(f"{base}/leg-result.json")
        reports, history = raw["solver_reports"], raw["history"]
        require(raw["iterations"] == len(reports) == len(history) == updates, "Update/readback accounting")
        require(raw["accepted_iterations"] == sum(bool(row["accepted"]) for row in history) == accepted, "Accepted updates")
        require(not leg["certified"] and leg["certificate"] is None and raw["status"] == "infeasible", "Qualification status")
        require(raw["seed_backend"] == "cuda_zoh_mass_scaled_replay", "Actual seed backend")
        require(raw["outer_transfer_bytes"]["trajectory_upload_bytes"] == (nodes * 3 + 7) * 8, "Seed payload bytes")
        require(read(f"{KIT}/{base}/legs/00/native-raw.npz") == read(f"{KIT}/{base}/legs/00/post-clamp.npz"), "Raw/post-clamp arrays differ")
        require(raw["max_defect"] > item("inputs/settings.json")["defect_tolerance"], "Dynamics failure disappeared")
        accepted_rows = [row for row in history if row["accepted"]]
        cases.append({"arrival_MJD": arrival, "days_added": arrival - 69713, "nodes": nodes,
                      "native_updates": updates, "accepted_updates": accepted,
                      "inner_conic_qualified": sum(bool(row["qualified"]) for row in reports),
                      "inner_conic_unqualified": sum(not bool(row["qualified"]) for row in reports),
                      "conditioning_retry_calls": sum(bool(row.get("conditioning_retry")) for row in reports),
                      "last_accepted_update": accepted_rows[-1]["iteration"],
                      "terminal_history": history[-1], "reported_max_defect": raw["max_defect"],
                      "reported_virtual_inf": raw["virtual_inf"], "defect_gate": 5e-9,
                      "returned_node_final_mass_kg_unverified": raw["final_mass_kg"],
                      "reported_propellant_kg_unverified": raw["propellant_kg"],
                      "native_call_seconds": item(f"{base}/legs/00/native-finish.json")["native_call_seconds"],
                      "raw_metadata_sha256": sha(read(f"{KIT}/{base}/legs/00/native-raw.json")),
                      "raw_arrays_sha256": sha(read(f"{KIT}/{base}/legs/00/native-raw.npz")),
                      "no_certificate_or_fleet_check": True,
                      "defect_larger_than_original_horizon": raw["max_defect"] > prior["max_defect"]})
    require(sum(row["native_updates"] for row in cases) == report["native_iterations"], "Aggregate update accounting")
    return {"passed": True, "scope": "Saved metadata, exact input bytes and uploaded/readback arrays only; no dynamics replay or new certificate",
            "Result_sha256": RESULT, "ready_sha256": READY, "cases": cases,
            "worker_seconds": report["worker_seconds"], "native_returns": 2, "native_updates": 76,
            "accepted_updates": 11, "certificates": 0, "full_fleet_checks": 0,
            "ephemeris_calls": 1, "ephemeris_states": 2, "ephemeris_bytes": {"elements_H2D": 56, "requests_H2D": 32, "results_D2H": 112},
            "original_horizon_reference": {"path": BASELINE, "sha256": sha(read(BASELINE)),
                "reported_max_defect": prior["max_defect"], "accepted_updates": prior["accepted_iterations"]},
            "returned_virtual_vector_saved": False, "new_residual_axis_or_interval_not_identified": True,
            "score_gain_kg": 0, "incumbent_retained": True, "cleanup_verified_empty": True,
            "fresh_audit_optimizer_calls": 0, "fresh_audit_propagations": 0, "fresh_audit_GPU_calls": 0}


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    result = audit(lambda name: (root / name).read_bytes())
    with (here / "saved-audit.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
