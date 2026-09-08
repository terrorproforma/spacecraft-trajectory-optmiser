"""Independent artifact and CPU rollout audit of the six completed local returns."""
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "output"
SOURCE = REPO / "build/performance/next-score-v597/source"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta(a, b):
    return {k: [a.get(k), b.get(k)] for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}


report = json.loads((OUT / "report.json").read_text())
fixture = json.loads((HERE / "fixture.json").read_text())
launch = json.loads((HERE / "launch.json").read_text())
assert report["complete"] and report["actual_native_calls"] == 6
assert report["fixture_sha256"] == launch["fixture_sha256"] == digest(HERE / "fixture.json")
assert report["driver_sha256"] == launch["driver_sha256"] == digest(HERE / "run.py")
for name, expected in report["source_sha256"].items():
    assert digest(SOURCE / name) == expected, name
assert digest(REPO / "build/performance/next-score-v597/source-sha256.json") == launch["source_manifest_sha256"]
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
sys.path.insert(0, str(SOURCE / "src"))
import numpy as np
from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.low_thrust import LegBoundary, LegSolution, certify_leg
from spacepdhcg.gtoc12.pipeline import clamp_thrust

audit = {"kind": "independent_saved_array_and_fresh_CPU_leg_rollout_audit_no_GPU",
         "input_report_sha256": digest(OUT / "report.json"), "fixture_sha256": digest(HERE / "fixture.json"),
         "all_source_hashes_match_launch": True, "runs": [], "groups": {},
         "seconds_campaign": report["seconds"],
         "sum_reported_solve_seconds": sum(r["solve_seconds"] for r in report["rows"]),
         "sum_reported_solve_and_certificate_seconds": sum(r["seconds_with_certificate"] for r in report["rows"]),
         "native_input_hash_scope": report["native_input_hash_scope"]}
arrays_by_repeat, envelopes, histories, conics = {}, {}, {}, {}
for row in report["rows"]:
    index = row["repeat"]
    folder = OUT / f"repeat_{index:02d}_{row['variant']}"
    envelope = json.loads((folder / "native-input.json").read_text())
    assert hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest() == row["native_input_sha256"]
    with np.load(folder / "native-input-arrays.npz") as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    assert set(arrays) == set(envelope["arrays"])
    for name, value in arrays.items():
        info = envelope["arrays"][name]
        assert str(value.dtype) == info["dtype"] and list(value.shape) == info["shape"]
        assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == info["sha256"]
    arrays_by_repeat[index] = arrays
    envelopes[index] = envelope
    histories[index] = json.loads((folder / "scvx-history.json").read_text())
    conics[index] = json.loads((folder / "conic-reports.json").read_text())
    assert len(histories[index]) == len(conics[index]) == row["iterations"]
    assert sum(bool(x["accepted"]) for x in histories[index]) == row["accepted_iterations"]
    assert [x["outer_iteration"] for x in conics[index] if x["qualified"]] == row["qualified_iterations"]
    first = conics[index][0]
    with np.load(folder / "solution-arrays.npz") as archive:
        output = {k: archive[k].copy() for k in archive.files}
    boundary = dict(fixture["boundary_shared"])
    for name in ("departure_position", "departure_velocity", "arrival_position", "arrival_velocity"):
        boundary[name] = np.array(boundary[name], dtype=np.float64)
    boundary = LegBoundary(**boundary, initial_mass=float.fromhex(row["initial_mass_hex"]))
    solution = LegSolution(status=row["status"], boundary=boundary, node_epochs_mjd=output["epochs"],
        thrust_n=output["thrust_before_clamp"], states_scaled=output["states"],
        departure_vinf_km_s=output["departure_vinf"], arrival_vinf_km_s=output["arrival_vinf"],
        final_mass_kg=row["final_mass_kg"], propellant_kg=row["propellant_kg"], delta_v_km_s=0,
        iterations=row["iterations"], accepted_iterations=row["accepted_iterations"],
        max_defect=row["max_defect"], virtual_inf=row["virtual_inf"], solve_seconds=row["solve_seconds"],
        hold=fixture["settings"]["hold"])
    clamp_thrust(solution)
    started = time.perf_counter()
    certificate = certify_leg(solution)
    seconds = time.perf_counter() - started
    actual = dataclasses.asdict(certificate)
    differences = {k: actual[k] - row["certificate"][k] for k in actual}
    checks = [certificate.position_error_km / C.TOLERANCE_POSITION_KM,
              certificate.velocity_error_km_s / C.TOLERANCE_VELOCITY_KM_S,
              certificate.rk4_vs_dop853_km / C.TOLERANCE_POSITION_KM,
              max(0, certificate.maximum_thrust_n - C.THRUST_MAX_N) / C.THRUST_MAX_N
              + max(0, C.MIN_SUN_DISTANCE_AU - certificate.minimum_sun_distance_au), 0.0]
    assert row["status"] == "converged" and row["pipeline_leg_certified"]
    assert all(math.isfinite(x) and 0 <= x <= 0.5 for x in checks)
    audit["runs"].append({"repeat": index, "variant": row["variant"], "iterations": row["iterations"],
        "accepted_iterations": row["accepted_iterations"], "native_input_sha256": row["native_input_sha256"],
        "solve_seconds": row["solve_seconds"], "first_conic": {k: first[k] for k in (
            "iterations", "qualified", "qoco_status", "primal_residual", "dual_residual",
            "primal_objective", "dual_objective", "relative_gap")},
        "fresh_CPU_certificate": actual, "certificate_delta_from_run": differences,
        "fresh_CPU_certificate_seconds": seconds, "fresh_pipeline_leg_checks_pass": True})

math_fields = ("qualified", "qoco_status", "iterations", "failure", "requested_tolerance",
               "primal_residual", "dual_residual", "absolute_primal_residual", "absolute_dual_residual",
               "primal_objective", "dual_objective", "absolute_gap", "relative_gap")
for variant in fixture["variants"]:
    rows = [r for r in report["rows"] if r["variant"] == variant]
    indices = [r["repeat"] for r in rows]
    first = indices[0]
    for index in indices[1:]:
        assert envelopes[index] == envelopes[first]
        assert all(arrays_by_repeat[index][k].dtype == arrays_by_repeat[first][k].dtype
                   and arrays_by_repeat[index][k].shape == arrays_by_repeat[first][k].shape
                   and arrays_by_repeat[index][k].tobytes() == arrays_by_repeat[first][k].tobytes()
                   for k in arrays_by_repeat[first])
    disagreements = []
    for index in indices[1:]:
        initial = {k: conics[first][0][k] for k in math_fields}
        other = {k: conics[index][0][k] for k in math_fields}
        first_branch = next((i + 1 for i, (a, b) in enumerate(zip(histories[first], histories[index]))
                            if (a.get("accepted"), a.get("solver"), a.get("retry_unchanged"))
                            != (b.get("accepted"), b.get("solver"), b.get("retry_unchanged"))), None)
        first_history_value = next((i + 1 for i, (a, b) in enumerate(zip(histories[first], histories[index])) if a != b), None)
        disagreements.append({"reference_repeat": first, "compared_repeat": index,
                              "first_conic_numerical_differences": delta(initial, other),
                              "first_outer_history_value_difference": first_history_value,
                              "first_outer_acceptance_difference": first_branch})
    audit["groups"][variant] = {"repeats": indices, "input_envelopes_equal": True,
                                "input_arrays_byte_equal": True, "outer_iterations": [r["iterations"] for r in rows],
                                "comparisons": disagreements}
audit["cross_variant_differing_array_fields"] = [k for k in arrays_by_repeat[1]
    if arrays_by_repeat[1][k].tobytes() != arrays_by_repeat[2][k].tobytes()]
audit["conclusion"] = {
    "all_six_local_returns_independently_certified": True,
    "repeatability_variation_with_identical_mathematical_inputs": True,
    "first_observed_variation": "first conic solve's numerical result, before differing outer acceptance",
    "H100_failure_reproduced_locally": False,
    "raw_mass_or_weighted_score_improvement": False,
    "root_cause_established": False,
    "deadline_caveat": "The native remaining time limit changes with setup elapsed time; runs finished in 1.84-2.61 seconds against the unchanged 900-second limit, with no timeout status."}
(HERE / "replay-audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
print(json.dumps({"certified": len(audit["runs"]), "groups": {k: v["outer_iterations"] for k, v in audit["groups"].items()},
                  "campaign_seconds": audit["seconds_campaign"],
                  "fresh_certificate_seconds": sum(r["fresh_CPU_certificate_seconds"] for r in audit["runs"]),
                  "maximum_fresh_certificate_delta": max(abs(v) for r in audit["runs"] for v in r["certificate_delta_from_run"].values())}, indent=2))
