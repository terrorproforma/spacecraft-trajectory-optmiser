"""At most six fixed-input local return solves; no GPU calls without --execute.

Default mode captures the CPU-created native input envelopes and stops before
calling CUDA. --execute replays the same inputs on the pinned v596/QOCO540 core.
No route or fleet is promoted by this diagnostic.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

EXPECTED_FIXTURE = "af5d8ad33e8a935b033de8ef0cc98e1dfba94d813d6a490b6e8a7d824877d4eb"
EXPECTED_LIBRARIES = {
    "SPACEPDHCG_GTOC12_CUDA_LIBRARY": "6fd02e87f3e148c0e62aac2bee731bc58397605a1d29ef3aebec95b0b4303c6c",
    "SPACEPDHCG_QOCO_LIBRARY": "0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(data), indent=2, default=float, allow_nan=False) + "\n")
    temp.replace(path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--fixture", type=Path, default=Path(__file__).resolve().parent / "fixture.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-solves", type=int, default=6)
    p.add_argument("--wall-seconds", type=float, default=600)
    p.add_argument("--lock", type=Path, default=Path.home() / ".spacepdhcg-gpu.lock")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if not 1 <= args.max_solves <= 6 or not 0 < args.wall_seconds <= 1800:
        p.error("at most six solves; total soft wall budget in (0,1800]")
    return args


def run(args):
    if digest(args.fixture) != EXPECTED_FIXTURE:
        raise ValueError("fixture hash differs from the audited H100 return inputs")
    fixture = json.loads(args.fixture.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    args._output_created = True
    report = {"pid": os.getpid(), "complete": False, "status": "preparing", "rows": [],
              "fixture_sha256": EXPECTED_FIXTURE, "driver_sha256": digest(Path(__file__)),
              "execute": args.execute, "actual_native_calls": 0,
              "physics_tolerances_changed": False, "fleet_promotion": False,
              "native_input_hash_scope": "All numerical arrays and physical/model/settings inputs before solve_native; excludes wall-clock start and remaining deadline."}
    save(out / "report.json", report)
    if args.execute:
        for name, expected in EXPECTED_LIBRARIES.items():
            library = Path(os.environ.get(name, ""))
            if not library.is_file() or digest(library) != expected:
                raise ValueError(f"{name} must be the pinned v596/QOCO540 library")
        report["native_sha256"] = dict(EXPECTED_LIBRARIES)
    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    import numpy as np
    from spacepdhcg.gtoc12 import constants as C, gpu_scvx
    from spacepdhcg.gtoc12.data import load_catalogue
    from spacepdhcg.gtoc12.gpu_execution import using_gpu_execution
    from spacepdhcg.gtoc12.low_thrust import LegBoundary, ScvxSettings, certify_leg, solve_leg
    from spacepdhcg.gtoc12.pipeline import body_state, clamp_thrust
    catalogue = load_catalogue()
    if catalogue.source_sha256 != fixture["catalogue_sha256"]:
        raise ValueError("catalogue does not match the pinned fixture")
    shared = dict(fixture["boundary_shared"])
    for name in ("departure_position", "departure_velocity", "arrival_position", "arrival_velocity"):
        shared[name] = np.array(shared[name], dtype=np.float64)
    r0, v0 = body_state(catalogue, fixture["from"], shared["departure_epoch"])
    rf, vf = body_state(catalogue, fixture["to"], shared["arrival_epoch"])
    for name, value in zip(("departure_position", "departure_velocity", "arrival_position", "arrival_velocity"), (r0, v0, rf, vf)):
        if not np.array_equal(shared[name], value):
            raise ValueError(f"reconstructed ephemeris differs: {name}")
    settings = ScvxSettings(**fixture["settings"])
    report.update(settings=dataclasses.asdict(settings), catalogue_sha256=catalogue.source_sha256,
                  source_sha256={str(p.relative_to(repo)): digest(p)
                                 for p in sorted((repo / "src/spacepdhcg/gtoc12").glob("*.py"))},
                  tolerances={"position_km": C.TOLERANCE_POSITION_KM,
                              "velocity_km_s": C.TOLERANCE_VELOCITY_KM_S,
                              "mass_kg": C.TOLERANCE_MASS_KG, "pipeline_normalized_maximum": 0.5})
    original = gpu_scvx.solve_native
    current_row = None
    current_directory = None

    class CaptureComplete(Exception):
        pass

    def captured(boundary, settings, model, times, days, bnd, fuel, seed_states, seed_controls, started):
        arrays = {"times": times, "days": days, "fuel": fuel, **bnd}
        if seed_states is not None:
            arrays["seed_states"] = seed_states
        if seed_controls is not None:
            arrays["seed_controls"] = seed_controls
        array_hashes = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
                        for name, value in arrays.items()}
        envelope = {"arrays": array_hashes, "settings": dataclasses.asdict(settings),
                    "initial_mass_hex": float(boundary.initial_mass).hex(),
                    "minimum_final_mass_hex": float(boundary.minimum_final_mass).hex(),
                    "minimum_mass_fraction_hex": float(boundary.minimum_final_mass / boundary.initial_mass).hex(),
                    "model_kappa_hex": float(model.kappa).hex(), "model_lam_hex": float(model.lam).hex(),
                    "radius_floor_hex": float(C.MIN_SUN_DISTANCE_AU).hex(),
                    "maximum_vinf_km_s_hex": float(C.MAX_VINF_EARTH_KM_S).hex(),
                    "free_departure_vinf": boundary.free_departure_vinf,
                    "free_arrival_vinf": boundary.free_arrival_vinf,
                    "seed_states_is_none": seed_states is None, "seed_controls_is_none": seed_controls is None}
        current_row["native_input_sha256"] = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest()
        current_row["nodes"] = len(times)
        save(current_directory / "native-input.json", envelope)
        np.savez_compressed(current_directory / "native-input-arrays.npz", **arrays)
        save(out / "report.json", report)
        if not args.execute:
            raise CaptureComplete()
        report["actual_native_calls"] += 1
        save(out / "report.json", report)
        return original(boundary, settings, model, times, days, bnd, fuel, seed_states, seed_controls, started)

    lock_context = args.lock.open("a") if args.execute else nullcontext()
    with lock_context as lock:
        if args.execute:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution = argparse.Namespace(gpu_execution="graph", outer_loop_backend="cuda", workers=1)
        with using_gpu_execution(execution):
            report["feature_flags"] = {k: v for k, v in os.environ.items() if k.startswith("SPACEPDHCG_TEST_")}
            began = time.perf_counter()
            deadline = began + args.wall_seconds
            gpu_scvx.solve_native = captured
            try:
                for index, name in enumerate(fixture["repeat_order"][:args.max_solves]):
                    if time.perf_counter() >= deadline:
                        break
                    variant = fixture["variants"][name]
                    mass = float.fromhex(variant["initial_mass_hex"])
                    if mass != variant["initial_mass"]:
                        raise ValueError("mass hex and decimal disagree")
                    boundary = LegBoundary(**shared, initial_mass=mass)
                    current_directory = out / f"repeat_{index + 1:02d}_{name}"
                    current_directory.mkdir()
                    current_row = {"repeat": index + 1, "variant": name, "initial_mass_kg": mass,
                                   "initial_mass_hex": mass.hex(), "historical_status": variant["historical_status"]}
                    report["rows"].append(current_row)
                    report["status"] = "native_return_solve" if args.execute else "capturing_without_gpu"
                    save(out / "report.json", report)
                    start = time.perf_counter()
                    try:
                        solution = solve_leg(boundary, settings)
                    except CaptureComplete:
                        current_row.update(status="captured_without_gpu", seconds=time.perf_counter() - start)
                        save(out / "report.json", report)
                        continue
                    except Exception as error:
                        current_row.update(status="exception", error=repr(error), seconds=time.perf_counter() - start)
                        save(out / "report.json", report)
                        continue
                    current_row.update(status=solution.status, diagnostic=solution.diagnostic,
                                       iterations=solution.iterations, accepted_iterations=solution.accepted_iterations,
                                       max_defect=solution.max_defect, virtual_inf=solution.virtual_inf,
                                       final_mass_kg=solution.final_mass_kg, propellant_kg=solution.propellant_kg,
                                       solve_seconds=solution.solve_seconds,
                                       qualified_iterations=[r["outer_iteration"] for r in solution.solver_reports if r["qualified"]])
                    save(current_directory / "scvx-history.json", solution.history)
                    save(current_directory / "conic-reports.json", solution.solver_reports)
                    np.savez_compressed(current_directory / "solution-arrays.npz", states=solution.states_scaled,
                                        thrust_before_clamp=solution.thrust_n, epochs=solution.node_epochs_mjd,
                                        departure_vinf=solution.departure_vinf_km_s, arrival_vinf=solution.arrival_vinf_km_s)
                    if solution.status not in ("failed", "infeasible", "timeout"):
                        # Exactly the existing pipeline clamp and independent rollout. This
                        # diagnostic never turns a reported infeasible solve into a usable arc.
                        clamp_thrust(solution)
                        certificate = certify_leg(solution)
                        checks = [certificate.position_error_km / C.TOLERANCE_POSITION_KM,
                                  certificate.velocity_error_km_s / C.TOLERANCE_VELOCITY_KM_S,
                                  certificate.rk4_vs_dop853_km / C.TOLERANCE_POSITION_KM,
                                  max(0, certificate.maximum_thrust_n - C.THRUST_MAX_N) / C.THRUST_MAX_N
                                  + max(0, C.MIN_SUN_DISTANCE_AU - certificate.minimum_sun_distance_au), 0.0]
                        current_row["certificate"] = dataclasses.asdict(certificate)
                        current_row["pipeline_normalized_checks"] = checks
                        current_row["pipeline_leg_certified"] = all(math.isfinite(x) and 0 <= x <= 0.5 for x in checks)
                        current_row["physical_mass_margin_kg"] = certificate.final_mass_kg - boundary.minimum_final_mass
                    else:
                        current_row["pipeline_leg_certified"] = False
                    current_row["seconds_with_certificate"] = time.perf_counter() - start
                    save(out / "report.json", report)
                    print(json.dumps(clean(current_row)), flush=True)
            finally:
                gpu_scvx.solve_native = original
            report["seconds"] = time.perf_counter() - began
            report["soft_budget_overrun_seconds"] = max(0, time.perf_counter() - deadline)
    report["per_variant"] = {name: {
        "repeats": len(rows := [r for r in report["rows"] if r["variant"] == name]),
        "distinct_native_input_hashes": sorted({r["native_input_sha256"] for r in rows if "native_input_sha256" in r}),
        "statuses": [r["status"] for r in rows],
        "certified": sum(bool(r.get("pipeline_leg_certified")) for r in rows)} for name in fixture["variants"]}
    report.update(complete=True, status="diagnostic_complete" if args.execute else "native_inputs_captured_no_gpu_execution")
    save(out / "report.json", report)
    print(json.dumps({"status": report["status"], "actual_native_calls": report["actual_native_calls"],
                      "per_variant": report["per_variant"]}), flush=True)
    return 0


if __name__ == "__main__":
    args = arguments()
    try:
        raise SystemExit(run(args))
    except Exception as error:
        path = args.output.resolve() / "report.json"
        if getattr(args, "_output_created", False) and path.exists():
            report = json.loads(path.read_text())
            report.update(complete=True, status="diagnostic_failed", error=repr(error))
            save(path, report)
        raise
