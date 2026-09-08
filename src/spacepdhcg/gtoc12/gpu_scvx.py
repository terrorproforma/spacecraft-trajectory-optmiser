"""GPU SCvx arithmetic and retained trajectories with native orchestration.

Seed generation is on GPU by default. Experimental native outer graphs retain
subproblem control on GPU after cold priming; setup and graph construction remain
host operations. Unmeasured per-phase graph timings are exported as None.
No Python numerical work or trajectory downloads occur between subproblems.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import time
from pathlib import Path

import numpy as np

from . import constants as C
from .gpu_discretisation import _DoublePointer, _pointer
from .gpu_qoco import _Report
from .low_thrust import VU_KM_S, LegSolution


class _Settings(ct.Structure):
    _fields_ = [
        (n, ct.c_int)
        for n in (
            "substeps",
            "polish_substeps",
            "polish_iterations",
            "max_iterations",
        )
    ] + [
        (n, ct.c_double)
        for n in (
            "virtual_weight",
            "smoothness_weight",
            "initial_trust_state",
            "initial_trust_control",
            "minimum_trust",
            "maximum_trust",
            "ratio_reject",
            "ratio_shrink",
            "ratio_grow",
            "shrink_factor",
            "grow_factor",
            "defect_tolerance",
            "step_tolerance",
            "objective_tolerance",
            "conic_tolerance",
            "time_limit_s",
            "minimum_mass",
            "radius_floor",
            "vinf_max",
        )
    ]


class _Record(ct.Structure):
    _fields_ = [
        (n, ct.c_int)
        for n in (
            "iteration",
            "accepted",
            "conic_rejected",
            "qoco_status",
        )
    ] + [
        (n, ct.c_double)
        for n in (
            "merit",
            "final_mass_fraction",
            "max_defect",
            "virtual_inf",
            "ratio",
            "step",
            "trust_state",
            "trust_control",
        )
    ]


class _Result(ct.Structure):
    _fields_ = (
        [
            (n, ct.c_int)
            for n in (
                "status",
                "diagnostic",
                "iterations",
                "accepted_iterations",
            )
        ]
        + [(n, ct.c_double) for n in ("max_defect", "virtual_inf")]
        + [
            ("departure_vinf", ct.c_double * 3),
            ("arrival_vinf", ct.c_double * 3),
        ]
        + [
            (n, ct.c_uint64)
            for n in (
                "trajectory_upload_bytes",
                "trajectory_download_bytes",
                "control_download_bytes",
            )
        ]
    )


def solve_native(
    boundary,
    settings,
    model,
    times,
    days,
    bnd,
    fuel,
    seed_states,
    seed_controls,
    started,
    *,
    seed=None,
):
    if seed is not None:
        from .trajectory_seed import ZohTrajectorySeed

        if not isinstance(seed, ZohTrajectorySeed):
            raise TypeError("seed must be a ZohTrajectorySeed")
        seed.validate_for(boundary, settings, boundary.departure_epoch + days)
        if seed_states is not None or seed_controls is not None:
            raise ValueError("a ZOH trajectory seed cannot be combined with internal seed arrays")
    conditioning_retry = os.environ.get("SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY") == "1"
    path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
    qoco = os.environ.get("SPACEPDHCG_QOCO_LIBRARY")
    if not path or not qoco or not Path(qoco).is_file():
        raise RuntimeError("CUDA outer loop requires configured native core and GPU QOCO libraries")
    library = ct.CDLL(str(Path(path).resolve(strict=True)))
    try:
        solve = (
            library.spacepdhcg_gtoc12_scvx_solve_zoh_seed_host
            if seed is not None
            else library.spacepdhcg_gtoc12_scvx_solve_host
        )
    except AttributeError as exc:
        if seed is not None:
            raise RuntimeError(
                "Native core lacks CUDA ZOH replay seed extension; no fallback"
            ) from exc
        raise RuntimeError("Native core lacks CUDA SCvx extension; no Python fallback") from exc
    if (
        seed is None
        and seed_states is None
        and not hasattr(library, "spacepdhcg_gtoc12_seed_evaluate_host")
    ):
        raise RuntimeError("Native core lacks GPU seed extension; no CPU fallback")
    solve.argtypes = (
        [ct.c_int] * 4
        + [ct.c_double] * 2
        + [_DoublePointer] * 5
        + [
            ct.c_int,
            ct.POINTER(_Settings),
            _DoublePointer,
            _DoublePointer,
            ct.POINTER(_Record),
            ct.POINTER(_Report),
            ct.POINTER(_Result),
        ]
    )
    solve.restype = ct.c_int
    params = _Settings()
    for name, kind in params._fields_:
        if hasattr(settings, name):
            value = getattr(settings, name)
            if kind is ct.c_int and (not isinstance(value, int) or not 0 <= value < 2**31):
                raise ValueError(f"{name} must be a nonnegative int32")
            setattr(params, name, value)
    params.conic_tolerance = settings.clarabel_tolerance
    params.minimum_mass = boundary.minimum_final_mass / boundary.initial_mass
    params.radius_floor = C.MIN_SUN_DISTANCE_AU
    params.vinf_max = C.MAX_VINF_EARTH_KM_S / VU_KM_S
    # Include Python seed/setup in the same user-specified wall budget.
    if not math.isfinite(settings.time_limit_s) or settings.time_limit_s < 0:
        raise ValueError("time_limit_s must be finite and nonnegative")
    params.time_limit_s = max(0.0, settings.time_limit_s - (time.perf_counter() - started))
    budget = settings.max_iterations + settings.polish_iterations
    if not 0 < budget < 2**31:
        raise ValueError("invalid total SCvx iteration budget")
    arrays = [
        None if v is None else np.ascontiguousarray(v, dtype=np.float64)
        for v in (
            times,
            np.concatenate([bnd[k] for k in ("r0", "v0", "rf", "vf")]),
            fuel,
            seed.initial_state if seed is not None else seed_states,
            seed.thrust_n if seed is not None else seed_controls,
        )
    ]
    states, controls = np.empty((len(times), 7)), np.empty((len(times), 4))
    records, reports, result = (_Record * budget)(), (_Report * budget)(), _Result()
    code = solve(
        len(times) - 1,
        int(settings.hold == "lagrange"),
        int(boundary.free_departure_vinf),
        int(boundary.free_arrival_vinf),
        model.kappa,
        model.lam,
        *[None if a is None else _pointer(a) for a in arrays],
        settings.qoco_ruiz_iterations,
        ct.byref(params),
        _pointer(states),
        _pointer(controls),
        records,
        reports,
        ct.byref(result),
    )
    if code:
        raise RuntimeError(f"CUDA SCvx failed (status {code}); no CPU fallback")
    history, solver_reports = [], []
    inaccurate_retries = 0
    for i in range(result.iterations):
        record, report = records[i], reports[i]
        if record.conic_rejected:
            rejection = "candidate_rejected" if report.qualified else "unqualified"
            entry = dict(
                iteration=record.iteration,
                accepted=0.0,
                solver=f"QOCO_{record.qoco_status}_{rejection}",
                retry_unchanged=record.conic_rejected == 2,
                trust_state=record.trust_state,
                trust_control=record.trust_control,
            )
        else:
            entry = {
                name: getattr(record, name)
                for name, _ in record._fields_
                if name not in {"conic_rejected", "qoco_status"}
            }
        history.append(entry)
        entry = {name: getattr(report, name) for name, _ in report._fields_}
        entry = {
            k: None if isinstance(v, float) and not math.isfinite(v) else v
            for k, v in entry.items()
        }
        # Reconstruct the native counter from its recorded unchanged-retry
        # transitions. This is final telemetry, never a host solve decision.
        conditioned = conditioning_retry and inaccurate_retries == 1
        entry.update(
            outer_iteration=i + 1,
            api_status=0 if report.qualified else 4,
            conditioning_retry=conditioned,
            ruiz_iterations=5 if conditioned else settings.qoco_ruiz_iterations,
        )
        inaccurate_retries = inaccurate_retries + 1 if record.conic_rejected == 2 else 0
        solver_reports.append(entry)
    diagnostics = (
        "",
        "convex subproblem unqualified",
        "trust region collapsed",
        "polish budget reached with converged defects",
        "iteration budget reached with converged defects",
        "trust region exhausted at a feasible point",
        f"virtual control remains {result.virtual_inf:.3e}",
        "stationary penalized point with nonzero defects; no infeasibility certificate",
    )
    final_mass = float(states[-1, 6]) * boundary.initial_mass
    thrust = controls[:, :3] * C.THRUST_MAX_N
    if settings.hold == "zoh":
        thrust[-1] = 0.0
    return LegSolution(
        status=("iteration_limit", "converged", "failed", "infeasible", "timeout")[result.status],
        boundary=boundary,
        node_epochs_mjd=boundary.departure_epoch + days,
        thrust_n=thrust,
        states_scaled=states,
        departure_vinf_km_s=np.array(result.departure_vinf) * VU_KM_S,
        arrival_vinf_km_s=np.array(result.arrival_vinf) * VU_KM_S,
        final_mass_kg=final_mass,
        propellant_kg=boundary.initial_mass - final_mass,
        delta_v_km_s=C.ISP_S * C.G0_M_S2 * 1e-3 * math.log(boundary.initial_mass / final_mass)
        if final_mass > 0
        else math.inf,
        iterations=result.iterations,
        accepted_iterations=result.accepted_iterations,
        max_defect=result.max_defect,
        virtual_inf=result.virtual_inf,
        solve_seconds=time.perf_counter() - started,
        hold=settings.hold,
        history=history,
        diagnostic=diagnostics[result.diagnostic],
        discretisation_backend="cuda",
        assembly_backend="cuda",
        convex_solver_backend="qoco",
        solver_reports=solver_reports,
        outer_loop_backend="cuda",
        seed_backend="cuda_zoh_replay"
        if seed is not None
        else ("cuda" if seed_states is None else "numpy"),
        outer_transfer_bytes={
            name: getattr(result, name)
            for name in (
                "trajectory_upload_bytes",
                "trajectory_download_bytes",
                "control_download_bytes",
            )
        },
    )
