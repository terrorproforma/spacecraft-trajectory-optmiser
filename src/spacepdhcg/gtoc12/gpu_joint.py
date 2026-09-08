"""Batched continuous itinerary arithmetic and element propagation on CUDA.

This is a search surrogate. Every promoted itinerary still needs independent
low-thrust certification; a feasible proxy is not a verified trajectory.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import time
from itertools import pairwise

import numpy as np

from . import constants as C


def _layout(ints, doubles):
    return np.dtype(
        [(name, np.int32) for name in ints] + [(name, np.float64) for name in doubles], align=True
    )


VISIT = _layout(
    ("deploy", "collect", "donor", "structural_failure"),
    ("foreign_epoch", "pinned_arrival", "dwell_limit", "weight"),
)
STAGE = _layout(
    ("model", "earth_out", "reserved0", "reserved1"),
    ("tof_min", "tof_max", "ratio_limit", "flat", "floor", "slope", "calibration"),
)
COST = _layout(("measured", "reserved"), ("lambert", "measured_delta_v", "measured_mass"))
POLICY = np.dtype(
    [
        (name, np.float64)
        for name in (
            "mission_start",
            "latest_arrival",
            "initial_mass",
            "dry_mass",
            "miner_mass",
            "minimum_stay",
            "mining_rate",
            "year_days",
            "thrust",
            "exhaust",
            "measured_mass_tolerance",
            "margin_price",
            "earth_out_tof_floor",
            "earth_out_inflation",
        )
    ]
    + [
        (name, np.int32)
        for name in ("free_earth_leg", "screen_earth_out", "reserved0", "reserved1")
    ],
    align=True,
)
RESULT = _layout(
    ("failure", "mass_count", "measured_legs", "rounds"),
    ("objective", "weighted", "collected", "spare", "propellant", "final_mass"),
)
SELECTION = np.dtype(
    [("index", np.int32), ("invalid_stay", np.int32), ("value", RESULT)], align=True
)
CACHED_COST = np.dtype(
    [
        ("leg", np.int32),
        ("cached", np.int32),
        ("departure", np.float64),
        ("arrival", np.float64),
        ("value", COST),
    ],
    align=True,
)
GEOMETRY_STATS = np.dtype(
    [(name, np.uint64) for name in ("computed_hops", "cached_hops", "rejected_hops")]
)
SEARCH_REPORT = np.dtype(
    [(name, np.int32) for name in ("levels", "moves", "stop", "invalid_stay")]
    + [(name, np.uint64) for name in ("batches", "evaluations")],
    align=True,
)
FAILURES = (
    "",
    "launch_before_window",
    "return_after_window",
    "earth_visits_have_no_dwell",
    "negative_dwell",
    "dwell_too_long",
    "pinned_arrival_moved",
    "tof_outside_limits",
    "double_deploy",
    "double_collect",
    "collect_without_deploy",
    "stay_too_short",
    "leg_infeasible",
    "earth_out_unmeasured_below_floor",
    "leg_authority",
    "mass_below_dry_plus_collected",
    "mass_below_dry",
    "invalid_stay",
)


def _metadata(joint, visits):
    n = len(visits)
    metadata = np.zeros(n, VISIT)
    stages = np.zeros(n - 1, STAGE)
    policy = np.zeros(1, POLICY)
    deployed, collected = {}, set()
    first_failure = False
    for j, visit in enumerate(visits):
        error = 0
        if visit.deploy:
            if visit.body in deployed:
                error = 8
            else:
                deployed[visit.body] = j
        if not error and visit.collect:
            if visit.body in collected:
                error = 9
            else:
                collected.add(visit.body)
                if visit.body not in deployed and visit.foreign_deploy_epoch is None:
                    error = 10
        if error and not first_failure:
            metadata[j]["structural_failure"] = error
            first_failure = True
        metadata[j]["deploy"] = visit.deploy
        metadata[j]["collect"] = visit.collect
        metadata[j]["foreign_epoch"] = (
            math.nan if visit.foreign_deploy_epoch is None else visit.foreign_deploy_epoch
        )
        metadata[j]["pinned_arrival"] = (
            math.nan if visit.pinned_arrival is None else visit.pinned_arrival
        )
        metadata[j]["dwell_limit"] = joint.dwell_limit(visit)
        metadata[j]["weight"] = 1.0 if joint.weights is None else joint.weights.get(visit.body, 1.0)
    for j, visit in enumerate(visits):
        metadata[j]["donor"] = deployed.get(
            visit.body, -2 if visit.foreign_deploy_epoch is None else -1
        )
    retimer, settings = joint.retimer, joint.retimer.settings
    for j, (visit, nxt) in enumerate(pairwise(visits)):
        stage = stages[j]
        role = visit.role_out
        stage["model"] = (
            2 if retimer._return_model(role) else 1 if retimer._ratio_model(role) else 0
        )
        stage["earth_out"] = role == "earth_out"
        stage["tof_min"], stage["tof_max"] = joint.tof_limits(role)
        stage["flat"], stage["ratio_limit"] = retimer._limits(role, visit.body, nxt.body)
        stage["floor"], stage["slope"] = (
            settings.hop_inflation_floor,
            settings.hop_inflation_slope or 0.0,
        )
        stage["calibration"] = retimer.inflations.get((visit.body, nxt.body), 1.0)
    for name, value in {
        "mission_start": C.MISSION_START_MJD,
        "latest_arrival": C.MISSION_END_MJD - settings.end_margin_days,
        "initial_mass": retimer.search_settings.initial_mass,
        "dry_mass": C.DRY_MASS_KG,
        "miner_mass": C.MINER_MASS_KG,
        "minimum_stay": C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS,
        "mining_rate": C.MINING_RATE_KG_PER_YEAR,
        "year_days": C.YEAR_DAYS,
        "thrust": C.THRUST_MAX_N,
        "exhaust": C.ISP_S * C.G0_M_S2 * 1e-3,
        "measured_mass_tolerance": joint.settings.measured_mass_tolerance_kg,
        "margin_price": joint.settings.margin_price,
        "earth_out_tof_floor": math.nan
        if retimer.earth_out_tof_floor is None
        else retimer.earth_out_tof_floor,
        "earth_out_inflation": math.nan
        if joint.earth_out_inflation is None
        else joint.earth_out_inflation,
        "free_earth_leg": joint.free_earth_leg,
        "screen_earth_out": joint._screen_earth_out,
    }.items():
        policy[0][name] = value
    return metadata, stages, policy


def _geometry_inputs(joint, visits):
    """Pack stage orbits and sparse authoritative overrides, never trial rows."""
    from .gpu_lambert import HopElements, body_elements

    elements = (HopElements * (len(visits) - 1))()
    stages_by_pair = {}
    for leg, (a, b) in enumerate(pairwise(visits)):
        elements[leg] = HopElements(
            body_elements(joint.catalogue, a.body),
            body_elements(joint.catalogue, b.body),
            C.MU_SUN_KM3_S2,
            C.MAX_VINF_EARTH_KM_S if a.body == 0 else 0.0,
            C.MAX_VINF_EARTH_KM_S if b.body == 0 else 0.0,
        )
        stages_by_pair.setdefault((a.body, b.body), []).append(leg)
    entries = []
    for key in joint._lambert.keys() | joint.measured.keys():
        a, b, departure, arrival = key
        if not math.isfinite(departure) or not math.isfinite(arrival):
            continue  # Cannot match any finite trial epoch accepted by the C API.
        for leg in stages_by_pair.get((a, b), ()):
            entries.append((leg, departure, arrival, key))
    entries.sort(key=lambda item: item[:3])
    records = np.zeros(len(entries), CACHED_COST)
    for index, (leg, departure, arrival, key) in enumerate(entries):
        row = records[index]
        row["leg"], row["departure"], row["arrival"] = leg, departure, arrival
        row["cached"] = key in joint._lambert
        row["value"]["lambert"] = joint._lambert.get(key, math.nan)
        measured = joint.measured.get(key)
        if measured is not None:
            row["value"]["measured"] = 1
            row["value"]["measured_delta_v"] = measured.delta_v_km_s
            row["value"]["measured_mass"] = measured.mass_before_kg
    return elements, records


class GpuJoint:
    def __init__(self, gpu, count, visits):
        self.gpu, self.capacity, self.visits = gpu, count, visits
        self.handle = ct.c_void_p()
        self.create = gpu.library.spacepdhcg_gtoc12_joint_create
        self.create.argtypes = [ct.c_int32, ct.c_int32, ct.c_int32, ct.POINTER(ct.c_void_p)]
        self.create.restype = ct.c_int
        self.evaluate = gpu.library.spacepdhcg_gtoc12_joint_evaluate_host
        self.evaluate.argtypes = [ct.c_void_p, ct.c_int32] + [ct.c_void_p] * 11
        self.evaluate.restype = ct.c_int
        self.best = getattr(gpu.library, "spacepdhcg_gtoc12_joint_best_host", None)
        self.geometry = getattr(gpu.library, "spacepdhcg_gtoc12_joint_geometry_host", None)
        self.mesh = getattr(gpu.library, "spacepdhcg_gtoc12_joint_mesh_host", None)
        self.search = getattr(gpu.library, "spacepdhcg_gtoc12_joint_search_host", None)
        if self.search is not None:
            self.search.argtypes = (
                [ct.c_void_p, ct.c_void_p, ct.c_int32, ct.c_int32, ct.c_double]
                + [ct.c_void_p] * 7
                + [ct.c_int32]
                + [ct.c_void_p] * 9
            )
            self.search.restype = ct.c_int
        if self.mesh is not None:
            self.mesh.argtypes = (
                [ct.c_void_p, ct.c_double]
                + [ct.c_void_p] * 7
                + [ct.c_int32, ct.c_double]
                + [ct.c_void_p] * 9
            )
            self.mesh.restype = ct.c_int
        if self.geometry is not None:
            self.geometry.argtypes = (
                [ct.c_void_p, ct.c_int32]
                + [ct.c_void_p] * 7
                + [ct.c_int32, ct.c_double]
                + [ct.c_void_p] * 7
            )
            self.geometry.restype = ct.c_int
        if self.best is not None:
            self.best.argtypes = (
                [ct.c_void_p, ct.c_int32] + [ct.c_void_p] * 6 + [ct.c_double] + [ct.c_void_p] * 5
            )
            self.best.restype = ct.c_int
        self.destroy = gpu.library.spacepdhcg_gtoc12_joint_destroy
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        self.destroy.restype = ct.c_int
        self._check(self.create(gpu.device_id, count, visits, ct.byref(self.handle)))

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA joint itinerary failed (native status {status})")

    def close(self):
        if self.handle.value:
            self.gpu._owned()
            self._check(self.destroy(ct.byref(self.handle)))

    def run(
        self,
        joint,
        visits,
        arrivals,
        departures,
        *,
        minimum_objective=None,
        mesh_delta=None,
        search_config=None,
    ):
        self.gpu._owned()
        if search_config is not None and self.search is None:
            raise RuntimeError("CUDA joint search requires spacepdhcg_gtoc12_joint_search_host")
        selection_override = os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION")
        if selection_override == "1" and self.best is None:
            raise RuntimeError(
                "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION=1 requires "
                "spacepdhcg_gtoc12_joint_best_host in the native library"
            )
        # Older cores retain native full-result arithmetic and host selection.
        # An explicit request for the optional reducer must fail before any work.
        device_selection = minimum_objective is not None and (
            self.best is not None if selection_override is None else selection_override == "1"
        )
        if mesh_delta is not None and (self.mesh is None or not device_selection):
            raise RuntimeError("CUDA joint mesh requires the mesh API and device winner selection")
        resident_geometry = (
            mesh_delta is not None
            or os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY", "0") == "1"
        )
        if resident_geometry and self.geometry is None:
            raise RuntimeError(
                "CUDA joint resident geometry requires spacepdhcg_gtoc12_joint_geometry_host"
            )
        metadata, stages, policy = _metadata(joint, visits)
        count, n = arrivals.shape
        mesh_epochs = None
        search_report = np.zeros(1, SEARCH_REPORT) if search_config is not None else None
        if mesh_delta is not None:
            count = 10 * n - 14
            mesh_epochs = (np.empty((1, n)), np.empty((1, n)))

        def finish(value):
            if search_report is not None:
                return value, *mesh_epochs, search_report[0]
            return value if mesh_epochs is None else (value, *mesh_epochs)

        if resident_geometry:
            elements, records = _geometry_inputs(joint, visits)
            costs = None
        else:
            costs = np.zeros((count, n - 1), COST)
            costs["lambert"] = np.nan
            preflight = np.empty(count, RESULT)
            # Validate epochs, event order and mining stays before touching body
            # geometry. An unmeasured NaN first leg stops each otherwise valid row
            # at leg_infeasible, after the same gates as scalar evaluate().
            preflight_arrays = (policy, metadata, stages, arrivals, departures, costs, preflight)
            self._check(
                self.evaluate(
                    self.handle,
                    count,
                    *(array.ctypes.data for array in preflight_arrays),
                    None,
                    None,
                    None,
                    None,
                )
            )
            telemetry = self.gpu.telemetry
            telemetry["joint_preflight_download_bytes"] = (
                telemetry.get("joint_preflight_download_bytes", 0) + preflight.nbytes
            )
            if np.any(preflight["failure"] == 17):
                raise ValueError("mining stay must be finite and nonnegative")
            active = np.flatnonzero(preflight["failure"] == 12)
            pending = {}
            keys = {}
            # Group unique uncached epochs by body pair. The native paired operator
            # propagates both bodies and screens the full batch on the GPU.
            for row in active:
                row_keys = []
                for j, (visit, nxt) in enumerate(pairwise(visits)):
                    departure, arrival = float(departures[row, j]), float(arrivals[row, j + 1])
                    key = joint.key(visit.body, nxt.body, departure, arrival)
                    row_keys.append(key)
                    if key not in joint._lambert:
                        pending.setdefault((visit.body, nxt.body), {}).setdefault(
                            key, (departure, arrival - departure)
                        )
                keys[row] = row_keys
            for (from_body, to_body), queries in pending.items():
                times = np.asarray(list(queries.values()), dtype=np.float64)
                hops = self.gpu.paired_hops(
                    joint.catalogue, from_body, to_body, times[:, 0], times[:, 1]
                )
                for key, value, feasible in zip(
                    queries, hops.total_delta_v, hops.feasible, strict=True
                ):
                    joint._lambert[key] = (
                        float(value) if feasible and math.isfinite(value) else math.inf
                    )
                joint.lambert_evaluations += 2 * len(queries)
            for row, row_keys in keys.items():
                for j, key in enumerate(row_keys):
                    costs[row, j]["lambert"] = joint._lambert[key]
                    measured = joint.measured.get(key)
                    if measured is not None:
                        costs[row, j]["measured"] = 1
                        costs[row, j]["measured_delta_v"] = measured.delta_v_km_s
                        costs[row, j]["measured_mass"] = measured.mass_before_kg
            # Current call owns its packed costs, so clearing cannot invalidate it.
            # Match the scalar path's bounded retained cache across mesh iterations.
            if len(joint._lambert) > 400_000:
                joint._lambert.clear()
        output_rows = 1 if device_selection else count
        result = np.empty(output_rows, RESULT)
        masses, inflations, proxies = (np.empty((output_rows, n - 1)) for _ in range(3))
        payload = np.empty((output_rows, n))
        arrays = (
            policy,
            metadata,
            stages,
            arrivals,
            departures,
            costs,
            result,
            masses,
            inflations,
            proxies,
            payload,
        )
        if resident_geometry:
            selection = np.empty(1, SELECTION) if device_selection else None
            stats = np.zeros(1, GEOMETRY_STATS)
            geometry_arguments = (
                self.handle,
                count if mesh_delta is None else float(mesh_delta),
                *(array.ctypes.data for array in (policy, metadata, stages, arrivals, departures)),
                ct.addressof(elements),
                records.ctypes.data,
                len(records),
                float(minimum_objective) if minimum_objective is not None else 0.0,
                result.ctypes.data,
                selection.ctypes.data if selection is not None else None,
                *(array.ctypes.data for array in (masses, inflations, proxies, payload)),
            )
            if mesh_epochs is not None:
                geometry_arguments += tuple(a.ctypes.data for a in mesh_epochs)
            geometry_arguments += (stats.ctypes.data,)
            if search_config is None:
                self._check(
                    (self.geometry if mesh_delta is None else self.mesh)(*geometry_arguments)
                )
            else:
                days, max_moves, deadline = search_config
                remaining = max(0.0, deadline - time.perf_counter())
                self._check(
                    self.search(
                        self.handle,
                        days.ctypes.data,
                        len(days),
                        max_moves,
                        remaining,
                        *(a.ctypes.data for a in (policy, metadata, stages, arrivals, departures)),
                        ct.addressof(elements),
                        records.ctypes.data,
                        len(records),
                        selection.ctypes.data,
                        *(
                            a.ctypes.data
                            for a in (masses, inflations, proxies, payload, *mesh_epochs)
                        ),
                        stats.ctypes.data,
                        search_report.ctypes.data,
                    )
                )
            if device_selection:
                result = selection["value"]
            computed = int(stats["computed_hops"][0])
            joint.lambert_evaluations += 2 * computed
            if computed:
                self.gpu._record(2 * computed)
            telemetry = self.gpu.telemetry
            for field in ("computed_hops", "cached_hops", "rejected_hops"):
                key = "joint_geometry_" + field
                telemetry[key] = telemetry.get(key, 0) + int(stats[field][0])
            telemetry["completed_element_hops"] = (
                telemetry.get("completed_element_hops", 0) + computed
            )
            telemetry["joint_geometry_stats_download_bytes"] = (
                telemetry.get("joint_geometry_stats_download_bytes", 0) + stats.nbytes
            )
            if mesh_epochs is not None:
                for key, amount in (
                    (
                        "completed_joint_mesh_batches",
                        1 if search_report is None else int(search_report["batches"][0]),
                    ),
                    ("joint_mesh_epoch_upload_bytes", arrivals.nbytes + departures.nbytes),
                    ("joint_mesh_epoch_download_bytes", sum(a.nbytes for a in mesh_epochs)),
                ):
                    telemetry[key] = telemetry.get(key, 0) + amount
        elif device_selection:
            selection = np.empty(1, SELECTION)
            self._check(
                self.best(
                    self.handle,
                    count,
                    *(array.ctypes.data for array in arrays[:6]),
                    float(minimum_objective),
                    selection.ctypes.data,
                    *(array.ctypes.data for array in arrays[7:]),
                )
            )
            result = selection["value"]
        else:
            self._check(self.evaluate(self.handle, count, *(array.ctypes.data for array in arrays)))
        if search_report is not None:
            count = int(search_report["evaluations"][0])
            telemetry["completed_joint_searches"] = telemetry.get("completed_joint_searches", 0) + 1
            telemetry["joint_search_accepted_moves"] = telemetry.get(
                "joint_search_accepted_moves", 0
            ) + int(search_report["moves"][0])
            telemetry["joint_search_report_download_bytes"] = (
                telemetry.get("joint_search_report_download_bytes", 0) + search_report.nbytes
            )
        joint.evaluations += count
        telemetry = self.gpu.telemetry
        telemetry["completed_joint_batches"] = telemetry.get("completed_joint_batches", 0) + 1
        telemetry["completed_joint_evaluations"] = (
            telemetry.get("completed_joint_evaluations", 0) + count
        )
        download_bytes = (SELECTION.itemsize if device_selection else result.nbytes) + sum(
            array.nbytes for array in (masses, inflations, proxies, payload)
        )
        telemetry["joint_result_download_bytes"] = (
            telemetry.get("joint_result_download_bytes", 0) + download_bytes
        )
        # Invalid stays raise even when the offending move cannot win, matching
        # the scalar controller. Preserve first-in-order ties with argmax.
        if (device_selection and selection["invalid_stay"][0]) or np.any(result["failure"] == 17):
            raise ValueError("mining stay must be finite and nonnegative")
        indices = range(count)
        if device_selection:
            winner = int(selection["index"][0])
            if winner < 0:
                return finish((None, None))
            indices = [winner]
        elif minimum_objective is not None:
            eligible = (result["failure"] == 0) & (result["objective"] > minimum_objective + 1e-9)
            if not np.any(eligible):
                return finish((None, None))
            winner = int(np.argmax(np.where(eligible, result["objective"], -np.inf)))
            indices = [winner]
        evaluations = []
        for row in indices:
            detail_row = 0 if device_selection else row
            value = result[detail_row]
            failure = int(value["failure"])
            if failure and failure != 15:
                evaluations.append(joint._fail(FAILURES[failure]))
                continue
            arr, dep = (
                (arrivals[row], departures[row])
                if mesh_epochs is None
                else (mesh_epochs[0][0], mesh_epochs[1][0])
            )
            evaluations.append(
                _decode_evaluation(
                    joint,
                    visits,
                    arr,
                    dep,
                    value,
                    masses[detail_row],
                    inflations[detail_row],
                    proxies[detail_row],
                    payload[detail_row],
                )
            )
        return finish(evaluations if minimum_objective is None else (indices[0], evaluations[0]))


def cuda_joint_enabled():
    """Whether this scope explicitly enabled the native joint evaluator."""
    from .lambert import _GPU_BACKEND

    return (
        os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_BATCH", "0") == "1"
        and _GPU_BACKEND.get() is not None
    )


def cuda_mesh_enabled():
    return (
        cuda_joint_enabled() and os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH") == "1"
    )


def cuda_insertions_enabled():
    """Use native insertion batches when the active core provides them."""
    from .lambert import _GPU_BACKEND

    if not cuda_joint_enabled():
        return False
    requested = os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_INSERTIONS")
    if requested is not None:
        return requested == "1"
    return hasattr(_GPU_BACKEND.get().library, "spacepdhcg_gtoc12_joint_insertions_host")


def cuda_search_enabled():
    return (
        cuda_joint_enabled() and os.environ.get("SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH") == "1"
    )


def evaluate_joint(
    joint,
    visits,
    arrivals,
    departures,
    *,
    minimum_objective=None,
    _mesh_delta=None,
    _search_config=None,
):
    from .jointopt import JointItinerary
    from .lambert import _GPU_BACKEND
    from .retiming import Retimer

    if not cuda_joint_enabled():
        return None
    gpu = _GPU_BACKEND.get()
    for owner, name, expected in (
        (joint, "lambert", JointItinerary.lambert),
        (joint, "_forward", JointItinerary._forward),
        (joint, "_state", JointItinerary._state),
        (joint, "tof_limits", JointItinerary.tof_limits),
        (joint, "dwell_limit", JointItinerary.dwell_limit),
        (joint, "earth_out_below_floor", JointItinerary.earth_out_below_floor),
        (joint.retimer, "leg_inflation", Retimer.leg_inflation),
        (joint.retimer, "_limits", Retimer._limits),
        (joint.retimer, "_ratio_model", Retimer._ratio_model),
        (joint.retimer, "_return_model", Retimer._return_model),
        (joint.retimer, "_modelled", Retimer._modelled),
        (joint.retimer, "search_settings_limits", Retimer.search_settings_limits),
    ):
        if getattr(getattr(owner, name), "__func__", None) is not expected:
            raise ValueError(f"CUDA joint evaluation does not support an overridden {name}")
    if joint.retimer.authority_ratio is not Retimer.authority_ratio:
        raise ValueError("CUDA joint evaluation does not support an overridden authority_ratio")
    arr, dep = (np.ascontiguousarray(values, dtype=np.float64) for values in (arrivals, departures))
    if arr.ndim != 2 or arr.shape[1] != len(visits) or dep.shape != arr.shape or len(visits) < 2:
        raise ValueError("joint batches require matching (candidates, visits) epoch matrices")
    if _mesh_delta is not None and (
        len(arr) != 1 or not math.isfinite(_mesh_delta) or _mesh_delta <= 0
    ):
        raise ValueError("joint mesh requires one incumbent and a finite positive mesh step")
    if not len(arr):
        return [] if minimum_objective is None else (None, None)
    gpu._owned()
    current = gpu.joint_workspace
    required = len(arr) if _mesh_delta is None else 10 * len(visits) - 14
    if current is None or current.capacity < required or current.visits != len(visits):
        if current is not None:
            current.close()
        # moves() emits both signs of two Earth epochs, five moves per inner
        # visit, and one whole-itinerary move. Reserve that neighbourhood even
        # when the first call only evaluates the incumbent.
        capacity = max(required, 2 * (3 + 5 * (len(visits) - 2)))
        gpu.joint_workspace = current = GpuJoint(gpu, capacity, len(visits))
    return current.run(
        joint,
        visits,
        arr,
        dep,
        minimum_objective=minimum_objective,
        mesh_delta=_mesh_delta,
        search_config=_search_config,
    )


def search_epochs(joint, visits, arrivals, departures, mesh, max_moves, deadline):
    """One device-controlled search; only final epochs/report return to Python."""
    from .jointopt import JointItinerary

    if (
        joint.moves is not JointItinerary.moves
        or getattr(joint.evaluate, "__func__", None) is not JointItinerary.evaluate
    ):
        raise ValueError("CUDA joint search does not support overridden moves/evaluation")
    days = np.ascontiguousarray(mesh, dtype=np.float64)
    if days.ndim != 1 or not np.isfinite(days).all() or np.any(days <= 0):
        raise ValueError("joint search mesh must contain finite positive steps")
    if (
        not isinstance(max_moves, (int, np.integer))
        or not 0 <= max_moves <= np.iinfo(np.int32).max
        or len(days) * max_moves > np.iinfo(np.int32).max
    ):
        raise ValueError("joint search move budget must fit nonnegative int32")
    if math.isnan(deadline):
        raise ValueError("joint search deadline must not be NaN")
    output = evaluate_joint(
        joint,
        visits,
        np.asarray(arrivals)[None],
        np.asarray(departures)[None],
        minimum_objective=-math.inf,
        _mesh_delta=1.0,
        _search_config=(days, int(max_moves), deadline),
    )
    if output is None:
        raise RuntimeError("CUDA joint search backend is unavailable")
    selected, arr, dep, report = output
    return arr[0], dep[0], selected[1], int(report["moves"])


def evaluate_mesh(joint, visits, arrivals, departures, delta, minimum_objective):
    """Generate and evaluate one neighbourhood on CUDA; return the winning epochs/plan."""
    from .jointopt import JointItinerary

    if joint.moves is not JointItinerary.moves:
        raise ValueError("CUDA joint mesh does not support an overridden move generator")
    output = evaluate_joint(
        joint,
        visits,
        np.asarray(arrivals)[None],
        np.asarray(departures)[None],
        minimum_objective=minimum_objective,
        _mesh_delta=delta,
    )
    if output is None:
        raise RuntimeError("CUDA joint mesh backend is unavailable")
    selected, arr, dep = output
    return None if selected[0] is None else (arr[0], dep[0], selected[1])


def _decode_evaluation(joint, visits, arr, dep, value, masses, inflations, proxies, payload):
    """Reconstruct a plan only after native evaluation; shared by all batch layouts."""
    from .jointopt import Evaluation
    from .search import PlannedLeg, RoutePlan

    failure = int(value["failure"])
    if failure and failure != 15:
        return joint._fail(FAILURES[failure])
    plan = None
    if not failure:
        deployed, collected, foreign, quantities = {}, {}, {}, {}
        for j, visit in enumerate(visits):
            if visit.deploy:
                deployed[visit.body] = float(arr[j])
            if visit.collect:
                collected[visit.body] = float(dep[j])
                quantities[visit.body] = float(payload[j])
                if visit.body not in deployed:
                    foreign[visit.body] = visit.foreign_deploy_epoch
        legs = []
        for j, (visit, nxt) in enumerate(pairwise(visits)):
            if dep[j] > arr[j] + 1e-9:
                legs.append(
                    PlannedLeg(
                        visit.body,
                        visit.body,
                        float(arr[j]),
                        float(dep[j]),
                        0.0,
                        1.0,
                        "camp",
                    )
                )
            legs.append(
                PlannedLeg(
                    visit.body,
                    nxt.body,
                    float(dep[j]),
                    float(arr[j + 1]),
                    float(proxies[j]),
                    float(inflations[j]),
                    visit.role_out,
                )
            )
        plan = RoutePlan(
            tuple(legs),
            deployed,
            collected,
            quantities,
            float(value["propellant"]),
            float(value["final_mass"]),
            foreign,
        )
    return Evaluation(
        plan,
        float(value["objective"]),
        float(value["weighted"]),
        float(value["collected"]),
        float(value["spare"]),
        float(value["propellant"]),
        masses[: int(value["mass_count"])].tolist(),
        int(value["measured_legs"]),
        FAILURES[failure],
    )
