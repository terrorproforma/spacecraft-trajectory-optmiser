"""Retained CUDA batches for the search's final forward mass calculation.

The explicit completion mode requires a CUDA screening scope and the completion
ABI. Host code packs existing route/table metadata; CUDA prices the flights.
These are proxy plans and still require full low-thrust and fleet verification.
"""

from __future__ import annotations

import ctypes as ct
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np

from . import constants as C
from .screening import exhaust_velocity_km_s

ENVIRONMENT = "SPACEPDHCG_TEST_GTOC12_COMPLETION_BATCH"


def _layout(ints, doubles):
    return np.dtype(
        [(name, np.int32) for name in ints] + [(name, np.float64) for name in doubles],
        align=True,
    )


POLICY = _layout(
    ("abi_version", "sum_mode", "reserved0", "reserved1"),
    (
        "initial_mass",
        "dry_mass",
        "miner_mass",
        "thrust",
        "exhaust",
        "mining_rate",
        "year_days",
        "minimum_stay",
    ),
)
CANDIDATE = _layout(("deploy_begin", "deploy_count", "leg_begin", "leg_count"), ("partial_mass",))
DEPLOY = np.dtype(
    [
        ("deploy_epoch", np.float64),
        ("collect_epoch", np.float64),
        ("has_collect", np.int32),
        ("reserved", np.int32),
    ],
    align=True,
)
LEG = np.dtype(
    [(name, np.int32) for name in ("role", "model", "source_deploy", "reserved")]
    + [(name, np.float64) for name in ("departure", "arrival", "dv", "flat", "floor", "slope")]
    + [("fit", np.float64, (5,))]
    + [(name, np.float64) for name in ("delta_a_au", "delta_longitude_rad", "authority_ratio")],
    align=True,
)
RESULT = _layout(
    ("failure", "failed_leg", "failed_deploy", "processed_legs"),
    ("propellant", "final_mass", "collected", "margin"),
)
LEG_RESULT = _layout(
    ("stage", "pickup"),
    (
        "mass_before",
        "gained",
        "departure_mass",
        "mass_after",
        "authority",
        "inflation",
        "propellant",
        "tof",
    ),
)
STATS = np.dtype(
    [(name, np.float64) for name in ("upload_ms", "kernel_ms", "download_ms")]
    + [(name, np.uint64) for name in ("candidates", "deploy_slots", "leg_slots")],
    align=True,
)
ROLES = {"camp": 0, "earth_out": 1, "deploy_hop": 2, "collect_hop": 3, "earth_return": 4}
FAILURES = (
    "",
    "uncollected",
    "stay_too_short",
    "leg_authority",
    "invalid_inflation",
    "mass_below_dry_plus_collected",
    "invalid_mining_stay",
)


def enabled():
    """The opt-in is independent of library discovery; absence cannot hide it."""
    value = os.environ.get(ENVIRONMENT, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENVIRONMENT} must be 0 or 1")
    return value == "1"


def _require_method(owner, name, expected):
    if getattr(getattr(owner, name), "__func__", None) is not expected:
        raise ValueError(f"CUDA completion does not support an overridden {name}")


def _compatible(search, use_table):
    from .collectdp import CollectPairTable
    from .hopcalib import InflationFit
    from .search import RouteSearch

    # Exact float-sum semantics are part of this adapter's currently tested ABI.
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("CUDA completion currently requires CPython 3.12 sum semantics")
    for name in (
        "_finish",
        "_finish_cpu",
        "_feasible",
        "limits",
        "_propellant",
        "hop_inflation_for",
        "return_inflation_for",
        "_dp_hop_inflation",
        "_plan_from_tour",
    ):
        _require_method(search, name, getattr(RouteSearch, name))
    if use_table:
        table = search.collect_table
        for name in ("return_inflation_at", "return_inflation", "return_override", "pair_geometry"):
            _require_method(table, name, getattr(CollectPairTable, name))
        fit = table.settings.inflation_fit
        if fit is not None:
            _require_method(fit, "inflation", InflationFit.inflation)
            if len(fit.coefficients) != 5:
                raise ValueError("CUDA completion requires the five-feature inflation fit")


def counts(requests):
    sizes = (len(requests), sum(len(r[1]) for r in requests), sum(len(r[3]) for r in requests))
    if any(n > np.iinfo(np.int32).max for n in sizes):
        raise ValueError("CUDA completion batch dimensions must fit int32")
    return sizes


def pack_inputs(search, requests, buffers=None):
    """Pack supported metadata; do not evaluate candidate masses or costs.

    Epochs must be finite, exact Python floats and collection sources must be
    present. Metadata lookup failures are whole-call input errors, distinct from
    the native per-candidate forward gates. The lower-level C ABI also tests
    nonfinite epoch semantics, which this adapter deliberately does not accept.
    """
    _compatible(search, any(row[4] for row in requests))
    for _, deploy, collect, forward, _ in requests:
        epochs = [*deploy.values(), *collect.values()]
        epochs.extend(
            value for leg in forward for value in (leg.departure_epoch, leg.arrival_epoch)
        )
        if any(type(epoch) is not float or not math.isfinite(epoch) for epoch in epochs):
            raise ValueError("CUDA completion metadata requires finite exact Python float epochs")
    n, nd, nl = counts(requests)
    if buffers is None:
        buffers = (
            np.zeros(1, POLICY),
            np.zeros(n, CANDIDATE),
            np.zeros(nd, DEPLOY),
            np.zeros(nl, LEG),
        )
    policy, candidates, deploys, legs = (
        a[:size] for a, size in zip(buffers, (1, n, nd, nl), strict=True)
    )
    for a in (policy, candidates, deploys, legs):
        a.fill(0)
    settings = search.settings
    policy[0] = (
        1,
        1,
        0,
        0,
        settings.initial_mass,
        C.DRY_MASS_KG,
        C.MINER_MASS_KG,
        C.THRUST_MAX_N,
        exhaust_velocity_km_s(),
        C.MINING_RATE_KG_PER_YEAR,
        C.YEAR_DAYS,
        C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS,
    )
    d0 = l0 = 0
    for index, (partial, deploy, collect, forward, use_table) in enumerate(requests):
        candidates[index] = (d0, len(deploy), l0, len(forward), partial.mass)
        positions = {body: j for j, body in enumerate(deploy)}
        for j, (body, epoch) in enumerate(deploy.items()):
            deploys[d0 + j] = (epoch, collect.get(body, 0.0), int(body in collect), 0)
        for j, leg in enumerate(forward):
            record = legs[l0 + j]
            if leg.role not in ROLES:
                raise ValueError(f"CUDA completion does not support role {leg.role!r}")
            record["role"] = ROLES[leg.role]
            record["source_deploy"] = -1
            record["departure"], record["arrival"], record["dv"] = (
                leg.departure_epoch,
                leg.arrival_epoch,
                leg.delta_v_proxy_km_s,
            )
            record["flat"] = leg.inflation
            record["authority_ratio"] = search.limits(leg.role)[1]
            if leg.role in ("collect_hop", "earth_return"):
                if leg.from_id not in positions:
                    raise ValueError(
                        "CUDA completion requires each collection source in the deploy dictionary"
                    )
                record["source_deploy"] = positions[leg.from_id]
            if leg.role == "collect_hop":
                fit = search.collect_table.settings.inflation_fit if use_table else None
                if fit is not None:
                    da, dl = search.collect_table.pair_geometry(
                        leg.from_id, leg.to_id, np.asarray([leg.departure_epoch])
                    )
                    record["model"], record["fit"], record["floor"] = 2, fit.coefficients, fit.floor
                    record["delta_a_au"], record["delta_longitude_rad"] = (
                        float(np.asarray(da).reshape(-1)[0]),
                        float(np.asarray(dl).reshape(-1)[0]),
                    )
                elif settings.hop_inflation_slope is None:
                    record["flat"] = settings.hop_inflation
                else:
                    record["model"], record["floor"], record["slope"] = (
                        1,
                        settings.hop_inflation_floor,
                        settings.hop_inflation_slope,
                    )
            elif leg.role == "earth_return":
                if use_table:
                    table = search.collect_table
                    override = table.return_override(int(leg.from_id))
                    if override is not None:
                        inflation, ok = override
                        k = round(
                            (float(leg.departure_epoch) - table.epochs[0])
                            / table.settings.step_days
                        )
                        t = int(np.argmin(np.abs(table.return_tofs - float(leg.tof_days))))
                        if 0 <= k < inflation.shape[0] and ok[k, t]:
                            record["model"], record["flat"] = 4, inflation[k, t]
                            continue
                    record["model"] = 5 if table.settings.return_tof_model else 0
                    record["flat"] = table.settings.return_inflation
                else:
                    record["model"] = 3 if settings.earth_return_tof_model else 0
                    record["flat"] = settings.earth_return_inflation
        d0 += len(deploy)
        l0 += len(forward)
    return policy, candidates, deploys, legs


class GpuCompletion:
    def __init__(self, gpu, capacity, deploy_capacity, leg_capacity):
        self.gpu = gpu
        self.capacities = (capacity, deploy_capacity, leg_capacity)
        self.handle = ct.c_void_p()
        try:
            self.create = gpu.library.spacepdhcg_gtoc12_completion_create
            self.evaluate = gpu.library.spacepdhcg_gtoc12_completion_evaluate_host
            self.destroy = gpu.library.spacepdhcg_gtoc12_completion_destroy
        except AttributeError as error:
            raise RuntimeError("CUDA completion requires the native completion ABI") from error
        self.create.argtypes = [ct.c_int32] * 4 + [ct.POINTER(ct.c_void_p)]
        self.evaluate.argtypes = [ct.c_void_p] + [ct.c_int32] * 3 + [ct.c_void_p] * 8
        self.destroy.argtypes = [ct.POINTER(ct.c_void_p)]
        for function in (self.create, self.evaluate, self.destroy):
            function.restype = ct.c_int
        self.inputs = (
            np.zeros(1, POLICY),
            np.zeros(capacity, CANDIDATE),
            np.zeros(deploy_capacity, DEPLOY),
            np.zeros(leg_capacity, LEG),
        )
        self.results = np.zeros(capacity, RESULT)
        self.leg_results = np.zeros(leg_capacity, LEG_RESULT)
        self.collected = np.zeros(deploy_capacity)
        self.stats = np.zeros(1, STATS)
        self.native_model = None
        self._check(self.create(gpu.device_id, *self.capacities, ct.byref(self.handle)))

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUDA completion failed (native status {status})")

    def close(self):
        if self.native_model is not None:
            self.native_model.close()
        if self.handle.value:
            self.gpu._owned()
            self._check(self.destroy(ct.byref(self.handle)))

    def run(self, search, requests):
        from .gpu_completion_model import NativeModelOwner, pack_requests
        from .gpu_completion_model import enabled as model_enabled
        from .search import RoutePlan

        self.gpu._owned()
        started = time.perf_counter()
        compact = model_enabled()
        if compact:
            if self.native_model is None:
                self.native_model = NativeModelOwner(self)
            inputs = pack_requests(search, requests, self.native_model.inputs)
            self.native_model.configure(search, requests)
        else:
            inputs = pack_inputs(search, requests, self.inputs)
        n, nd, nl = counts(requests)
        outputs = (self.results[:n], self.leg_results[:nl], self.collected[:nd], self.stats)
        packed = time.perf_counter()
        if compact:
            self.native_model.run(inputs, outputs)
        else:
            self._check(
                self.evaluate(self.handle, n, nd, nl, *(a.ctypes.data for a in (*inputs, *outputs)))
            )
        downloaded = time.perf_counter()
        telemetry = self.gpu.telemetry
        for key, value in {
            "completion_batches": 1,
            "completion_candidates": n,
            "completion_leg_slots": nl,
            "completion_pack_seconds": packed - started,
            "completion_native_call_seconds": downloaded - packed,
            "completion_kernel_seconds": float(self.stats[0]["kernel_ms"]) / 1000.0,
        }.items():
            telemetry[key] = telemetry.get(key, 0) + value
        telemetry["gpu_used"] = True
        plans = []
        l0 = d0 = 0
        recorder = getattr(search, "completion_capture", None)
        for index, (partial, deploy, collect, forward, _) in enumerate(requests):
            value = self.results[index]
            failure = int(value["failure"])
            if failure == 6:
                raise ValueError("stay must be finite and non-negative")
            if not 0 <= failure < len(FAILURES):
                raise RuntimeError("CUDA completion returned an unknown failure")
            if recorder is not None:
                recorder(
                    requests[index],
                    value,
                    self.leg_results[l0 : l0 + len(forward)],
                    self.collected[d0 : d0 + len(deploy)],
                    (
                        inputs[0],
                        inputs[1][index : index + 1],
                        inputs[2][d0 : d0 + len(deploy)],
                        inputs[3][l0 : l0 + len(forward)],
                    ),
                    self.native_model.signature if compact else None,
                )
            if failure:
                plans.append((None, FAILURES[failure]))
            else:
                legs = list(partial.legs)
                cargo = {}
                for j, leg in enumerate(forward):
                    detail = self.leg_results[l0 + j]
                    if detail["pickup"]:
                        cargo[leg.from_id] = float(detail["gained"])
                    legs.append(replace(leg, inflation=float(detail["inflation"])))
                plan = RoutePlan(
                    tuple(legs),
                    deploy,
                    collect,
                    cargo,
                    float(value["propellant"]),
                    float(value["final_mass"]),
                )
                plans.append((plan, ""))
            l0 += len(forward)
            d0 += len(deploy)
        telemetry["completion_total_seconds"] = (
            telemetry.get("completion_total_seconds", 0.0) + time.perf_counter() - started
        )
        return plans


def finish_many(search, requests):
    if not enabled():
        return None
    from .lambert import _GPU_BACKEND

    gpu = _GPU_BACKEND.get()
    if gpu is None:
        raise RuntimeError("CUDA completion requires using_lambert_backend('cuda')")
    gpu._owned()
    if not requests:
        return []
    required = counts(requests)
    current = gpu.completion_workspace
    if current is None or any(a < b for a, b in zip(current.capacities, required, strict=True)):
        previous = current.capacities if current is not None else (0, 0, 0)
        capacity = tuple(
            min(
                np.iinfo(np.int32).max, max(initial, retained, 1 << (max(1, need) - 1).bit_length())
            )
            for initial, retained, need in zip((256, 5120, 5376), previous, required, strict=True)
        )
        replacement = GpuCompletion(gpu, *capacity)
        if current is not None:
            try:
                current.close()
            except BaseException:
                replacement.close()
                raise
        gpu.completion_workspace = current = replacement
    return current.run(search, requests)
